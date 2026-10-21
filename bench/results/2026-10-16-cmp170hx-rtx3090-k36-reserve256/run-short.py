#!/usr/bin/env python3
"""Short K36 MMQ: RTX reserve/monitor256 MiB; CMP reserve1024/monitor640."""
import argparse
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time


OUT = Path(__file__).resolve().parent
PRIOR = OUT.parent / "2026-10-16-cmp170hx-rtx3090-short-profile"
spec = importlib.util.spec_from_file_location("short_profile", PRIOR / "profile-short.py")
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)
base = profile.base
ROOT = profile.ROOT
GPU_STOP_MIB = {0: 256, 1: 640}
GIB = 1 << 30


class Memory(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in ("total", "free", "used")]


class Nvml:
    def __init__(self):
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        self.check(self.lib.nvmlInit_v2())
        self.handles = []
        for index in (0, 1):
            handle = ctypes.c_void_p()
            self.check(self.lib.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle)))
            self.handles.append(handle)

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f"NVML error {code}")

    def memory(self):
        result = []
        for index, handle in enumerate(self.handles):
            memory = Memory()
            self.check(self.lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)))
            result.append({"physical_index": index, **{
                name + "_bytes": getattr(memory, name) for name in ("total", "free", "used")
            }})
        return result


def check_pressure(memory, gpus, others):
    if memory["MemAvailable"] < 5 * GIB:
        raise RuntimeError("host below 5 GiB reaction threshold (4 GiB floor plus 1 GiB)")
    for g in gpus:
        threshold = GPU_STOP_MIB[g["physical_index"]]
        if g["free_bytes"] < threshold * (1 << 20):
            raise RuntimeError(f"GPU {g['physical_index']} below {threshold} MiB monitor guard")
    if others:
        raise RuntimeError(f"other compute process appeared: {others}")


def process_state(process):
    try:
        return base.process_state(process.pid)
    except (PermissionError, FileNotFoundError):
        # /proc/<pid>/io can become unreadable as the engine exits.
        returncode = process.poll()
        if returncode is None:
            raise
        return {"exited_return_code": returncode}


def run(mode):
    nvml = Nvml()
    profile.OUT = OUT
    bundle = profile.prompts()
    tokenizer = profile.prepare.load_tokenizer()
    baseline = json.loads((PRIOR / "control/results.json").read_text())
    if base.compute_processes().strip():
        raise RuntimeError("another GPU compute process active")
    if base.read_meminfo()["MemAvailable"] < 6 * GIB:
        raise RuntimeError("host start floor 6 GiB not met")
    check_pressure(base.read_meminfo(), nvml.memory(), [])
    folder = OUT / mode
    folder.mkdir(exist_ok=False)
    argv = baseline["argv"].copy()
    argv[argv.index("--layer-split") + 1] = "36"
    argv[argv.index("--vram-reserve-later-mib") + 1] = "256"
    overrides = dict(baseline["environment_overrides"])
    if mode == "mmq":
        overrides["STRATA_MMQ_KQ"] = "1"
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("STRATA_") and k != "CUDA_VISIBLE_DEVICES"}
    env.update(overrides)
    result = {
        "status": "starting", "argv": argv, "environment_overrides": overrides,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "engine_sha256": base.sha256(Path(argv[0])),
        "kquant_library_sha256": base.sha256(Path(argv[0]).parent / "libstrata_kq.so"),
        "cache_sha256": base.sha256(Path(argv[0]).parent / "CMakeCache.txt"),
        "artifact_identity": base.prior_artifact_identity(),
        "guards": {"gpu_stop_mib_physical": GPU_STOP_MIB, "host_start_gib": 6, "host_stop_gib": 5,
                   "host_floor_gib": 4, "reaction_headroom_gib": 1, "poll_seconds": 1},
        "authorization": "User requests K36 retry after discussing RTX allocation reserve256 MiB; RTX monitor256, CMP reserve1024/monitor640, host unchanged.",
        "before_memory": nvml.memory(), "started_utc": base.utc_now(), "requests": [],
    }
    def save():
        (folder / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    save()
    started = time.monotonic()
    process = None
    phase = "startup"
    last_sample = -10.0
    lines = queue.Queue()
    with (folder / "engine.log").open("w", buffering=1) as log, \
            (folder / "monitor.jsonl").open("w", buffering=1) as monitor:
        def sample(force=False):
            nonlocal last_sample
            now = time.monotonic()
            if not force and now - last_sample < 1:
                return
            last_sample = now
            memory, gpus = base.read_meminfo(), nvml.memory()
            others = [row for row in base.compute_processes().splitlines()
                      if int(row.split(",")[0].strip()) != process.pid]
            monitor.write(json.dumps({
                "utc": base.utc_now(), "elapsed_seconds": now - started, "phase": phase,
                "memory": memory, "gpu_memory_bytes": gpus, "gpus": base.gpu_state(),
                "process": process_state(process), "other_compute": others,
            }, sort_keys=True) + "\n")
            check_pressure(memory, gpus, others)

        def read_until(predicate, timeout, on_line=None):
            deadline = time.monotonic() + timeout
            captured = []
            while time.monotonic() < deadline:
                sample()
                try:
                    line = lines.get(timeout=0.1)
                except queue.Empty:
                    continue
                if line is None:
                    raise RuntimeError("engine exited unexpectedly")
                log.write(line + "\n")
                captured.append(line)
                if on_line:
                    on_line(line)
                if line.startswith("ERR "):
                    raise RuntimeError(line)
                if predicate(line):
                    return captured
            raise TimeoutError(f"{phase} exceeded {timeout}s")

        try:
            process = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1)
            def reader():
                for line in process.stdout:
                    lines.put(line.rstrip("\n"))
                lines.put(None)
            thread = threading.Thread(target=reader, daemon=True)
            thread.start()
            print(f"{base.utc_now()} START {mode} pid={process.pid}", flush=True)
            startup = read_until(lambda s: s == "READY 262144 stop", 1200)
            if not any("all 24576 experts are resident" in line for line in startup):
                raise RuntimeError("full owned residency not confirmed")
            if mode == "mmq" and not any("K-quant prompt kernels from" in s for s in startup):
                raise RuntimeError("MMQ library load not confirmed")
            result["startup_seconds"] = time.monotonic() - started
            result["ready_memory"] = nvml.memory()
            result["status"] = "running"
            save()
            print(f"{base.utc_now()} READY {mode} startup={result['startup_seconds']:.3f}s", flush=True)
            schedule = [("warmup", bundle[-1])] + [
                (f"rep{rep}-{meta['name']}", (meta, ids))
                for rep in range(1, 4) for meta, ids in bundle
            ]
            for label, (meta, ids) in schedule:
                phase = label
                request_start = time.monotonic()
                first_token = []
                def observe(line):
                    if line.startswith("T ") and not first_token:
                        first_token.append(time.monotonic() - request_start)
                process.stdin.write("GEN 32 " + ",".join(map(str, ids)) + "\n")
                process.stdin.flush()
                captured = read_until(lambda s: s.startswith("DONE "), 120, observe)
                wall = time.monotonic() - request_start
                generated = [int(s.split()[1]) for s in captured if s.startswith("T ")]
                data = profile.validate_done(captured[-1], ids, generated, tokenizer)
                counters = read_until(lambda s: "file tier I/O this request:" in s, 30)
                # The first line includes startup reads; subsequent lines must be zero deltas.
                if label != "warmup" and not counters[-1].endswith("for 0.0 MB of expert reads"):
                    raise RuntimeError(f"request expert rereads: {counters[-1]}")
                data.update({
                    "name": label, "wall_seconds_to_done": wall,
                    "ttft_seconds_observed": first_token[0] if first_token else None,
                    "prompt_ids_sha256": meta["prompt_ids_sha256"], "counter_lines": counters,
                    "memory_after": nvml.memory(),
                })
                result["requests"].append(data)
                sample(force=True)
                save()
                print(f"{base.utc_now()} PASS {mode} {label} prefill={data['prefill_ms']}ms wall={wall:.4f}s", flush=True)
            result["status"] = "passed"
        except BaseException as exc:
            result["status"] = "failed"
            result["error"] = f"{type(exc).__name__}: {exc}"
            save()
            raise
        finally:
            if process is not None:
                if result["status"] != "passed" and process.poll() is None:
                    base.safe_stop(process, result.get("error", "short run failed"))
                elif process.poll() is None:
                    process.stdin.close()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        base.safe_stop(process, "EOF timeout")
                thread.join(timeout=5)
                while not lines.empty():
                    line = lines.get_nowait()
                    if line is not None:
                        log.write(line + "\n")
                result["engine_return_code"] = process.returncode
                if process.returncode != 0:
                    result["status"] = "failed"
            result["finished_utc"] = base.utc_now()
            result["elapsed_seconds"] = time.monotonic() - started
            result["after_memory"] = nvml.memory()
            result["compute_processes_after"] = base.compute_processes()
            save()
    if result["status"] != "passed":
        raise RuntimeError("short comparison arm did not pass")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("mmq",), required=True)
    run(parser.parse_args().mode)
