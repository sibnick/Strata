#!/usr/bin/env python3
"""Serial, guarded short K37 gather experiment. No model writes or long requests."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import threading
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
PRIOR = OUT.parent / "2026-10-16-cmp170hx-rtx3090-mmq-640"
REFERENCE = json.loads((PRIOR / "mmq/results.json").read_text())
GIB = 1 << 30
MIB = 1 << 20


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prior = load("gather_prior", PRIOR / "run-short.py")
base = prior.base


def check_pressure(memory, gpus, others):
    if memory["MemAvailable"] < 5 * GIB:
        raise RuntimeError("host below 5 GiB reaction threshold (4 GiB floor + 1 GiB)")
    if any(g["free_bytes"] < 640 * MIB for g in gpus):
        raise RuntimeError("GPU below 640 MiB hard stop")
    if others:
        raise RuntimeError(f"unrelated compute present: {others}")


def bundle(names):
    result = []
    for name in names:
        meta = json.loads((PRIOR / f"prompts/{name}.json").read_text())
        raw = (PRIOR / f"prompts/{name}.ids").read_bytes()
        ids = [int(s) for s in raw.decode().strip().split(",")]
        if hashlib.sha256(raw).hexdigest() != meta["prompt_ids_sha256"]:
            raise RuntimeError("original prompt ID hash mismatch")
        if len(ids) != meta["actual_input_tokens"] or len(ids) > 1977:
            raise RuntimeError("prompt count/bound mismatch")
        refs = [r for r in REFERENCE["requests"] if r["prompt_ids_sha256"] == meta["prompt_ids_sha256"]]
        if not refs or any(r["generated_ids"] != refs[0]["generated_ids"] for r in refs):
            raise RuntimeError("historical exact output is not stable")
        result.append((meta, ids, refs[0]))
    return result


def validate_done(line, ids, generated, reference):
    fields = line.split()
    if len(fields) < 16 or int(fields[1]) != len(generated) or int(fields[2]) != len(ids):
        raise RuntimeError(f"invalid DONE token count: {line}")
    if fields[5] != "stop" or not 0 < len(generated) < 32:
        raise RuntimeError("generation did not complete below cap32")
    if int(fields[8]) != 0 or int(fields[14]) != len(ids):
        raise RuntimeError("input not fully fresh / zero reuse")
    if generated != reference["generated_ids"]:
        raise RuntimeError(f"output IDs differ from historical exact recall: {generated}")
    return {
        "done": line, "input_tokens": len(ids), "generated_ids": generated,
        "answer": reference["answer"], "prefill_ms": float(fields[3]),
        "decode_ms": float(fields[4]), "mtp_accepted_offered": [int(fields[6]), int(fields[7])],
        "fresh": int(fields[14]), "reused": int(fields[8]),
    }


def startup_gate(lines, argv, engine):
    joined = "\n".join(lines)
    for flag, value in (
        ("--gpu", "1,0"), ("--layer-split", "37"), ("--prefill", "256"),
        ("--max-context", "262144"), ("--kv", "int8"),
        ("--vram-reserve-mib", "1024"), ("--vram-reserve-later-mib", "1024"),
        ("--prompt-cache", "0"), ("--spec", "2"),
    ):
        if argv[argv.index(flag) + 1] != value:
            raise RuntimeError(f"configuration mismatch: {flag}")
    for required in (
        "GPU 0: NVIDIA CMP 170HX, compute capability 8.0",
        "CUDA0 owns layers [0,37), 18944 experts, 59192115200 resident bytes",
        "CUDA1 owns layers [37,48), 5632 experts, 17825792000 resident bytes",
        "all 24576 experts are resident in their owning GPU caches",
        "layer split: CUDA0 runs layers 0-36",
        "layer split: CUDA1 runs layers 37-47",
        "INFO logit_bias=1 context=262144 kv=int8",
        "expert_slots=24576", "expert_slots_primary=18944",
        "prompt path allocates its own buffers",
        f"K-quant prompt kernels from {engine.parent / 'libstrata_kq.so'} (STRATA_MMQ_KQ=1)",
    ):
        if required not in joined:
            raise RuntimeError(f"startup admission missing: {required}")
    for line in lines:
        if line.startswith("ERR ") or ("fallback" in line.lower() and "no file fallback is expected" not in line):
            raise RuntimeError(f"startup error/fallback: {line}")


def telemetry():
    query = ("index,uuid,name,power.limit,power.draw,clocks.current.graphics,"
             "clocks.current.sm,temperature.gpu,clocks_event_reasons.active")
    return subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        text=True, timeout=8).strip()


def run(args):
    engine = Path(args.engine).resolve()
    nvml = prior.Nvml()
    if base.compute_processes().strip():
        raise RuntimeError("another compute process is active")
    if base.read_meminfo()["MemAvailable"] < 6 * GIB:
        raise RuntimeError("host startup requires >=6 GiB MemAvailable")
    check_pressure(base.read_meminfo(), nvml.memory(), [])
    gpu_identity = telemetry()
    if "GPU-01f295eb-2ee0-838a-ba5b-db97f075a5f7" not in gpu_identity or \
            "GPU-79d96de3-4783-ba96-d95c-fe5e0364528e" not in gpu_identity:
        raise RuntimeError("physical GPU UUID mismatch")
    argv = list(REFERENCE["argv"])
    argv[0] = str(engine)
    help_text = subprocess.check_output([str(engine), "--help"], text=True, stderr=subprocess.STDOUT, timeout=15)
    source = (ROOT / "src/program/generate.cpp").read_text()
    for flag in (token for token in argv if token.startswith("--")):
        if flag not in source:
            raise RuntimeError(f"exact flag absent from parser: {flag}")
    if "--gpu-owned-experts" not in help_text:
        raise RuntimeError("binary lacks owned mode")
    overrides = dict(REFERENCE["environment_overrides"])
    overrides.pop("STRATA_TRACE", None)
    overrides["STRATA_PREFILL_BATCH_GATHER"] = "1" if args.on else "0"
    if args.counters:
        overrides["STRATA_PREFILL_GATHER_COUNTS"] = "1"
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("STRATA_") and k not in ("CUDA_VISIBLE_DEVICES", "LD_PRELOAD")}
    env.update(overrides)
    artifact_identity = base.prior_artifact_identity()
    prompts = bundle(args.sizes)
    folder = OUT / args.arm
    folder.mkdir(exist_ok=False)
    result = {
        "status": "starting", "argv": argv, "environment_overrides": overrides,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "runtime_diff_sha256": hashlib.sha256(subprocess.check_output(
            ["git", "diff", "HEAD", "--", "src", "include", "CMakeLists.txt"])).hexdigest(),
        "engine_sha256": base.sha256(engine),
        "kquant_library_sha256": base.sha256(engine.parent / "libstrata_kq.so"),
        "cache_sha256": base.sha256(engine.parent / "CMakeCache.txt"),
        "artifact_identity": artifact_identity, "physical_gpu_identity": gpu_identity,
        "guards": {"gpu_stop_mib": 640, "allocation_reserves_mib": [1024, 1024],
                   "host_start_gib": 6, "host_stop_gib": 5, "host_floor_gib": 4,
                   "reaction_headroom_gib": 1, "poll_seconds": 1},
        "requests": [], "started_utc": base.utc_now(),
    }

    def save():
        (folder / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    save()
    process = None
    reader_thread = None
    started = time.monotonic()
    last_sample = -10.
    phase = "startup"
    lines = queue.Queue()
    with (folder / "engine.log").open("w", buffering=1) as log, \
            (folder / "monitor.jsonl").open("w", buffering=1) as monitor:
        def sample(force=False):
            nonlocal last_sample
            now = time.monotonic()
            if not force and now - last_sample < 1:
                return
            gap = now - last_sample if last_sample > 0 else 0
            last_sample = now
            memory, gpus = base.read_meminfo(), nvml.memory()
            others = [row for row in base.compute_processes().splitlines()
                      if int(row.split(",")[0].strip()) != process.pid]
            record = {"utc": base.utc_now(), "elapsed_seconds": now - started, "phase": phase,
                      "sample_gap_seconds": gap, "memory": memory, "gpu_memory_bytes": gpus,
                      "telemetry": telemetry(), "process": base.process_state(process.pid),
                      "other_compute": others}
            monitor.write(json.dumps(record, sort_keys=True) + "\n")
            check_pressure(memory, gpus, others)

        def read_until(predicate, timeout):
            deadline = time.monotonic() + timeout
            captured = []
            while time.monotonic() < deadline:
                sample()
                try:
                    line = lines.get(timeout=.1)
                except queue.Empty:
                    continue
                if line is None:
                    raise RuntimeError("engine exited unexpectedly")
                log.write(line + "\n")
                captured.append(line)
                if line.startswith("ERR "):
                    raise RuntimeError(line)
                if predicate(line):
                    return captured
            raise TimeoutError(f"{phase}: exceeded {timeout}s")

        try:
            process = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1)

            def reader():
                for line in process.stdout:
                    lines.put(line.rstrip("\n"))
                lines.put(None)

            reader_thread = threading.Thread(target=reader, daemon=True)
            reader_thread.start()
            print(f"{base.utc_now()} START {args.arm} pid={process.pid}", flush=True)
            startup = read_until(lambda s: s == "READY 262144 stop", 1200)
            startup_gate(startup, argv, engine)
            loaded = [s.strip().removesuffix(" (deleted)") for s in
                      Path(f"/proc/{process.pid}/maps").read_text().splitlines()
                      if "libstrata_kq.so" in s]
            loaded_paths = sorted({line.split()[-1] for line in loaded})
            if loaded_paths != [str(engine.parent / "libstrata_kq.so")]:
                raise RuntimeError(f"unexpected loaded KQ DSO: {loaded_paths}")
            result["loaded_dso"] = loaded_paths
            result["startup_seconds"] = time.monotonic() - started
            result["ready_memory"] = nvml.memory()
            sample(force=True)
            save()
            schedule = [("warmup", prompts[-1])] + [
                (f"rep{rep}-{p[0]['name']}", p) for rep in range(1, args.reps + 1) for p in prompts
            ]
            for label, (meta, ids, reference) in schedule:
                phase = label
                request_start = time.monotonic()
                process.stdin.write("GEN 32 " + ",".join(map(str, ids)) + "\n")
                process.stdin.flush()
                captured = read_until(lambda s: s.startswith("DONE "), 120)
                wall = time.monotonic() - request_start
                generated = [int(s.split()[1]) for s in captured if s.startswith("T ")]
                data = validate_done(captured[-1], ids, generated, reference)
                counters = read_until(lambda s: "file tier I/O this request:" in s, 30)
                if label != "warmup" and not counters[-1].endswith("for 0.0 MB of expert reads"):
                    raise RuntimeError(f"unexpected expert rereads: {counters[-1]}")
                data.update({
                    "name": label, "wall_seconds_to_done": wall,
                    "prompt_ids_sha256": meta["prompt_ids_sha256"], "counter_lines": counters,
                    "gather_counts": [s for s in captured if s.startswith("strata gather counts:")],
                    "memory_after": nvml.memory(),
                })
                result["requests"].append(data)
                sample(force=True)
                save()
                print(f"{base.utc_now()} PASS {args.arm} {label} prefill={data['prefill_ms']}ms", flush=True)
            result["status"] = "passed"
        except BaseException as exc:
            result["status"] = "failed"
            result["error"] = f"{type(exc).__name__}: {exc}"
            save()
            raise
        finally:
            if process is not None:
                if result["status"] != "passed" and process.poll() is None:
                    base.safe_stop(process, result.get("error", "arm failed"))
                elif process.poll() is None:
                    process.stdin.close()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        base.safe_stop(process, "EOF timeout")
                if reader_thread:
                    reader_thread.join(timeout=5)
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
    if result["status"] != "passed" or result["compute_processes_after"].strip():
        raise RuntimeError("arm/cleanup failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--on", action="store_true")
    parser.add_argument("--counters", action="store_true")
    parser.add_argument("--sizes", nargs="+", choices=("small", "medium", "large"), default=["large"])
    parser.add_argument("--reps", type=int, choices=(1, 3), default=3)
    run(parser.parse_args())
