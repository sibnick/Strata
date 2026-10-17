#!/usr/bin/env python3
"""Two controlled 32K checks, then exactly one gated batching-ON near-limit request."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SHORT = OUT.parent / "2026-10-17-cmp170hx-rtx3090-k37-mmq-batched-gather"
PRIOR = OUT.parent / "2026-10-16-cmp170hx-rtx3090-k37-mmq-long-context"
HISTORICAL = json.loads((PRIOR / "mmq/results.json").read_text())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


short = load("batched_long_short", SHORT / "run.py")
long = load("batched_long_prior", PRIOR / "run-long.py")
base = short.base
ENGINE = ROOT / "build/cuda-batched/strata"
EXPECTED_BUILD = json.loads((SHORT / "build-identity.json").read_text())["candidate_hashes"]


def fixtures():
    manifest = json.loads((PRIOR / "prompt-references.json").read_text())
    for item in manifest["files"]:
        path = OUT.parent / item["path_relative_to_results"]
        if path.stat().st_size != item["bytes"] or base.sha256(path) != item["sha256"]:
            raise RuntimeError(f"immutable original fixture changed: {path}")
    result = {}
    meta, ids, reference = short.bundle(["large"])[0]
    result["warmup"] = (meta, ids, reference)
    for name, count in (("32k", 32788), ("near-max", 261682)):
        meta, ids = base.load_bundle(name)
        reference = next(r for r in HISTORICAL["requests"] if r["name"] == name)
        if len(ids) != count or meta["generation_cap"] != 192 or len(ids) + 192 > 262144:
            raise RuntimeError("original fixture count/cap/headroom mismatch")
        if meta["prompt_ids_sha256"] != reference["prompt_ids_sha256"]:
            raise RuntimeError("historical prompt identity mismatch")
        result[name] = (meta, ids, reference)
    return result


def validate(line, ids, generated, tokenizer, meta, reference):
    result = long.validate_done(line, ids, generated, tokenizer, meta)
    if generated != reference["generated_ids"]:
        raise RuntimeError("output IDs differ from historical exact recall")
    return result


def off_gate(off, argv, overrides):
    if off["status"] != "passed" or off["engine_return_code"] != 0 or off["compute_processes_after"]:
        raise RuntimeError("OFF sanity did not pass/clean up")
    if [r["name"] for r in off["requests"]] != ["warmup", "32k"]:
        raise RuntimeError("OFF schedule mismatch")
    if off["argv"] != argv:
        raise RuntimeError("OFF/ON binary or flags differ")
    on_env = dict(overrides)
    on_env["STRATA_PREFILL_BATCH_GATHER"] = "0"
    if off["environment_overrides"] != on_env:
        raise RuntimeError("OFF/ON environment differs beyond batching")
    for request in off["requests"]:
        if request["reused"] != 0 or request["fresh"] != request["input_tokens"]:
            raise RuntimeError("OFF sanity freshness gate failed")


def verify_identity():
    for name, expected in EXPECTED_BUILD.items():
        path = ROOT / name
        if base.sha256(path) != expected:
            raise RuntimeError(f"unchanged candidate identity mismatch: {path}")
    if subprocess.check_output(["git", "diff", "97b83332", "--", "src", "include", "CMakeLists.txt"], cwd=ROOT):
        raise RuntimeError("runtime source changed after validated batching commit")


def run(arm):
    verify_identity()
    nvml = short.prior.Nvml()
    if base.compute_processes().strip():
        raise RuntimeError("another compute process is active")
    if base.read_meminfo()["MemAvailable"] < 6 * short.GIB:
        raise RuntimeError("host startup below 6 GiB")
    short.check_pressure(base.read_meminfo(), nvml.memory(), [])
    identity = short.telemetry()
    for uuid in ("GPU-01f295eb-2ee0-838a-ba5b-db97f075a5f7", "GPU-79d96de3-4783-ba96-d95c-fe5e0364528e"):
        if uuid not in identity:
            raise RuntimeError("physical GPU identity mismatch")
    argv = list(short.REFERENCE["argv"])
    argv[0] = str(ENGINE)
    overrides = dict(short.REFERENCE["environment_overrides"])
    overrides.pop("STRATA_TRACE", None)
    overrides["STRATA_PREFILL_BATCH_GATHER"] = "1" if arm == "on" else "0"
    if any(k in overrides for k in ("STRATA_PREFILL_TIMING", "STRATA_PREFILL_GATHER_COUNTS", "STRATA_TRACE")):
        raise RuntimeError("instrumented timing is forbidden")
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("STRATA_") and k not in ("CUDA_VISIBLE_DEVICES", "LD_PRELOAD")}
    env.update(overrides)
    bundle = fixtures()
    tokenizer = long.profile.prepare.load_tokenizer()
    off = None
    if arm == "on":
        off = json.loads((OUT / "off/results.json").read_text())
        off_gate(off, argv, overrides)
    folder = OUT / arm
    folder.mkdir(exist_ok=False)
    result = {
        "status": "starting", "arm": arm, "argv": argv, "environment_overrides": overrides,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "engine_sha256": base.sha256(ENGINE), "kquant_library_sha256": base.sha256(ENGINE.parent / "libstrata_kq.so"),
        "cache_sha256": base.sha256(ENGINE.parent / "CMakeCache.txt"),
        "artifact_identity": base.prior_artifact_identity(), "physical_gpu_identity": identity,
        "harness_sha256": base.sha256(Path(__file__)),
        "guards": {"gpu_stop_mib": 640, "allocation_reserves_mib": [1024, 1024],
                   "host_start_gib": 6, "host_stop_gib": 5, "host_floor_gib": 4,
                   "reaction_headroom_gib": 1, "poll_seconds": 1},
        "started_utc": base.utc_now(), "requests": [],
    }

    def save():
        (folder / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    save()
    process = reader_thread = pcie = pcie_thread = None
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
            monitor.write(json.dumps({
                "utc": base.utc_now(), "elapsed_seconds": now - started, "phase": phase,
                "sample_gap_seconds": gap, "memory": memory, "gpu_memory_bytes": gpus,
                "telemetry": short.telemetry(), "process": long.process_state(process),
                "other_compute": others,
            }, sort_keys=True) + "\n")
            short.check_pressure(memory, gpus, others)
            if pcie.poll() is not None:
                raise RuntimeError("PCIe telemetry helper exited unexpectedly")

        def read_until(predicate, timeout, observe=None):
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
                if observe:
                    observe(line)
                if line.startswith("ERR "):
                    raise RuntimeError(line)
                if predicate(line):
                    return captured
            raise TimeoutError(f"{phase} exceeded generous {timeout}s bound")

        try:
            pcie, pcie_thread, error = base.start_pcie_logger(folder / "pcie.log")
            if error or pcie is None:
                raise RuntimeError(f"PCIe telemetry start failed: {error}")
            result["pcie_helper_pid"] = pcie.pid
            process = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

            def reader():
                for line in process.stdout:
                    lines.put(line.rstrip("\n"))
                lines.put(None)

            reader_thread = threading.Thread(target=reader, daemon=True)
            reader_thread.start()
            result["engine_pid"] = process.pid
            print(f"{base.utc_now()} START {arm} engine={process.pid} pcie={pcie.pid}", flush=True)
            startup = read_until(lambda s: s == "READY 262144 stop", 1200)
            short.startup_gate(startup, argv, ENGINE)
            loaded = sorted({s.split()[-1] for s in Path(f"/proc/{process.pid}/maps").read_text().splitlines()
                             if "libstrata_kq.so" in s})
            if loaded != [str(ENGINE.parent / "libstrata_kq.so")]:
                raise RuntimeError(f"unexpected loaded DSO: {loaded}")
            result["loaded_dso"] = loaded
            result["pcie_ready"] = base.pcie_summary(folder / "pcie.log", time.time() - 10, time.time())
            if not result["pcie_ready"]["available"]:
                raise RuntimeError("PCIe helper not responsive/parseable")
            result["startup_seconds"] = time.monotonic() - started
            result["status"] = "running"
            sample(force=True)
            save()
            print(f"{base.utc_now()} READY {arm} startup={result['startup_seconds']:.3f}s", flush=True)
            schedule = ["warmup", "32k"] + (["near-max"] if arm == "on" else [])
            preceding_rate = 766.854
            for label in schedule:
                if label == "near-max":
                    if off is None or [r["name"] for r in result["requests"]] != ["warmup", "32k"]:
                        raise RuntimeError("both sanity requests must pass before near-limit")
                    sanity = result["requests"][-1]
                    if sanity["generated_ids"] != off["requests"][-1]["generated_ids"]:
                        raise RuntimeError("32K controlled output parity failed")
                    verify_identity()
                    result["near_limit_gate"] = {"off_status": off["status"], "on_32k_exact_parity": True,
                                                "off_results_sha256": base.sha256(OUT / "off/results.json")}
                meta, ids, reference = bundle[label]
                phase = label
                request_start, request_epoch = time.monotonic(), time.time()
                timeout = long.request_timeout(len(ids), preceding_rate)
                result["active_request"] = {"name": label, "input_tokens": len(ids),
                                            "timeout_seconds": timeout, "preceding_rate": preceding_rate,
                                            "started_utc": base.utc_now()}
                save()
                print(f"{base.utc_now()} REQUEST {arm} {label} input={len(ids)} timeout={timeout:.1f}s", flush=True)
                first_token, last_report, write_errors = [], [-30.], []

                def observe(line):
                    elapsed = time.monotonic() - request_start
                    if line.startswith("T ") and not first_token:
                        first_token.append(elapsed)
                    if line.startswith("PP ") and elapsed - last_report[0] >= 30:
                        last_report[0] = elapsed
                        result["active_request"]["last_checkpoint"] = line
                        save()
                        print(f"{base.utc_now()} PROGRESS {arm} {label} {line}", flush=True)

                def write_request():
                    try:
                        process.stdin.write(f"GEN {meta['generation_cap']} " + ",".join(map(str, ids)) + "\n")
                        process.stdin.flush()
                    except (OSError, ValueError) as exc:
                        write_errors.append(exc)

                writer = threading.Thread(target=write_request, daemon=True)
                writer.start()
                captured = read_until(lambda s: s.startswith("DONE "), timeout, observe)
                wall = time.monotonic() - request_start
                writer.join(timeout=5)
                if write_errors or writer.is_alive():
                    raise RuntimeError(f"request submission failed: {write_errors}")
                generated = [int(s.split()[1]) for s in captured if s.startswith("T ")]
                data = validate(captured[-1], ids, generated, tokenizer, meta, reference)
                if arm == "on" and label == "warmup":
                    if not any("batched gather enabled for routed native MMQ experts" in s for s in captured):
                        raise RuntimeError("actual batched-gather opt-in not confirmed")
                    result["batch_opt_in_confirmed"] = True
                if any(s.startswith(("strata prefill timing:", "strata gather counts:")) for s in captured):
                    raise RuntimeError("forbidden profiling/counters observed")
                counters = read_until(lambda s: "file tier I/O this request:" in s, 30)
                if label != "warmup" and not counters[-1].endswith("for 0.0 MB of expert reads"):
                    raise RuntimeError(f"unexpected expert rereads: {counters[-1]}")
                data.update({
                    "name": label, "wall_seconds_to_done": wall,
                    "ttft_seconds_observed": first_token[0] if first_token else None,
                    "prompt_ids_sha256": meta["prompt_ids_sha256"], "prompt_metadata": meta,
                    "counter_lines": counters, "timeout_seconds": timeout,
                    "pcie": base.pcie_summary(folder / "pcie.log", request_epoch, time.time()),
                    "memory_after": nvml.memory(),
                })
                result["requests"].append(data)
                result.pop("active_request", None)
                preceding_rate = data["prefill_tokens_per_second"]
                sample(force=True)
                save()
                print(f"{base.utc_now()} PASS {arm} {label} prefill={data['prefill_ms']}ms "
                      f"ttft={data['ttft_seconds_observed']:.3f}s wall={wall:.3f}s", flush=True)
            result["status"] = "passed"
        except BaseException as exc:
            result["status"] = "failed"
            result["error"] = f"{type(exc).__name__}: {exc}"
            save()
            raise
        finally:
            if process is not None:
                if result["status"] != "passed" and process.poll() is None:
                    base.safe_stop(process, result.get("error", "long validation failed"))
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
            if pcie is not None:
                base.safe_stop(pcie, "telemetry complete")
                pcie_thread.join(timeout=5)
                result["pcie_helper_return_code"] = pcie.poll()
            result["finished_utc"] = base.utc_now()
            result["after_memory"] = nvml.memory()
            result["compute_processes_after"] = base.compute_processes()
            save()
    if result["status"] != "passed" or result["compute_processes_after"].strip():
        raise RuntimeError("arm or cleanup failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("off", "on"), required=True)
    run(parser.parse_args().arm)
