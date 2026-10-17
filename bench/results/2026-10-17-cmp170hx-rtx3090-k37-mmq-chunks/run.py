#!/usr/bin/env python3
"""Bounded serial chunk experiment, using unchanged K37 binary, artifacts and guards."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import queue
import re
import statistics
import subprocess
import threading
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
spec = importlib.util.spec_from_file_location(
    "chunk_prior", OUT.parent / "2026-10-17-cmp170hx-rtx3090-k37-mmq-batched-long/run.py")
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
short, base, long = prior.short, prior.base, prior.long
REFERENCE = json.loads((prior.OUT / "on/results.json").read_text())
ENGINE = Path(REFERENCE["argv"][0])


def identity():
    for relative, expected in prior.EXPECTED_BUILD.items():
        path = ENGINE.parent / Path(relative).name if relative.startswith("build/") else ROOT / relative
        if base.sha256(path) != expected:
            raise RuntimeError(f"validated identity mismatch: {path}")
    if subprocess.check_output(["git", "diff", "97b83332", "--", "src", "include", "CMakeLists.txt"]):
        raise RuntimeError("runtime changed")


def shapes(lines, tokens, chunk):
    positions = [int(s.split()[1]) for s in lines if s.startswith("PP ")]
    expected = list(range(chunk, tokens - 1, chunk)) + [tokens - 1]
    if positions != expected:
        raise RuntimeError(f"actual prompt chunks differ: {positions} != {expected}")
    return [b - a for a, b in zip([0] + positions[:-1], positions)]


def startup_gate(lines, argv, chunk):
    normalized = list(argv)
    normalized[normalized.index("--prefill") + 1] = "256"
    short.startup_gate(lines, normalized, ENGINE)
    joined = "\n".join(lines)
    size = json.loads((OUT / "offline.json").read_text())["k37"][str(chunk)]
    scratch = {stage: value["owned_prefill_allocation_estimate"] for stage, value in size.items()}
    if f"own buffers for a {chunk}-token chunk: {scratch['0'] >> 20} MiB" not in joined:
        raise RuntimeError("actual configured chunk / exact primary scratch not confirmed")
    if any("trying a " in s or "token chunks" in s for s in lines):
        raise RuntimeError("scheduler clamped requested chunk")
    correction = re.search(r"CUDA1 adds (\d+) bytes for exact owned-prefill scratch", joined)
    estimated = 160 + chunk * 680 // 1024
    if correction is None or int(correction[1]) != max(0, scratch["1"] + 64 * short.MIB - estimated * short.MIB):
        raise RuntimeError("stage-specific scratch pricing mismatch")
    return {"requested_chunk": chunk, "configured_chunk": chunk,
            "owned_prefill_bytes_per_stage": scratch, "startup_lines": lines}


def fixture(name):
    if name == "short":
        return short.bundle(["large"])[0]
    meta, ids = base.load_bundle("32k")
    ref = next(r for r in prior.HISTORICAL["requests"] if r["name"] == "32k")
    if len(ids) != 32788 or meta["generation_cap"] != 192 or meta["prompt_ids_sha256"] != ref["prompt_ids_sha256"]:
        raise RuntimeError("immutable 32K fixture mismatch")
    return meta, ids, ref


def validate(lines, meta, ids, ref, chunk, tokenizer):
    generated = [int(s.split()[1]) for s in lines if s.startswith("T ")]
    if len(ids) == 1977:
        data = short.validate_done(lines[-1], ids, generated, ref)
        data["prefill_tokens_per_second"] = len(ids) * 1000 / data["prefill_ms"]
    else:
        data = prior.validate(lines[-1], ids, generated, tokenizer, meta, ref)
    data["actual_chunk_shape"] = shapes(lines, len(ids), chunk)
    data["actual_chunk_count"] = len(data["actual_chunk_shape"])
    if any(s.startswith(("strata prefill timing:", "strata gather counts:")) for s in lines):
        raise RuntimeError("forbidden profiler / diagnostic counters")
    return data


def run(chunk, bracket=False):
    if bracket and chunk != 256:
        raise ValueError("the one drift bracket is baseline256 only")
    identity()
    folder = OUT / (f"chunk{chunk}-bracket" if bracket else f"chunk{chunk}")
    folder.mkdir(exist_ok=False)
    nvml = short.prior.Nvml()
    memory = base.read_meminfo()
    if base.compute_processes():
        raise RuntimeError("unrelated compute before startup")
    if memory["MemAvailable"] < 6 * short.GIB:
        raise RuntimeError("host startup below 6 GiB")
    short.check_pressure(memory, nvml.memory(), [])
    sizing = json.loads((OUT / "offline.json").read_text())["k37"][str(chunk)]
    admission = {}
    for stage, physical in (("0", 1), ("1", 0)):
        need = sizing[stage]["required_free_before_model_load"]
        available = nvml.memory()[physical]["free_bytes"]
        admission[stage] = {"physical_gpu": physical, "free_bytes": available,
                            "ledger_required_including_reserves": need, "margin_bytes": available - need}
        if available < need:
            (folder / "results.json").write_text(json.dumps({
                "status": "offline-infeasible", "admission": admission}, indent=2) + "\n")
            return
    argv = list(REFERENCE["argv"])
    argv[argv.index("--prefill") + 1] = str(chunk)
    overrides = dict(REFERENCE["environment_overrides"])
    env = {k: v for k, v in os.environ.items() if not k.startswith("STRATA_")
           and k not in ("CUDA_VISIBLE_DEVICES", "LD_PRELOAD")}
    env.update(overrides)
    if any(k in env for k in ("STRATA_TRACE", "STRATA_PREFILL_TIMING", "STRATA_PREFILL_GATHER_COUNTS")):
        raise RuntimeError("instrumented timing forbidden")
    gpu_identity = short.telemetry()
    for uuid in ("GPU-01f295eb-2ee0-838a-ba5b-db97f075a5f7", "GPU-79d96de3-4783-ba96-d95c-fe5e0364528e"):
        if uuid not in gpu_identity:
            raise RuntimeError("physical GPU identity mismatch")
    result = {
        "status": "starting", "chunk": chunk, "bracket": bracket, "argv": argv, "environment_overrides": overrides,
        "admission": admission, "physical_gpu_identity": gpu_identity,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "engine_sha256": base.sha256(ENGINE),
        "kquant_library_sha256": base.sha256(ENGINE.parent / "libstrata_kq.so"),
        "cache_sha256": base.sha256(ENGINE.parent / "CMakeCache.txt"),
        "artifact_identity": base.prior_artifact_identity(),
        "harness_sha256": base.sha256(Path(__file__)), "started_utc": base.utc_now(),
        "guards": {"gpu_stop_mib": [640, 640], "engine_reserve_mib": [1024, 1024],
                   "host_start_gib": 6, "host_stop_gib": 5, "host_floor_gib": 4,
                   "poll_seconds": 1, "polling_not_oom_guarantee": True}, "requests": [],
    }

    def save():
        (folder / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    save()
    process = reader_thread = pcie = pcie_thread = None
    lines = queue.Queue()
    started, last_sample, phase = time.monotonic(), 0., "startup"
    tokenizer = long.profile.prepare.load_tokenizer()
    with (folder / "engine.log").open("w", buffering=1) as log, \
            (folder / "monitor.jsonl").open("w", buffering=1) as monitor:
        def sample(force=False):
            nonlocal last_sample
            now = time.monotonic()
            if not force and now - last_sample < 1:
                return
            gap = now - last_sample if last_sample else 0
            last_sample = now
            mem, gpus = base.read_meminfo(), nvml.memory()
            others = [s for s in base.compute_processes().splitlines()
                      if process is None or int(s.split(",")[0].strip()) != process.pid]
            monitor.write(json.dumps({
                "utc": base.utc_now(), "elapsed_seconds": now - started, "phase": phase,
                "sample_gap_seconds": gap, "memory": mem, "gpu_memory_bytes": gpus,
                "telemetry": short.telemetry(), "process": long.process_state(process) if process else {},
                "other_compute": others}, sort_keys=True) + "\n")
            short.check_pressure(mem, gpus, others)
            if pcie is not None and pcie.poll() is not None:
                raise RuntimeError("PCIe helper exited")

        def read_until(predicate, timeout):
            deadline, captured = time.monotonic() + timeout, []
            while time.monotonic() < deadline:
                sample()
                try:
                    line = lines.get(timeout=.1)
                except queue.Empty:
                    continue
                if line is None:
                    raise RuntimeError("engine exited before expected response")
                log.write(line + "\n")
                captured.append(line)
                if line.startswith("ERR "):
                    raise RuntimeError(line)
                if predicate(line):
                    return captured
            raise TimeoutError(f"{phase} exceeded {timeout}s")

        try:
            pcie, pcie_thread, error = base.start_pcie_logger(folder / "pcie.log")
            if error or pcie is None:
                raise RuntimeError(f"PCIe logger failed: {error}")
            result["pcie_helper_pid"] = pcie.pid
            process = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            result["engine_pid"] = process.pid

            def reader():
                for line in process.stdout:
                    lines.put(line.rstrip("\n"))
                lines.put(None)

            reader_thread = threading.Thread(target=reader, daemon=True)
            reader_thread.start()
            print(f"{base.utc_now()} START chunk{chunk} pid={process.pid}", flush=True)
            startup = read_until(lambda s: s == "READY 262144 stop", 1200)
            result["startup_gate"] = startup_gate(startup, argv, chunk)
            loaded = sorted({s.split()[-1] for s in Path(f"/proc/{process.pid}/maps").read_text().splitlines()
                             if "libstrata_kq.so" in s})
            if loaded != [str(ENGINE.parent / "libstrata_kq.so")]:
                raise RuntimeError(f"unexpected loaded DSO: {loaded}")
            result["loaded_dso"] = loaded
            result["pcie_ready"] = base.pcie_summary(folder / "pcie.log", time.time() - 10, time.time())
            if not result["pcie_ready"]["available"]:
                raise RuntimeError("PCIe logger not responsive")
            result["startup_seconds"] = time.monotonic() - started
            save()
            print(f"{base.utc_now()} READY chunk{chunk}", flush=True)
            schedule = [("warmup", "short")] + [(f"short-{i}", "short") for i in range(1, 4)]
            if not bracket:
                schedule.append(("32k", "32k"))
            for label, name in schedule:
                if name == "32k" and chunk != 256:
                    control = json.loads((OUT / "chunk256/results.json").read_text())
                    baseline = statistics.median(r["prefill_ms"] for r in control["requests"] if r["name"].startswith("short-"))
                    candidate = statistics.median(r["prefill_ms"] for r in result["requests"] if r["name"].startswith("short-"))
                    if candidate >= baseline * 1.05:
                        result["escalation"] = "stopped: short median >=5% slower than contemporaneous baseline"
                        break
                    result["escalation"] = "short median within 5% of baseline; one descriptive 32K check"
                meta, ids, ref = fixture(name)
                phase = label
                request_start, epoch = time.monotonic(), time.time()
                write_errors = []

                def write():
                    try:
                        process.stdin.write(f"GEN {meta['generation_cap']} " + ",".join(map(str, ids)) + "\n")
                        process.stdin.flush()
                    except (OSError, ValueError) as exc:
                        write_errors.append(str(exc))

                writer = threading.Thread(target=write, daemon=True)
                writer.start()
                captured = read_until(lambda s: s.startswith("DONE "), 300)
                wall = time.monotonic() - request_start
                writer.join(timeout=5)
                if writer.is_alive() or write_errors:
                    raise RuntimeError(f"request write failed: {write_errors}")
                # Preserve outputs and timing even when a correctness gate rejects escalation.
                raw = {"name": label, "lines": captured, "wall_seconds_to_done": wall,
                       "prompt_ids_sha256": meta["prompt_ids_sha256"], "prompt_metadata": meta}
                (folder / f"{label}.json").write_text(json.dumps(raw, indent=2) + "\n")
                data = validate(captured, meta, ids, ref, chunk, tokenizer)
                if label == "warmup" and not any("batched gather enabled for routed native MMQ experts" in s for s in captured):
                    raise RuntimeError("actual batching not confirmed")
                counters = read_until(lambda s: "file tier I/O this request:" in s, 30)
                if label != "warmup" and not counters[-1].endswith("for 0.0 MB of expert reads"):
                    raise RuntimeError("resident experts reread from source")
                data.update({k: v for k, v in raw.items() if k != "lines"})
                data.update({"counter_lines": counters,
                             "pcie": base.pcie_summary(folder / "pcie.log", epoch, time.time()),
                             "memory_after": nvml.memory()})
                result["requests"].append(data)
                sample(force=True)
                save()
                print(f"{base.utc_now()} PASS chunk{chunk} {label} {data['prefill_ms']}ms", flush=True)
                if chunk != 256 and label.startswith("short-"):
                    control = json.loads((OUT / "chunk256/results.json").read_text())
                    baseline = statistics.median(r["prefill_ms"] for r in control["requests"] if r["name"].startswith("short-"))
                    if data["prefill_ms"] > 1.10 * baseline:
                        result["escalation"] = "stopped early: measured short >10% slower than baseline median"
                        break
            result["status"] = "passed"
        except BaseException as exc:
            result["status"], result["error"] = "failed", f"{type(exc).__name__}: {exc}"
            save()
            raise
        finally:
            if process is not None:
                if process.poll() is None and result["status"] != "passed":
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
            if pcie is not None:
                base.safe_stop(pcie, "telemetry complete")
                pcie_thread.join(timeout=5)
                result["pcie_helper_return_code"] = pcie.poll()
            result["finished_utc"] = base.utc_now()
            result["after_memory"], result["compute_processes_after"] = nvml.memory(), base.compute_processes()
            if result.get("engine_return_code") != 0 and result["status"] == "passed":
                result["status"] = "failed"
            save()
    if result["status"] != "passed" or result["compute_processes_after"]:
        raise RuntimeError("arm or cleanup failed")
    identity()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk", type=int, choices=(256, 384, 512), required=True)
    parser.add_argument("--bracket", action="store_true", help="one extra 256 warmup + three short repeats, no 32K")
    args = parser.parse_args()
    run(args.chunk, args.bracket)
