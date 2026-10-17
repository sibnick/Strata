#!/usr/bin/env python3
"""Bounded K36 test of 2048-token CMP / 1024-token RTX prompt chunks."""
import argparse
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
spec = importlib.util.spec_from_file_location(
    "chunk_prior", OUT.parent / "2026-10-17-cmp170hx-rtx3090-k37-mmq-batched-long/run.py")
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
short, base, long = prior.short, prior.base, prior.long
REFERENCE = json.loads((prior.OUT / "on/results.json").read_text())
ENGINE = ROOT / "build/cuda-per-stage/strata"
STAGE_CHUNKS = (2048, 1024)
SPLIT = 36


def check_pressure(memory, gpus, others):
    if memory["MemAvailable"] < 5 * short.GIB:
        raise RuntimeError("host below 5 GiB reaction threshold")
    if len(gpus) != 2 or any(g["free_bytes"] < 0 for g in gpus):
        raise RuntimeError("invalid GPU monitoring sample")
    if others:
        raise RuntimeError(f"unrelated compute present: {others}")


def identity():
    sizing = json.loads((OUT / "offline.json").read_text())
    if base.sha256(ENGINE) != sizing["engine_sha256"]:
        raise RuntimeError(f"engine identity mismatch: {ENGINE}")
    if base.sha256(ENGINE.parent / "libstrata_kq.so") != sizing["kquant_library_sha256"]:
        raise RuntimeError("K-quant library identity mismatch")
    for relative, expected in sizing["source_sha256"].items():
        if base.sha256(ROOT / relative) != expected:
            raise RuntimeError(f"source identity mismatch: {relative}")


def shapes(lines, tokens, chunk):
    positions = [int(s.split()[1]) for s in lines if s.startswith("PP ")]
    expected = list(range(chunk, tokens - 1, chunk)) + [tokens - 1]
    if positions != expected:
        raise RuntimeError(f"actual prompt chunks differ: {positions} != {expected}")
    return [b - a for a, b in zip([0] + positions[:-1], positions)]


def startup_gate(lines, argv):
    joined = "\n".join(lines)
    size = json.loads((OUT / "offline.json").read_text())["sizing"]["36"]
    scratch = {stage: value["owned_prefill_allocation_estimate"] for stage, value in size.items()}
    for flag, value in (
        ("--gpu", "1,0"), ("--layer-split", "36"), ("--prefill", "2048"),
        ("--max-context", "262144"), ("--kv", "int8"),
        ("--vram-reserve-mib", "0"), ("--vram-reserve-later-mib", "0"),
        ("--prompt-cache", "0"), ("--spec", "2"),
    ):
        if argv[argv.index(flag) + 1] != value:
            raise RuntimeError(f"configuration mismatch: {flag}")
    if argv[argv.index("--prefill-stage-chunks") + 1] != "2048,1024":
        raise RuntimeError("unexpected per-stage chunk argv")
    required = (
        "GPU 0: NVIDIA CMP 170HX, compute capability 8.0",
        "CUDA0 owns layers [0,36), 18432 experts, 57619251200 resident bytes",
        "CUDA1 owns layers [36,48), 6144 experts, 19398656000 resident bytes",
        "all 24576 experts are resident in their owning GPU caches",
        "layer split: CUDA0 runs layers 0-35",
        "layer split: CUDA1 runs layers 36-47",
        "INFO logit_bias=1 context=262144 kv=int8",
        "expert_slots=24576",
        "expert_slots_primary=18432",
        "prompt path allocates its own buffers",
        f"K-quant prompt kernels from {ENGINE.parent / 'libstrata_kq.so'} (STRATA_MMQ_KQ=1)",
        "per-stage prefill chunks: CUDA0=2048 CUDA1=1024",
        f"own buffers for a 2048-token chunk: {(scratch['0'] + short.MIB - 1) // short.MIB} MiB",
        f"CUDA1 chunk 1024 exact scratch {scratch['1']} bytes",
    )
    for item in required:
        if item not in joined:
            raise RuntimeError(f"startup admission missing: {item}")
    if any("trying a " in s or "token chunks" in s for s in lines):
        raise RuntimeError("scheduler clamped requested chunk")
    for line in lines:
        if line.startswith("ERR ") or ("fallback" in line.lower() and "no file fallback is expected" not in line):
            raise RuntimeError(f"startup error/fallback: {line}")
    return {"requested_stage_chunks": list(STAGE_CHUNKS),
            "owned_prefill_bytes_per_stage": scratch, "startup_lines": lines,
            "stream_all_min": 1024, "configured_ring_slots": [96, 96]}


def handoff_gate(lines, prompt_tokens):
    handoffs = [s for s in lines if "layer handoff CUDA1:" in s]
    rows = prompt_tokens - 1
    subchunks = (rows + STAGE_CHUNKS[1] - 1) // STAGE_CHUNKS[1]
    expected = f"{rows} rows consumed in {subchunks} sub-chunks (1024-token stage chunks)"
    if len(handoffs) != 1 or expected not in handoffs[0]:
        raise RuntimeError(f"larger stage-0 handoff was not consumed in 1024-token sub-chunks: {handoffs}")
    return handoffs[0]


def fixture(name):
    if name == "short":
        return short.bundle(["large"])[0]
    meta, ids = base.load_bundle("32k")
    ref = next(r for r in prior.HISTORICAL["requests"] if r["name"] == "32k")
    if len(ids) != 32788 or meta["generation_cap"] != 192 or meta["prompt_ids_sha256"] != ref["prompt_ids_sha256"]:
        raise RuntimeError("immutable 32K fixture mismatch")
    return meta, ids, ref


def validate(lines, meta, ids, ref, tokenizer):
    generated = [int(s.split()[1]) for s in lines if s.startswith("T ")]
    if len(ids) == 1977:
        data = short.validate_done(lines[-1], ids, generated, ref)
        data["prefill_tokens_per_second"] = len(ids) * 1000 / data["prefill_ms"]
    else:
        data = prior.validate(lines[-1], ids, generated, tokenizer, meta, ref)
    data["actual_chunk_shape"] = shapes(lines, len(ids), STAGE_CHUNKS[1])
    data["actual_chunk_count"] = len(data["actual_chunk_shape"])
    data["stage_chunks"] = list(STAGE_CHUNKS)
    data["final_stage_consumer_chunk"] = STAGE_CHUNKS[1]
    data["execution_strategy_per_chunk"] = [
        "stream_all resident-filtered" if n >= 1024 else "routed MMQ" for n in data["actual_chunk_shape"]]
    if any(s.startswith(("strata prefill timing:", "strata gather counts:")) for s in lines):
        raise RuntimeError("forbidden profiler / diagnostic counters")
    return data


def run():
    identity()
    folder = OUT / "mixed2048-1024"
    folder.mkdir(exist_ok=False)
    nvml = short.prior.Nvml()
    memory = base.read_meminfo()
    compute_before = base.compute_processes()
    if compute_before:
        raise RuntimeError("unrelated compute before startup")
    if memory["MemAvailable"] < 6 * short.GIB:
        raise RuntimeError("host startup below 6 GiB")
    gpu_memory = nvml.memory()
    check_pressure(memory, gpu_memory, [])
    sizing = json.loads((OUT / "offline.json").read_text())["sizing"]["36"]
    admission = {}
    for stage, physical in (("0", 1), ("1", 0)):
        need = sizing[stage]["required_with_mandatory_prompt_init_headroom"]
        available = gpu_memory[physical]["free_bytes"]
        admission[stage] = {"physical_gpu": physical, "free_bytes": available,
                            "ledger_required_including_reserves_and_512_mib_prompt_headroom": need,
                            "margin_bytes": available - need}
        if available < need:
            result = {
                "status": "offline-infeasible",
                "gpu_workload_started": False,
                "reason": "mandatory 512 MiB prompt-init headroom does not fit on every stage",
                "admission": admission,
                "preflight": {
                    "host_memavailable_bytes": memory["MemAvailable"],
                    "compute_processes_before": compute_before.splitlines(),
                    "gpu_memory_by_physical_index": gpu_memory,
                    "telemetry": short.telemetry(),
                    "engine_sha256": base.sha256(ENGINE),
                    "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    "engine_reserves_mib": [0, 0],
                    "external_gpu_free_abort_mib": None,
                    "mandatory_prompt_init_headroom_bytes": 512 * short.MIB,
                    "host_start_gib": 6,
                    "host_stop_gib": 5,
                },
            }
            (folder / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            print("NOT STARTED: K36 RTX 3090 fails mandatory 512 MiB prompt-init headroom", flush=True)
            return
    argv = list(REFERENCE["argv"])
    argv[0] = str(ENGINE)
    argv[argv.index("--layer-split") + 1] = str(SPLIT)
    argv[argv.index("--prefill") + 1] = str(STAGE_CHUNKS[0])
    argv[argv.index("--prefill") + 2:argv.index("--prefill") + 2] = [
        "--prefill-stage-chunks", ",".join(map(str, STAGE_CHUNKS))]
    for flag in ("--vram-reserve-mib", "--vram-reserve-later-mib"):
        argv[argv.index(flag) + 1] = "0"
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
        "status": "starting", "layer_split": SPLIT, "stage_chunks": list(STAGE_CHUNKS),
        "argv": argv, "environment_overrides": overrides,
        "admission": admission, "physical_gpu_identity": gpu_identity,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "engine_sha256": base.sha256(ENGINE),
        "kquant_library_sha256": base.sha256(ENGINE.parent / "libstrata_kq.so"),
        "cache_sha256": base.sha256(ENGINE.parent / "CMakeCache.txt"),
        "artifact_identity": base.prior_artifact_identity(),
        "harness_sha256": base.sha256(Path(__file__)), "started_utc": base.utc_now(),
        "guards": {"gpu_stop_mib": None, "engine_reserve_mib": [0, 0],
                   "host_start_gib": 6, "host_stop_gib": 5, "host_floor_gib": 4,
                   "poll_seconds": 1, "polling_not_oom_guarantee": True,
                   "mandatory_cuda_admission_errors_retained": True,
                   "authorization": "User-authorized 0 MiB GPU reserves/no external GPU-free abort; retain host5GiB and mandatory allocation checks"},
        "requests": [],
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
            check_pressure(mem, gpus, others)
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
                if any(s in line.lower() for s in ("illegal memory access", "out of memory",
                                                    "cuda error", "non-finite", "failed to launch")):
                    raise RuntimeError(f"device/allocation error: {line}")
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
            print(f"{base.utc_now()} START K36 stage chunks {STAGE_CHUNKS} pid={process.pid}", flush=True)
            startup = read_until(lambda s: s == "READY 262144 stop", 1200)
            result["startup_gate"] = startup_gate(startup, argv)
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
            print(f"{base.utc_now()} READY K36 stage chunks {STAGE_CHUNKS}", flush=True)
            schedule = [("warmup", "short")] + [(f"short-{i}", "short") for i in range(1, 4)] + [("32k", "32k")]
            for label, name in schedule:
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
                data = validate(captured, meta, ids, ref, tokenizer)
                if label == "warmup" and not any("batched gather enabled for routed native MMQ experts" in s for s in captured):
                    raise RuntimeError("actual batching not confirmed")
                if label == "warmup":
                    data["first_stage_handoff"] = handoff_gate(captured, len(ids))
                counters = read_until(lambda s: "file tier I/O this request:" in s, 30)
                (folder / f"{label}-counters.json").write_text(json.dumps(counters, indent=2) + "\n")
                if label != "warmup" and not counters[-1].endswith("for 0.0 MB of expert reads"):
                    raise RuntimeError("resident experts reread from source")
                data.update({k: v for k, v in raw.items() if k != "lines"})
                data.update({"counter_lines": counters,
                             "pcie": base.pcie_summary(folder / "pcie.log", epoch, time.time()),
                             "memory_after": nvml.memory()})
                result["requests"].append(data)
                sample(force=True)
                save()
                print(f"{base.utc_now()} PASS mixed2048-1024 {label} {data['prefill_ms']}ms", flush=True)
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
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
