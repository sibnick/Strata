#!/usr/bin/env python3
"""Offline result audit, including failed admission; never runs a workload."""
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
PREVIOUS = OUT.parent / "2026-10-17-cmp170hx-rtx3090-k37-mmq-chunks"
spec = importlib.util.spec_from_file_location(
    "larger_safety", OUT.parent / "2026-10-17-cmp170hx-rtx3090-k37-mmq-batched-long/analyze.py")
safety_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(safety_module)
run_spec = importlib.util.spec_from_file_location("larger_run", OUT / "run.py")
run_module = importlib.util.module_from_spec(run_spec)
run_spec.loader.exec_module(run_module)


def main():
    run_module.identity()
    tokenizer = run_module.long.profile.prepare.load_tokenizer()
    historical = json.loads((PREVIOUS / "chunk512/results.json").read_text())
    baseline_short = statistics.median(r["prefill_ms"] for r in historical["requests"]
                                       if r["name"].startswith("short-"))
    baseline_long = next(r["prefill_ms"] for r in historical["requests"] if r["name"] == "32k")
    arms = {}
    cleanup_pids = {}
    for chunk in (768, 1024):
        folder = OUT / f"chunk{chunk}"
        result = json.loads((folder / "results.json").read_text())
        if result["status"] == "offline-infeasible":
            arms[chunk] = result
            continue
        samples = [json.loads(s) for s in (folder / "monitor.jsonl").read_text().splitlines()]
        safety = safety_module.safety(samples)
        if safety["host_min_available_bytes"] < 5 * (1 << 30) or safety["other_compute_observed"]:
            raise RuntimeError("host/other-compute guard violation; report explicitly before finalizing")
        if result["compute_processes_after"]:
            raise RuntimeError("compute remains after arm")
        expected_guards = {"gpu_stop_mib": None, "engine_reserve_mib": [0, 0],
                           "host_start_gib": 6, "host_stop_gib": 5,
                           "mandatory_cuda_admission_errors_retained": True}
        for key, value in expected_guards.items():
            if result["guards"][key] != value:
                raise RuntimeError(f"authorization scope mismatch: {key}")
        normalized = list(result["argv"])
        normalized[normalized.index("--prefill") + 1] = "512"
        for flag in ("--vram-reserve-mib", "--vram-reserve-later-mib"):
            if result["argv"][result["argv"].index(flag) + 1] != "0":
                raise RuntimeError("nonzero discretionary GPU reserve")
            normalized[normalized.index(flag) + 1] = "1024"
        if normalized != historical["argv"]:
            raise RuntimeError("argv differs beyond authorized chunk and GPU reserves")
        if result["environment_overrides"] != historical["environment_overrides"]:
            raise RuntimeError("runtime environment changed")
        for key in ("engine_sha256", "kquant_library_sha256", "cache_sha256", "artifact_identity"):
            if result[key] != historical[key]:
                raise RuntimeError(f"unchanged provenance differs: {key}")
        if result["harness_sha256"] != hashlib.sha256((OUT / "run.py").read_bytes()).hexdigest():
            raise RuntimeError("measurement harness changed after run")
        if result.get("startup_gate"):
            run_module.startup_gate(result["startup_gate"]["startup_lines"], result["argv"], chunk)
            if result["loaded_dso"] != [str(run_module.ENGINE.parent / "libstrata_kq.so")]:
                raise RuntimeError("loaded DSO identity differs")
        requests = result["requests"]
        names = [r["name"] for r in requests]
        allowed = ["warmup", "short-1", "short-2", "short-3", "32k"]
        if names != allowed[:len(names)]:
            raise RuntimeError("bounded schedule mismatch")
        if result["status"] == "passed" and result["engine_return_code"] != 0:
            raise RuntimeError("passed run did not exit cleanly")
        for request in requests:
            if request["input_tokens"] not in (1977, 32788) or request["reused"] or request["fresh"] != request["input_tokens"]:
                raise RuntimeError("unapproved prompt or reuse")
            control = next(r for r in historical["requests"] if r["input_tokens"] == request["input_tokens"])
            if request["generated_ids"] != control["generated_ids"]:
                raise RuntimeError("passed request output IDs differ")
            raw = json.loads((folder / f"{request['name']}.json").read_text())
            meta, ids, ref = run_module.fixture("32k" if request["name"] == "32k" else "short")
            validated = run_module.validate(raw["lines"], meta, ids, ref, chunk, tokenizer)
            for key, value in validated.items():
                if request[key] != value:
                    raise RuntimeError(f"raw request validation differs: {request['name']} {key}")
            counters = json.loads((folder / f"{request['name']}-counters.json").read_text())
            if counters != request["counter_lines"]:
                raise RuntimeError("raw request counters differ")
            if request["name"] != "warmup" and not counters[-1].endswith("for 0.0 MB of expert reads"):
                raise RuntimeError("measured expert source reads")
        shorts = [r["prefill_ms"] for r in requests if r["name"].startswith("short-")]
        arms[chunk] = {
            "status": result["status"], "error": result.get("error"),
            "warnings": [line for line in (folder / "engine.log").read_text().splitlines()
                         if "WARNING" in line],
            "startup_gate": result.get("startup_gate"), "admission": result["admission"],
            "engine_return_code": result["engine_return_code"],
            "pcie_helper_return_code": result["pcie_helper_return_code"],
            "overall_safety": safety,
            "phases": {phase: {
                "safety": safety_module.safety([s for s in samples if s["phase"] == phase]),
                "telemetry": safety_module.telemetry([s for s in samples if s["phase"] == phase])}
                for phase in sorted({s["phase"] for s in samples})},
            "short_ms": shorts, "short_median_ms": statistics.median(shorts) if shorts else None,
            "requests": requests, "escalation": result.get("escalation"),
            "comparisons": {
                "historical512_short_median_ms": baseline_short,
                "short_prefill_time_reduction_percent": 100 * (1 - statistics.median(shorts) / baseline_short) if shorts else None,
                "historical512_single32k_ms": baseline_long,
                "single32k_prefill_time_reduction_percent": next((
                    100 * (1 - r["prefill_ms"] / baseline_long) for r in requests if r["name"] == "32k"), None),
                "limit": "Historical sequential comparison, different discretionary GPU reserves and external safety. "
                         "1024 additionally changes stream_all execution strategy. Not same-safety randomized proof.",
            },
        }
        cleanup_pids[str(chunk)] = {
            str(result[k]): Path("/proc", str(result[k])).exists() for k in ("engine_pid", "pcie_helper_pid")}
    compute = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], text=True).strip()
    if compute or any(alive for pids in cleanup_pids.values() for alive in pids.values()):
        raise RuntimeError("final own-process/compute cleanup failed")
    runtime_diff = subprocess.check_output(["git", "diff", "97b83332", "--", "src", "include", "CMakeLists.txt"], text=True)
    if runtime_diff:
        raise RuntimeError("runtime source changed")
    (OUT / "summary.json").write_text(json.dumps({"arms": arms}, indent=2, sort_keys=True) + "\n")
    (OUT / "cleanup.json").write_text(json.dumps({
        "compute_processes": compute, "own_pids_alive": cleanup_pids, "runtime_diff": runtime_diff,
        "gpu_state": subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.free,power.limit",
                                              "--format=csv,noheader"], text=True)}, indent=2) + "\n")
    (OUT / "checks.json").write_text(json.dumps({
        "authorization_scope_held": True, "host_guard_held": True, "no_unrelated_compute": True,
        "unchanged_runtime_binary_dso_artifacts": True, "bounded_schedule": True,
        "no_gpu_free_abort_threshold": True, "no_compute_after": True,
        "arm_statuses": {c: a["status"] for c, a in arms.items()}}, indent=2) + "\n")
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "SHA256SUMS" and "__pycache__" not in p.parts)
    (OUT / "SHA256SUMS").write_text("".join(
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(OUT)}\n" for p in files))


if __name__ == "__main__":
    main()
