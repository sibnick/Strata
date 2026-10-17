#!/usr/bin/env python3
"""Offline checks and summaries; never launches an engine or a prompt."""
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
spec = importlib.util.spec_from_file_location(
    "chunks_summary_prior", OUT.parent / "2026-10-17-cmp170hx-rtx3090-k37-mmq-batched-long/analyze.py")
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)


def main():
    arms = {}
    for chunk in ("256", "384", "512", "256-bracket"):
        folder = OUT / f"chunk{chunk}"
        result = json.loads((folder / "results.json").read_text())
        samples = [json.loads(s) for s in (folder / "monitor.jsonl").read_text().splitlines()]
        safety = prior.safety(samples)
        if result["compute_processes_after"] or safety["other_compute_observed"]:
            raise RuntimeError("cleanup/other compute failed")
        if any(b < 640 * (1 << 20) for b in safety["gpu_min_free_bytes"].values()):
            raise RuntimeError("GPU safety threshold crossed")
        if safety["host_min_available_bytes"] < 5 * (1 << 30):
            raise RuntimeError("host safety threshold crossed")
        measured = [r for r in result["requests"] if r["name"].startswith("short-")]
        if len(measured) > 3 or len([r for r in result["requests"] if r["name"] == "32k"]) > 1:
            raise RuntimeError("bounded measurement schedule exceeded")
        if result["status"] == "passed" and result["engine_return_code"] != 0:
            raise RuntimeError("passed result without clean engine exit")
        for request in result["requests"]:
            if request["input_tokens"] not in (1977, 32788) or request["reused"] or request["fresh"] != request["input_tokens"]:
                raise RuntimeError("unapproved prompt or reuse")
        if result["engine_sha256"] != "aba34837fe52effa15450a85a82d6cde9af6a55c7866361ecd1f49099a5f6af9":
            raise RuntimeError("binary changed")
        if result["kquant_library_sha256"] != "3ba4cc2c791be319e5c010965b95c75443fb52194b52ea28c43c1fd8397335a0":
            raise RuntimeError("DSO changed")
        harness = OUT / ("run.py" if chunk == "256-bracket" else "run-initial.py")
        if result["harness_sha256"] != hashlib.sha256(harness.read_bytes()).hexdigest():
            raise RuntimeError("exact measurement harness not preserved")
        normalized = list(result["argv"])
        normalized[normalized.index("--prefill") + 1] = "256"
        control_result = json.loads((OUT / "chunk256/results.json").read_text())
        if normalized != control_result["argv"] or result["environment_overrides"] != control_result["environment_overrides"]:
            raise RuntimeError("arms differ beyond requested chunk")
        if result["artifact_identity"] != control_result["artifact_identity"]:
            raise RuntimeError("artifact sizes/mtimes/hashes changed across arms")
        if chunk == "256-bracket" and [r["name"] for r in result["requests"]] != [
                "warmup", "short-1", "short-2", "short-3"]:
            raise RuntimeError("baseline bracket schedule mismatch")
        arms[chunk] = {
            "status": result["status"], "error": result.get("error"),
            "engine_return_code": result["engine_return_code"],
            "pcie_helper_return_code": result["pcie_helper_return_code"],
            "escalation": result.get("escalation"), "startup_seconds": result.get("startup_seconds"),
            "admission": result["admission"], "startup_gate": result.get("startup_gate"),
            "overall_safety": safety,
            "phases": {phase: {"safety": prior.safety([s for s in samples if s["phase"] == phase]),
                              "telemetry": prior.telemetry([s for s in samples if s["phase"] == phase])}
                       for phase in sorted({s["phase"] for s in samples})},
            "short_prefill_ms": [r["prefill_ms"] for r in measured],
            "short_median_prefill_ms": statistics.median(r["prefill_ms"] for r in measured) if measured else None,
            "requests": result["requests"],
        }
    baseline = arms["256"]["short_median_prefill_ms"]
    bracket = arms["256-bracket"]["short_median_prefill_ms"]
    for chunk, arm in arms.items():
        if arm["short_median_prefill_ms"] is not None:
            arm["short_time_reduction_percent_vs256"] = 100 * (1 - arm["short_median_prefill_ms"] / baseline)
            arm["short_time_reduction_percent_vs256_bracket"] = 100 * (1 - arm["short_median_prefill_ms"] / bracket)
        for r in arm["requests"]:
            if r["name"] == "32k":
                control = next(r for r in arms["256"]["requests"] if r["name"] == "32k")
                arm["single_32k_time_reduction_percent_vs256"] = 100 * (1 - r["prefill_ms"] / control["prefill_ms"])
    summary = {"arms": arms,
               "order": ["256", "384", "512", "256-bracket"],
               "baseline_bracket_reason": "RTX short-phase clocks were 1785..1815 MHz at initial 256, "
                                         "1935 MHz at 384. One permitted short-only bracket checks drift.",
               "limits": "Sequential conditions, one 32K per eligible arm; not randomized repeat proof. "
                         "No near-limit/128K/262K request; no full-context speed extrapolation. "
                         "Polling is not an OOM guarantee. Exact output changes reject equivalence."}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    compute = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        text=True).strip()
    if compute:
        raise RuntimeError("final cleanup still has compute")
    cleanup = {
        "compute_processes": compute,
        "engine_and_pcie_pids": {
            str(c): {str(json.loads((OUT / f"chunk{c}/results.json").read_text())[key]):
                     Path("/proc", str(json.loads((OUT / f"chunk{c}/results.json").read_text())[key])).exists()
                     for key in ("engine_pid", "pcie_helper_pid")} for c in ("256", "384", "512", "256-bracket")},
        "gpu_state": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.free,power.limit", "--format=csv,noheader"], text=True),
        "runtime_diff": subprocess.check_output(
            ["git", "diff", "97b83332", "--", "src", "include", "CMakeLists.txt"], text=True),
    }
    if any(alive for pids in cleanup["engine_and_pcie_pids"].values() for alive in pids.values()):
        raise RuntimeError("own process still alive")
    (OUT / "cleanup.json").write_text(json.dumps(cleanup, indent=2) + "\n")
    (OUT / "checks.json").write_text(json.dumps({
        "bounded_schedule": True, "guards_held": True, "unchanged_binary_dso": True,
        "no_runtime_source_changes": not cleanup["runtime_diff"], "no_compute_after": True,
        "own_engine_and_helpers_exited": True,
        "arm_statuses": {c: a["status"] for c, a in arms.items()}}, indent=2) + "\n")
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "SHA256SUMS"
                   and "__pycache__" not in p.parts)
    (OUT / "SHA256SUMS").write_text("".join(
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(OUT)}\n" for p in files))


if __name__ == "__main__":
    main()
