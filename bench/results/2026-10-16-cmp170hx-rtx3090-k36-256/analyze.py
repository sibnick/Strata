#!/usr/bin/env python3
"""Summarize the authorized K36 run, including failures, without GPU execution."""
import json
from pathlib import Path
import statistics
import re

OUT = Path(__file__).resolve().parent


def main():
    run = json.loads((OUT / "mmq/results.json").read_text())
    prior = json.loads((OUT.parent / "2026-10-16-cmp170hx-rtx3090-mmq-640/mmq/results.json").read_text())
    expected_argv = prior["argv"].copy()
    expected_argv[expected_argv.index("--layer-split") + 1] = "36"
    assert run["argv"] == expected_argv
    assert run["environment_overrides"] == prior["environment_overrides"]
    for key in ("engine_sha256", "cache_sha256", "kquant_library_sha256", "artifact_identity"):
        assert run[key] == prior[key]
    samples = [json.loads(s) for s in (OUT / "mmq/monitor.jsonl").read_text().splitlines()]
    log = (OUT / "mmq/engine.log").read_text().splitlines()
    result = {
        "status": run["status"], "error": run.get("error"),
        "engine_return_code": run.get("engine_return_code"),
        "startup_seconds": run.get("startup_seconds"),
        "completed_requests": len(run["requests"]),
        "min_gpu_free_bytes": {
            str(i): min(s["gpu_memory_bytes"][i]["free_bytes"] for s in samples) for i in (0, 1)
        },
        "min_host_available_bytes": min(s["memory"]["MemAvailable"] for s in samples),
        "max_process_rss_bytes": max(s["process"].get("VmRSS", 0) for s in samples),
        "max_process_swap_bytes": max(s["process"].get("VmSwap", 0) for s in samples),
        "max_system_swap_bytes": max(s["memory"]["SwapTotal"] - s["memory"]["SwapFree"] for s in samples),
        "max_sample_gap_seconds": max(b["elapsed_seconds"] - a["elapsed_seconds"] for a, b in zip(samples, samples[1:])),
        "ownership_lines": [s for s in log if "gpu-owned-experts:" in s and ("owns layers" in s or "all 24576" in s)],
        "last_checkpoint": next((s for s in reversed(log) if s.startswith("PP ")), None),
        "requests": {},
    }
    admission = next((s for s in log if "refusing partial ownership" in s), None)
    if admission:
        match = re.search(r"needs (\d+) bytes for (\d+) assigned experts, but (\d+) bytes are free \((\d+) after", admission)
        if not match:
            raise ValueError("unexpected capacity rejection diagnostic")
        required, experts, free, usable = map(int, match.groups())
        result["engine_admission_failure"] = {
            "raw_line": admission, "required_expert_bytes": required,
            "assigned_experts": experts, "free_before_experts_bytes": free,
            "usable_after_reserves_bytes": usable, "deficit_bytes": required - usable,
            "reserved_bytes": free - usable,
        }
        result["primary_failure"] = "engine full-residency admission rejected K36 before READY"
        result["secondary_monitor_failure"] = run.get("error")
    for req in run["requests"]:
        assert req["answer"].strip() == "ORIGIN: 714\nCENTER: 928\nFINAL: 365"
        assert req["reused"] == 0 and req["fresh"] == req["input_tokens"] <= 2048
        assert req["output_tokens"] < 32
    for size in ("small", "medium", "large"):
        requests = [r for r in run["requests"] if r["name"].endswith("-" + size)]
        if not requests:
            continue
        old = [r for r in prior["requests"] if r["name"].endswith("-" + size)]
        for req in requests:
            old_req = next(r for r in old if r["name"] == req["name"])
            assert req["prompt_ids_sha256"] == old_req["prompt_ids_sha256"]
        wall = statistics.median(r["wall_seconds_to_done"] for r in requests)
        result["requests"][size] = {
            "input_tokens": requests[0]["input_tokens"],
            "repetitions": len(requests),
            "prefill_ms": [r["prefill_ms"] for r in requests],
            "wall_seconds_to_done": [r["wall_seconds_to_done"] for r in requests],
            "median_prefill_ms": statistics.median(r["prefill_ms"] for r in requests),
            "median_wall_seconds_to_done": wall,
            "prior_k37_median_wall_seconds": statistics.median(r["wall_seconds_to_done"] for r in old),
            "prior_k37_wall_over_k36": statistics.median(r["wall_seconds_to_done"] for r in old) / wall,
            "generated_ids_match_k37": all(r["generated_ids"] == next(x for x in old if x["name"] == r["name"])["generated_ids"] for r in requests),
            "zero_expert_read_deltas": all(r["counter_lines"][-1].endswith("for 0.0 MB of expert reads") for r in requests),
        }
    result["limits"] = [
        "K37 comparison is a prior serial run, not randomized or same-guard paired measurement.",
        "Short inputs only; no near-max quality/memory guarantee or optimal-split claim.",
        "RTX monitor256MiB, CMP640MiB; allocation reserves1024MiB and host5GiB reaction unchanged.",
        "Polling cannot guarantee against transient pressure or OOM.",
    ]
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
