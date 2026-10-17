#!/usr/bin/env python3
"""Offline summaries: controlled single 32K pair, descriptive historical near-limit only."""
import csv
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
PRIOR = OUT.parent / "2026-10-16-cmp170hx-rtx3090-k37-mmq-long-context"


def safety(samples):
    return {
        "samples": len(samples),
        "gpu_min_free_bytes": {
            str(i): min(g["free_bytes"] for s in samples for g in s["gpu_memory_bytes"] if g["physical_index"] == i)
            for i in (0, 1)},
        "host_min_available_bytes": min(s["memory"]["MemAvailable"] for s in samples),
        "max_rss_bytes": max(s["process"].get("VmRSS", 0) for s in samples),
        "max_process_swap_bytes": max(s["process"].get("VmSwap", 0) for s in samples),
        "max_system_swap_bytes": max(s["memory"]["SwapTotal"] - s["memory"]["SwapFree"] for s in samples),
        "max_sample_gap_seconds": max(s["sample_gap_seconds"] for s in samples),
        "other_compute_observed": any(s["other_compute"] for s in samples),
    }


def telemetry(samples):
    result = {}
    for device in (0, 1):
        records = [row for s in samples for row in csv.reader(s["telemetry"].splitlines(), skipinitialspace=True)
                   if int(row[0]) == device]
        result[device] = {
            "power_limits_w": sorted({float(r[3]) for r in records}),
            "power_w_min_max": [min(float(r[4]) for r in records), max(float(r[4]) for r in records)],
            "sm_clock_mhz_min_max": [min(int(r[6]) for r in records), max(int(r[6]) for r in records)],
            "temperature_c_min_max": [min(int(r[7]) for r in records), max(int(r[7]) for r in records)],
            "active_clock_event_reason_masks": sorted({r[8] for r in records}),
        }
    return result


def summarize(arm):
    result = json.loads((OUT / arm / "results.json").read_text())
    samples = [json.loads(s) for s in (OUT / arm / "monitor.jsonl").read_text().splitlines()]
    return {
        "status": result["status"], "engine_return_code": result.get("engine_return_code"),
        "cleanup_compute": result.get("compute_processes_after"),
        "startup_seconds": result.get("startup_seconds"),
        "overall_safety": safety(samples),
        "phases": {phase: {"safety": safety([s for s in samples if s["phase"] == phase]),
                          "telemetry": telemetry([s for s in samples if s["phase"] == phase])}
                   for phase in sorted({s["phase"] for s in samples})},
        "requests": [{**{key: r[key] for key in (
            "name", "input_tokens", "output_tokens", "prefill_ms", "prefill_tokens_per_second",
            "decode_ms", "wall_seconds_to_done", "ttft_seconds_observed", "mtp_accepted",
            "mtp_offered", "fresh", "reused", "answer", "timeout_seconds")},
            "decode_tokens_per_second": r["output_tokens"] * 1000 / r["decode_ms"],
            "pcie": r["pcie"]} for r in result["requests"]],
    }


def main():
    arms = {a: summarize(a) for a in ("off", "on") if (OUT / a / "results.json").exists()}
    comparisons = {}
    if all(a in arms and any(r["name"] == "32k" for r in arms[a]["requests"]) for a in ("off", "on")):
        off, on = [next(r for r in arms[a]["requests"] if r["name"] == "32k") for a in ("off", "on")]
        comparisons["controlled_32k_single_pair"] = {
            "off_ms": off["prefill_ms"], "on_ms": on["prefill_ms"],
            "speedup": off["prefill_ms"] / on["prefill_ms"],
            "prefill_time_reduction_percent": 100 * (1 - on["prefill_ms"] / off["prefill_ms"]),
            "note": "One request per arm after warmup; same binary/config, sequential OFF then ON; not repeated randomized evidence.",
        }
    if "on" in arms and any(r["name"] == "near-max" for r in arms["on"]["requests"]):
        on = next(r for r in arms["on"]["requests"] if r["name"] == "near-max")
        historical = next(r for r in json.loads((PRIOR / "mmq/results.json").read_text())["requests"]
                          if r["name"] == "near-max")
        comparisons["descriptive_historical_near_limit"] = {
            "historical_prefill_ms": historical["prefill_ms"], "current_on_prefill_ms": on["prefill_ms"],
            "historical_ttft_s": historical["ttft_seconds_observed"],
            "current_on_ttft_s": on["ttft_seconds_observed"],
            "time_ratio_historical_over_current": historical["prefill_ms"] / on["prefill_ms"],
            "time_difference_percent": 100 * (1 - on["prefill_ms"] / historical["prefill_ms"]),
            "note": "Historical different build/order/cache/thermal/power state; not a controlled batching speedup. No fresh OFF near-limit.",
        }
    print(json.dumps({"arms": arms, "comparisons": comparisons}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
