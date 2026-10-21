#!/usr/bin/env python3
"""Summarize persisted arms without treating counter-enabled requests as speed evidence."""
import json
import csv
from pathlib import Path
import re
import statistics

OUT = Path(__file__).resolve().parent
COUNT = re.compile(r"device=(\d+).* launches=(\d+) experts=(\d+) bytes=(\d+) fallbacks=(\d+)")


def summarize(folder):
    result = json.loads((folder / "results.json").read_text())
    samples = [json.loads(s) for s in (folder / "monitor.jsonl").read_text().splitlines()]
    rows = result["requests"]
    measured = [r for r in rows if r["name"] != "warmup"]
    perf = {}
    for size in sorted({r["input_tokens"] for r in measured}):
        reps = [r for r in measured if r["input_tokens"] == size]
        perf[size] = {
            "prefill_ms": [r["prefill_ms"] for r in reps],
            "prefill_median_ms": statistics.median(r["prefill_ms"] for r in reps),
            "decode_ms": [r["decode_ms"] for r in reps],
            "decode_median_ms": statistics.median(r["decode_ms"] for r in reps),
            "wall_seconds": [r["wall_seconds_to_done"] for r in reps],
            "wall_median_seconds": statistics.median(r["wall_seconds_to_done"] for r in reps),
            "mtp_accepted_offered": [r["mtp_accepted_offered"] for r in reps],
        }
    counts = []
    for request in rows:
        devices = {}
        for line in request["gather_counts"]:
            match = COUNT.search(line)
            if not match:
                raise RuntimeError(f"bad count line: {line}")
            device, *values = map(int, match.groups())
            totals = devices.setdefault(device, dict.fromkeys(("launches", "experts", "bytes", "fallbacks"), 0))
            for key, value in zip(totals, values):
                totals[key] += value
        counts.append({"name": request["name"], "input_tokens": request["input_tokens"], "devices": devices})
    safety = {
        "gpu_min_free_bytes": {
            str(i): min(g["free_bytes"] for s in samples for g in s["gpu_memory_bytes"] if g["physical_index"] == i)
            for i in (0, 1)
        },
        "host_min_available_bytes": min(s["memory"]["MemAvailable"] for s in samples),
        "max_rss_bytes": max(s["process"].get("VmRSS", 0) for s in samples),
        "max_process_swap_bytes": max(s["process"].get("VmSwap", 0) for s in samples),
        "max_system_swap_bytes": max(s["memory"]["SwapTotal"] - s["memory"]["SwapFree"] for s in samples),
        "max_sample_gap_seconds": max(s["sample_gap_seconds"] for s in samples),
        "other_compute_observed": any(s["other_compute"] for s in samples),
    }
    telemetry = {}
    for device in (0, 1):
        records = [row for s in samples if s["phase"] != "startup"
                   for row in csv.reader(s["telemetry"].splitlines(), skipinitialspace=True)
                   if int(row[0]) == device]
        if records:
            telemetry[device] = {
                "power_limits_w": sorted({float(r[3]) for r in records}),
                "power_w_min_max": [min(float(r[4]) for r in records), max(float(r[4]) for r in records)],
                "sm_clock_mhz_min_max": [min(int(r[6]) for r in records), max(int(r[6]) for r in records)],
                "temperature_c_min_max": [min(int(r[7]) for r in records), max(int(r[7]) for r in records)],
                "active_clock_event_reason_masks": sorted({r[8] for r in records}),
            }
    return {
        "status": result["status"], "engine_return_code": result.get("engine_return_code"),
        "cleanup_compute": result.get("compute_processes_after"), "startup_seconds": result.get("startup_seconds"),
        "instrumented": "STRATA_PREFILL_GATHER_COUNTS" in result["environment_overrides"],
        "batch": result["environment_overrides"]["STRATA_PREFILL_BATCH_GATHER"],
        "performance": perf, "counts": counts, "safety": safety, "request_telemetry": telemetry,
    }


def main():
    arms = {p.name: summarize(p) for p in sorted(OUT.iterdir()) if (p / "results.json").is_file()}
    comparison = {}
    for name, arm in arms.items():
        if arm["instrumented"] or arm["batch"] != "1":
            continue
        for size, perf in arm["performance"].items():
            controls = [(n, a["performance"][size]) for n, a in arms.items()
                        if not a["instrumented"] and a["batch"] == "0" and n != "preserved"
                        and size in a["performance"]]
            for control, off in controls:
                old, new = off["prefill_median_ms"], perf["prefill_median_ms"]
                comparison[f"{control}:{name}:{size}"] = {
                    "off_ms": old, "on_ms": new, "speedup": old / new,
                    "prefill_time_reduction_percent": 100 * (old - new) / old,
                }
    print(json.dumps({"arms": arms, "performance_comparisons": comparison}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
