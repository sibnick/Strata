#!/usr/bin/env python3
"""Summarize the committed run without GPU access or model startup."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import statistics
from pathlib import Path


EVIDENCE = Path(__file__).resolve().parent.parent
EDGE_SECONDS = 2.0


def summary(values: list[float]) -> dict:
    if not values:
        return {"sample_count": 0}
    return {
        "sample_count": len(values),
        "mean": round(statistics.mean(values), 3),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def summarize_gpus(rows: list[dict]) -> dict:
    result = {}
    indices = sorted({gpu["index"] for row in rows for gpu in row["gpus"]})
    for index in indices:
        gpus = [gpu for row in rows for gpu in row["gpus"] if gpu["index"] == index]
        utilization = [gpu["utilization_gpu_percent"] for gpu in gpus]
        result[str(index)] = {
            "name": gpus[0]["name"],
            "utilization_percent": summary(utilization),
            "samples_at_or_below_20_percent": sum(value <= 20 for value in utilization),
            "samples_at_or_above_90_percent": sum(value >= 90 for value in utilization),
            "power_w": summary([gpu["power_w"] for gpu in gpus]),
        }
    return result


def prefill_rows(samples: list[dict], request: dict) -> list[dict]:
    return [
        row for row in samples
        if row["stage"] == request["name"]
        and row["request_elapsed_seconds"] is not None
        and EDGE_SECONDS <= row["request_elapsed_seconds"]
        <= request["ttft_seconds"] - EDGE_SECONDS
    ]


def pcie_rows(path: Path) -> list[tuple[float, int, float, float]]:
    rows = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) > 1 and fields[1] == "#":
            continue
        if len(fields) != 4:
            raise ValueError(f"Unexpected PCIe telemetry row: {line!r}")
        stamp, index, rx, tx = fields
        rows.append((float(stamp), int(index), float(rx), float(tx)))
    return rows


def main() -> None:
    results = json.loads((EVIDENCE / "main-results.json").read_text())
    samples = [
        json.loads(line)
        for line in (EVIDENCE / "main-monitor.jsonl").read_text().splitlines()
    ]
    pcie = pcie_rows(EVIDENCE / "main-pcie.log")
    completed = [r for r in results["requests"] if r["status"] == "passed"]
    output = {
        "method": {
            "execution": "offline only; no CUDA calls, model startup, or artifact rehash",
            "gpu_indices": "physical nvidia-smi indices, not command-order CUDA indices",
            "prefill_window": "request elapsed [2, TTFT - 2] seconds; approximate phase interior",
            "statistics": "unweighted sample statistics, not exact device time fractions",
            "utilization": "whole-device sampled kernel activity, not achieved FLOPs or occupancy",
            "pcie_units": "MB/s as printed by nvidia-smi dmon; whole-device counters",
            "decode": "0.389-0.601 seconds; too short for reliable phase attribution at this sampling cadence",
        },
        "input_sha256": {
            name: hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest()
            for name in ("main-results.json", "main-monitor.jsonl", "main-pcie.log", "main-engine.log")
        },
        "sample_intervals_seconds": summary([
            b["elapsed_seconds"] - a["elapsed_seconds"]
            for a, b in zip(samples, samples[1:])
        ]),
        "startup": summarize_gpus([s for s in samples if s["stage"] == "startup"]),
        "completed_prefill": {},
    }
    for request in completed:
        rows = prefill_rows(samples, request)
        if not rows:
            raise ValueError(f"No interior prefill samples for {request['name']}")
        start = dt.datetime.fromisoformat(request["started_utc"]).timestamp()
        low, high = EDGE_SECONDS, request["ttft_seconds"] - EDGE_SECONDS
        phase = {
            "gpu": summarize_gpus(rows),
            "pcie": {},
            "input_tokens": request["input_tokens"],
            "prefill_tokens_per_second": request["prefill_tokens_per_second"],
        }
        for index in (0, 1):
            transfers = [
                (rx, tx) for stamp, gpu, rx, tx in pcie
                if gpu == index and start + low <= stamp <= start + high
            ]
            phase["pcie"][str(index)] = {
                "rx_mb_s": summary([rx for rx, _ in transfers]),
                "tx_mb_s": summary([tx for _, tx in transfers]),
            }
        if request["name"] == "near-max":
            phase["elapsed_time_quarters"] = [
                {
                    "quarter": i + 1,
                    "gpu": summarize_gpus([
                        row for row in rows
                        if low + i * (high - low) / 4 <= row["request_elapsed_seconds"]
                        < low + (i + 1) * (high - low) / 4
                    ]),
                }
                for i in range(4)
            ]
        output["completed_prefill"][request["name"]] = phase
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
