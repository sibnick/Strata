#!/usr/bin/env python3
"""Offline comparison and K36 byte arithmetic, without CUDA/model initialization."""
import csv
import hashlib
import json
from pathlib import Path
import statistics


OUT = Path(__file__).resolve().parent
MIB = 1 << 20
LEDGER = OUT.parent / "2026-10-16-cmp170hx-rtx3090-k37/ledger-chunk256.csv"


def k36_delta():
    with LEDGER.open() as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    for stage in (0, 1):
        components = {
            split: {r["component"]: int(r["bytes"]) for r in rows
                    if int(r["split"]) == split and int(r["stage"]) == stage}
            for split in (36, 37)
        }
        result[str(stage)] = {
            component: {
                "k37_bytes": components[37][component],
                "k36_bytes": components[36][component],
                "delta_bytes": components[36][component] - components[37][component],
            } for component in components[36]
        }
    return result


def summarize(mode):
    folder = OUT / mode
    run = json.loads((folder / "results.json").read_text())
    if run["status"] != "passed" or run["engine_return_code"] != 0:
        raise ValueError(f"{mode} did not complete: {run.get('error')}")
    samples = [json.loads(s) for s in (folder / "monitor.jsonl").read_text().splitlines()]
    measured = [r for r in run["requests"] if r["name"] != "warmup"]
    result = {
        "startup_seconds": run["startup_seconds"],
        "min_gpu_free_bytes": {
            str(i): min(s["gpu_memory_bytes"][i]["free_bytes"] for s in samples) for i in (0, 1)
        },
        "min_host_available_bytes": min(s["memory"]["MemAvailable"] for s in samples),
        "max_rss_bytes": max(s["process"].get("VmRSS", 0) for s in samples),
        "max_process_swap_bytes": max(s["process"].get("VmSwap", 0) for s in samples),
        "max_system_swap_bytes": max(s["memory"]["SwapTotal"] - s["memory"]["SwapFree"] for s in samples),
        "sample_interval_seconds": {
            "min": min(b["elapsed_seconds"] - a["elapsed_seconds"] for a, b in zip(samples, samples[1:])),
            "max": max(b["elapsed_seconds"] - a["elapsed_seconds"] for a, b in zip(samples, samples[1:])),
        },
        "requests": {},
    }
    expected = "ORIGIN: 714\nCENTER: 928\nFINAL: 365"
    assert len(run["requests"]) == 10
    for request in run["requests"]:
        assert request["answer"].strip() == expected and request["reused"] == 0
        assert request["fresh"] == request["input_tokens"] <= 2048
        assert request["output_tokens"] < 32
    for size in ("small", "medium", "large"):
        requests = [r for r in measured if r["name"].endswith("-" + size)]
        assert len(requests) == 3
        result["requests"][size] = {
            "input_tokens": requests[0]["input_tokens"],
            "prefill_ms": [r["prefill_ms"] for r in requests],
            "wall_seconds_to_done": [r["wall_seconds_to_done"] for r in requests],
            "ttft_seconds_observed": [r["ttft_seconds_observed"] for r in requests],
            "decode_ms": [r["decode_ms"] for r in requests],
            "mtp_accepted_offered": [[int(x) for x in r["done"].split()[6:8]] for r in requests],
            "generated_ids": [r["generated_ids"] for r in requests],
            "median_prefill_ms": statistics.median(r["prefill_ms"] for r in requests),
            "median_wall_seconds_to_done": statistics.median(r["wall_seconds_to_done"] for r in requests),
            "zero_expert_read_deltas": all(
                r["counter_lines"][-1].endswith("for 0.0 MB of expert reads") for r in requests
            ),
        }
        assert result["requests"][size]["zero_expert_read_deltas"]
    log = (folder / "engine.log").read_text().splitlines()
    result["ownership_lines"] = [s for s in log if "gpu-owned-experts:" in s and (
        "owns layers" in s or "all 24576" in s)]
    assert any("all 24576 experts are resident" in s for s in result["ownership_lines"])
    return result


def main():
    runs = {mode: summarize(mode) for mode in ("control", "mmq")}
    manifests = {mode: json.loads((OUT / mode / "results.json").read_text()) for mode in runs}
    a, b = manifests["control"], manifests["mmq"]
    assert a["argv"] == b["argv"]
    assert a["engine_sha256"] == b["engine_sha256"]
    assert a["cache_sha256"] == b["cache_sha256"]
    assert a["kquant_library_sha256"] == b["kquant_library_sha256"]
    assert a["artifact_identity"] == b["artifact_identity"]
    assert a["guards"] == b["guards"] and a["guards"]["gpu_stop_mib"] == 640
    assert b["environment_overrides"] == {**a["environment_overrides"], "STRATA_MMQ_KQ": "1"}
    comparison = {}
    for size in ("small", "medium", "large"):
        x, y = (runs[mode]["requests"][size] for mode in ("control", "mmq"))
        comparison[size] = {
            "median_wall_speedup": x["median_wall_seconds_to_done"] / y["median_wall_seconds_to_done"],
            "median_prefill_speedup": x["median_prefill_ms"] / y["median_prefill_ms"],
            "generated_ids_identical": x["generated_ids"] == y["generated_ids"],
        }
    delta = k36_delta()
    suffix_increase = delta["1"]["required_free_before_model_load"]["delta_bytes"]
    fits = {}
    for mode, run in runs.items():
        free = run["min_gpu_free_bytes"]["0"]
        fits[mode] = {
            "k37_sampled_rtx_free_bytes": free,
            "surplus_above_640_mib_bytes": free - 640 * MIB,
            "k36_estimated_increment_bytes": suffix_increase,
            "k36_estimated_free_after_increment_bytes": free - suffix_increase,
            "deficit_against_640_mib_bytes": max(0, suffix_increase + 640 * MIB - free),
            "safe_capacity_established": False,
            "caveat": "Ledger granule estimates, not an exact physical peak or a K36 run.",
        }
    output = {
        "runs": runs, "comparison": comparison,
        "k36": {
            "ledger_path": str(LEDGER.relative_to(OUT.parent)),
            "ledger_sha256": hashlib.sha256(LEDGER.read_bytes()).hexdigest(),
            "component_deltas": delta, "against_resident_samples": fits,
            "layer36": {"kind": "GDN", "gu_type": "Q4_K", "down_type": "Q5_1",
                        "expert_blob_bytes": 3072000, "experts": 512,
                        "expert_payload_bytes": 1572864000},
        },
        "limits": [
            "Both arms timing disabled; sequential control then MMQ, not randomized/interleaved.",
            "Warmup excluded; startup and OS cache order effects are not eliminated.",
            "No near-max input, K36 run, split optimum claim or generalized quality validation.",
            "NVML physical free bytes sampled about once per second, not a hard OOM guarantee.",
            "Monitor threshold 640 binary MiB; configured allocation reserves remain 1024 MiB.",
        ],
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
