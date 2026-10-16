#!/usr/bin/env python3
"""Analyze K36 retry without executing GPU work."""
import json
from pathlib import Path
import statistics

OUT = Path(__file__).resolve().parent


def main():
    run = json.loads((OUT / "mmq/results.json").read_text())
    prior_dir = OUT.parent / "2026-10-16-cmp170hx-rtx3090-mmq-640"
    old = json.loads((prior_dir / "mmq/results.json").read_text())
    argv = old["argv"].copy()
    argv[argv.index("--layer-split") + 1] = "36"
    argv[argv.index("--vram-reserve-later-mib") + 1] = "256"
    assert run["argv"] == argv
    assert run["environment_overrides"] == old["environment_overrides"]
    for key in ("engine_sha256", "cache_sha256", "kquant_library_sha256", "artifact_identity"):
        assert run[key] == old[key]
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
    if run["status"] == "passed":
        assert run["engine_return_code"] == 0 and len(run["requests"]) == 10
        assert any("all 24576 experts are resident" in s for s in result["ownership_lines"])
    for req in run["requests"]:
        assert req["answer"].strip() == "ORIGIN: 714\nCENTER: 928\nFINAL: 365"
        assert req["reused"] == 0 and req["fresh"] == req["input_tokens"] <= 2048
        assert req["output_tokens"] < 32
    for size in ("small", "medium", "large"):
        requests = [r for r in run["requests"] if r["name"].endswith("-" + size)]
        if not requests:
            continue
        prior = [r for r in old["requests"] if r["name"].endswith("-" + size)]
        wall = statistics.median(r["wall_seconds_to_done"] for r in requests)
        result["requests"][size] = {
            "input_tokens": requests[0]["input_tokens"], "repetitions": len(requests),
            "prefill_ms": [r["prefill_ms"] for r in requests],
            "wall_seconds_to_done": [r["wall_seconds_to_done"] for r in requests],
            "ttft_seconds_observed": [r["ttft_seconds_observed"] for r in requests],
            "decode_ms": [r["decode_ms"] for r in requests],
            "mtp_accepted_offered": [[int(x) for x in r["done"].split()[6:8]] for r in requests],
            "median_prefill_ms": statistics.median(r["prefill_ms"] for r in requests),
            "median_wall_seconds_to_done": wall,
            "prior_k37_median_wall_seconds": statistics.median(r["wall_seconds_to_done"] for r in prior),
            "prior_k37_wall_over_k36": statistics.median(r["wall_seconds_to_done"] for r in prior) / wall,
            "generated_ids_match_k37": all(r["generated_ids"] == next(x for x in prior if x["name"] == r["name"])["generated_ids"] for r in requests),
            "zero_expert_read_deltas": all(r["counter_lines"][-1].endswith("for 0.0 MB of expert reads") for r in requests),
        }
        for req in requests:
            assert req["prompt_ids_sha256"] == next(x for x in prior if x["name"] == req["name"])["prompt_ids_sha256"]
    result["limits"] = [
        "K37 was an earlier serial run with different reserve/monitor; not randomized/interleaved or best-split evidence.",
        "Only <=2K input, <=32 output; allocated262144 capacity is not processed tokens.",
        "RTX reserve/monitor256MiB; CMP reserve1024/monitor640; host5GiB reaction unchanged.",
        "Periodic sampling cannot guarantee no transient pressure/OOM.",
    ]
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
