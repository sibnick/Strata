#!/usr/bin/env python3
"""Offline summary of authorized long-MMQ run and historical same-prompt control."""
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ORIGINAL = OUT.parent / "2026-10-16-cmp170hx-rtx3090-gpu-owned-long-context"


def resources(samples):
    return {
        "min_gpu_free_bytes": {
            str(i): min(s["gpu_memory_bytes"][i]["free_bytes"] for s in samples) for i in (0, 1)
        },
        "min_host_available_bytes": min(s["memory"]["MemAvailable"] for s in samples),
        "max_rss_bytes": max(s["process"].get("VmRSS", 0) for s in samples),
        "max_process_swap_bytes": max(s["process"].get("VmSwap", 0) for s in samples),
        "max_system_swap_bytes": max(s["memory"]["SwapTotal"] - s["memory"]["SwapFree"] for s in samples),
    }


def main():
    folder = OUT / "mmq"
    run = json.loads((folder / "results.json").read_text())
    prior_short = json.loads((OUT.parent / "2026-10-16-cmp170hx-rtx3090-mmq-640/mmq/results.json").read_text())
    assert run["argv"] == prior_short["argv"]
    for key in ("environment_overrides", "engine_sha256", "cache_sha256",
                "kquant_library_sha256", "artifact_identity"):
        assert run[key] == prior_short[key]
    assert run["guards"]["gpu_stop_mib_physical"] == {"0": 640, "1": 640}
    assert "STRATA_PREFILL_TIMING" not in run["environment_overrides"]
    samples = [json.loads(s) for s in (folder / "monitor.jsonl").read_text().splitlines()]
    log = (folder / "engine.log").read_text().splitlines()
    output = {
        "status": run["status"], "error": run.get("error"),
        "engine_return_code": run.get("engine_return_code"),
        "startup_seconds": run.get("startup_seconds"),
        "overall_resources": resources(samples),
        "max_sample_gap_seconds": max(b["elapsed_seconds"] - a["elapsed_seconds"] for a, b in zip(samples, samples[1:])),
        "ownership_lines": [s for s in log if "gpu-owned-experts:" in s and ("owns layers" in s or "all 24576" in s)],
        "last_checkpoint": next((s for s in reversed(log) if s.startswith("PP ")), None),
        "active_request_at_stop": run.get("active_request"),
        "requests": [],
        "limits": [
            "Historical non-MMQ control, not randomized/interleaved; binary rebuilt, runtime source unchanged.",
            "Clock/cache/thermal/host/desktop differences not controlled.",
            "Profiler disabled; physical memory sampled, not guaranteed peaks or OOM prevention.",
            "One near-limit request only; no repeat/MTP-off/placement/expert-service comparison.",
        ],
    }
    historical = json.loads((ORIGINAL / "main-results.json").read_text())
    for req in run["requests"]:
        name = req["name"]
        meta = req["prompt_metadata"]
        assert req["fresh"] == req["input_tokens"] and req["reused"] == 0
        assert req["output_tokens"] < meta["generation_cap"]
        assert req["answer"].strip() == "\n".join(
            f"{key}: {meta['expected'][key]}" for key in ("ORIGIN", "CENTER", "FINAL")
        )
        stage_samples = [s for s in samples if s["phase"] == name]
        result = {
            "name": name, "input_tokens": req["input_tokens"],
            "output_tokens": req["output_tokens"], "cap": meta["generation_cap"],
            "prefill_ms": req["prefill_ms"], "prefill_tokens_per_second": req["prefill_tokens_per_second"],
            "wall_seconds_to_done": req["wall_seconds_to_done"],
            "ttft_seconds_observed": req["ttft_seconds_observed"],
            "decode_ms": req["decode_ms"],
            "decode_tokens_per_second_engine_convention": req["output_tokens"] * 1000 / req["decode_ms"],
            "mtp_accepted": req["mtp_accepted"], "mtp_offered": req["mtp_offered"],
            "resources": resources(stage_samples),
            "counter_lines": req["counter_lines"],
            "pcie": req["pcie"],
        }
        if name != "warmup":
            assert req["counter_lines"][-1].endswith("for 0.0 MB of expert reads")
            fixture = ORIGINAL / "prompts" / f"{name}.ids"
            assert hashlib.sha256(fixture.read_bytes()).hexdigest() == req["prompt_ids_sha256"]
            old = next(r for r in historical["requests"] if r["name"] == name)
            result["historical_non_mmq"] = {
                "prefill_ms": old["prompt_ms"], "prefill_tokens_per_second": old["prefill_tokens_per_second"],
                "ttft_seconds": old["ttft_seconds"], "wall_seconds_to_done": old["elapsed_seconds"],
                "mtp_accepted": old["mtp_drafts_accepted"], "mtp_offered": old["mtp_drafts_offered"],
                "prefill_speedup": old["prompt_ms"] / req["prefill_ms"],
                "ttft_speedup": old["ttft_seconds"] / req["ttft_seconds_observed"],
                "generated_ids_identical": req["generated_ids"] == old["generated_token_ids"],
            }
        output["requests"].append(result)
    if run["status"] == "passed":
        assert run["engine_return_code"] == 0
        assert [r["name"] for r in run["requests"]] == ["warmup", "32k", "near-max"]
        assert any("all 24576 experts are resident" in s for s in output["ownership_lines"])
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
