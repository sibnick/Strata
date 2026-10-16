#!/usr/bin/env python3
"""Summarize short control/profile runs; never sum overlapping stage wall times."""
import json
import re
import statistics
from pathlib import Path


OUT = Path(__file__).resolve().parent
TIMING = re.compile(
    r"^strata prefill timing: (\d+) tokens, GPU timeline ([\d.]+) ms, "
    r"wall ([\d.]+) ms, host staging ([\d.]+) ms:(.*)$"
)


def stage_times(lines: list[str]) -> dict:
    stages = {"CMP_prefix": [], "RTX_suffix": []}
    for line in lines:
        match = TIMING.match(line)
        if not match:
            continue
        count, gpu_ms, wall_ms, host_ms, phases = match.groups()
        count = int(count)
        record = {
            "tokens": count, "gpu_timeline_ms": float(gpu_ms), "wall_ms": float(wall_ms),
            "phase_ms": {
                name.strip(): float(ms)
                for name, ms, _ in re.findall(r"\s+([^()]+?) ([\d.]+) \(([\d.]+)%\)", phases)
            },
        }
        stages["CMP_prefix" if count > 256 else "RTX_suffix"].append(record)
    result = {}
    for name, records in stages.items():
        if not records:
            raise ValueError(f"no timing records for {name}")
        phases = {}
        for record in records:
            for phase, ms in record["phase_ms"].items():
                phases[phase] = phases.get(phase, 0) + ms
        result[name] = {
            "calls": len(records),
            "tokens": sum(r["tokens"] for r in records),
            "sum_gpu_timeline_ms": sum(r["gpu_timeline_ms"] for r in records),
            "sum_wall_ms": sum(r["wall_ms"] for r in records),
            "phase_ms": phases,
        }
    if result["CMP_prefix"]["calls"] != 1:
        raise ValueError("expected one multi-chunk prefix call")
    if result["CMP_prefix"]["tokens"] != result["RTX_suffix"]["tokens"]:
        raise ValueError("prefix/suffix processed token mismatch")
    return result


def main():
    output = {
        "limits": [
            "CUDA-event phases include stream gaps; not pure kernel compute times.",
            "Stage wall times overlap and must not be added as request wall time.",
            "Timing lines rounded to milliseconds; phase sums can differ from reported total.",
            "Stage attribution uses one >256-token prefix call versus <=256-token suffix calls.",
            "Host staging counter is cumulative in source and is not summed.",
            "Warmup excluded; separate serial startups, not randomized trials.",
        ],
        "runs": {},
    }
    for folder in sorted(OUT.iterdir()):
        if not folder.is_dir() or not (folder / "results.json").exists():
            continue
        run = json.loads((folder / "results.json").read_text())
        samples = [json.loads(line) for line in (folder / "monitor.jsonl").read_text().splitlines()]
        measured = [r for r in run["requests"] if r["name"] != "warmup"]
        result = {
            "status": run["status"],
            "error": run.get("error"),
            "engine_return_code": run.get("engine_return_code"),
            "startup_seconds": run.get("startup_seconds"),
            "host_min_available_gib": min(s["memory"]["MemAvailable"] for s in samples) / (1 << 30),
            "max_process_rss_bytes": max(s["process"].get("VmRSS", 0) for s in samples),
            "max_process_swap_bytes": max(s["process"].get("VmSwap", 0) for s in samples),
            "max_system_swap_used_bytes": max(
                s["memory"]["SwapTotal"] - s["memory"]["SwapFree"] for s in samples
            ),
            "sample_interval_seconds": {
                "min": min(b["elapsed_seconds"] - a["elapsed_seconds"] for a, b in zip(samples, samples[1:])),
                "max": max(b["elapsed_seconds"] - a["elapsed_seconds"] for a, b in zip(samples, samples[1:])),
            },
            "gpu_min_free_mib_physical": {
                str(i): min(s["gpus"][i]["memory_free_mib"] for s in samples) for i in (0, 1)
            },
            "prompts": {},
        }
        log = (folder / "engine.log").read_text().splitlines()
        result["ownership_lines"] = [
            line for line in log
            if "gpu-owned-experts:" in line and ("owns layers" in line or "all 24576" in line)
        ]
        result["engine_file_io_lines"] = [
            line for line in log if "file tier I/O this request:" in line
        ]
        result["last_prefill_checkpoint"] = next(
            (line for line in reversed(log) if line.startswith("PP ")), None
        )
        if run["status"] == "failed" and not measured:
            result["incomplete_request"] = samples[-1]["phase"]
            output["runs"][folder.name] = result
            continue
        for size in ("small", "medium", "large"):
            requests = [r for r in measured if r["name"].endswith("-" + size)]
            if len(requests) != 3:
                raise ValueError(f"{folder.name}: expected 3 {size} repetitions")
            data = {
                "input_tokens": requests[0]["input_tokens"],
                "prefill_ms": [r["prefill_ms"] for r in requests],
                "median_prefill_ms": statistics.median(r["prefill_ms"] for r in requests),
                "median_wall_seconds_to_done": statistics.median(r["wall_seconds_to_done"] for r in requests),
                "decode_ms": [r["decode_ms"] for r in requests],
                "mtp_accepted_offered": [
                    [int(r["done"].split()[6]), int(r["done"].split()[7])] for r in requests
                ],
                "answers_match": len({r["answer"].strip() for r in requests}) == 1,
                "no_reuse": all(r["reused"] == 0 for r in requests),
            }
            if requests[0]["profile_lines"]:
                stages = [stage_times(r["profile_lines"]) for r in requests]
                data["median_stage_totals"] = {
                    name: {
                        key: statistics.median(s[name][key] for s in stages)
                        for key in ("calls", "tokens", "sum_gpu_timeline_ms", "sum_wall_ms")
                    } for name in ("CMP_prefix", "RTX_suffix")
                }
                data["median_phase_ms"] = {
                    name: {
                        phase: statistics.median(s[name]["phase_ms"].get(phase, 0) for s in stages)
                        for phase in sorted({p for s in stages for p in s[name]["phase_ms"]})
                    } for name in ("CMP_prefix", "RTX_suffix")
                }
            result["prompts"][size] = data
        output["runs"][folder.name] = result
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
