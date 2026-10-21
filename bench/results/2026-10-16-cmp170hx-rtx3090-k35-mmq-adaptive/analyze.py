#!/usr/bin/env python3
"""Offline checks and descriptive K35 adaptive A/B summary."""
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics
import struct


OUT = Path(__file__).resolve().parent
GIB = 1 << 30
MIB = 1 << 20
GPU_STOP = {"0": 256 * MIB, "1": 640 * MIB}
EXPECTED_SHORT = "ORIGIN: 714\nCENTER: 928\nFINAL: 365"
EXPECTED_LONG = (
    "ORIGIN: IRIS-4Q7-LANTERN\n"
    "CENTER: CEDAR-8M2-QUARRY\n"
    "FINAL: ORBIT-6V9-KETTLE"
)


def read_arm(name):
    folder = OUT / name
    result = json.loads((folder / "results.json").read_text())
    if result.get("status") != "passed" or result.get("engine_return_code") != 0:
        raise ValueError(f"{name} did not pass: {result.get('error')}")
    if "STRATA_TRACE" in result["environment_overrides"]:
        raise ValueError(f"{name} unexpectedly enabled tracing")
    if "STRATA_PREFILL_TIMING" in result["environment_overrides"]:
        raise ValueError(f"{name} unexpectedly enabled prefill timing")
    if "--gpu-owned-experts" in result["argv"] or "--expert-cache" in result["argv"]:
        raise ValueError(f"{name} did not use the approved partial AUTO-cache mode")
    if result["startup_gate"] is None:
        raise ValueError(f"{name} has no validated startup gate")

    samples = [
        json.loads(line) for line in (folder / "monitor.jsonl").read_text().splitlines()
    ]
    if len(samples) < 2:
        raise ValueError(f"{name} has too few memory samples")
    for sample in samples:
        if sample["memory"]["MemAvailable"] < 5 * GIB:
            raise ValueError(f"{name} crossed the host stop threshold")
        for gpu in sample["gpu_memory_bytes"]:
            if gpu["free_bytes"] < GPU_STOP[str(gpu["physical_index"])]:
                raise ValueError(f"{name} crossed GPU {gpu['physical_index']} stop threshold")
        if sample["other_compute"]:
            raise ValueError(f"{name} overlapped another GPU compute process")
    if not result["requests"]:
        raise ValueError(f"{name} has no completed requests")
    for request in result["requests"]:
        expected = EXPECTED_LONG if request["name"] == "sanity-32k" else EXPECTED_SHORT
        if request["answer"].strip() != expected:
            raise ValueError(f"{name}/{request['name']} retrieval mismatch")
        if request["reused"] != 0 or request["fresh"] != request["input_tokens"]:
            raise ValueError(f"{name}/{request['name']} reused or skipped prompt tokens")
        if request["output_tokens"] >= request["generation_cap"]:
            raise ValueError(f"{name}/{request['name']} reached its generation cap")
        if request["name"] != "warmup":
            if not request["counter_lines"][-1].endswith("for 0.0 MB of expert reads"):
                raise ValueError(f"{name}/{request['name']} read cold experts from files")
    expected_names = 11 if name == "adaptive" else 10
    if len(result["requests"]) != expected_names:
        raise ValueError(f"{name} completed {len(result['requests'])} requests, expected {expected_names}")

    gaps = [
        right["elapsed_seconds"] - left["elapsed_seconds"]
        for left, right in zip(samples, samples[1:])
    ]
    min_free = {
        str(index): min(
            sample["gpu_memory_bytes"][index]["free_bytes"] for sample in samples
        )
        for index in (0, 1)
    }
    return result, {
        "sample_count": len(samples),
        "maximum_sample_gap_seconds": max(gaps),
        "minimum_host_available_bytes": min(
            sample["memory"]["MemAvailable"] for sample in samples
        ),
        "maximum_rss_bytes": max(sample["process"].get("VmRSS", 0) for sample in samples),
        "maximum_process_swap_bytes": max(sample["process"].get("VmSwap", 0) for sample in samples),
        "maximum_system_swap_bytes": max(
            sample["memory"]["SwapTotal"] - sample["memory"]["SwapFree"] for sample in samples
        ),
        "minimum_gpu_free_bytes": min_free,
        "compute_processes_after": result["compute_processes_after"],
    }


def normalize_adaptive_argv(argv):
    result = []
    skip = False
    for token in argv:
        if skip:
            skip = False
            continue
        if token in ("--adapt-every", "--adapt-swaps", "--adapt-async"):
            result.append(token)
            skip = True
        else:
            result.append(token)
    return result


def extract_adaptive_metrics(result):
    folder = OUT / result["arm"]
    all_lines = (folder / "engine.log").read_text(errors="replace").splitlines()
    lines = [
        line for line in all_lines
        if any(term in line.lower() for term in (
            "adaptive tier", "expert tiers:", "cpu experts", "decode expert cache hit rate",
            "resident ram:", "file tier i/o this request:", "accepted per round",
        ))
    ]
    per_request = []
    previous_exchanges = 0
    previous_ram_blobs = 0
    for request in result["requests"]:
        diagnostics = request.get("adaptive_diagnostics", [])
        cache_metrics = None
        for line in diagnostics:
            match = re.search(
                r"decode expert cache hit rate: ([\d.]+)% \((\d+) hits / (\d+) lookups\); "
                r"(\d+) more read by the GPU over PCIe or from another GPU "
                r"\(([\d.]+)% of all (\d+) routed\)",
                line,
            )
            if match:
                cache_metrics = {
                    "hit_rate_percent": float(match[1]),
                    "gpu_cache_hits": int(match[2]),
                    "gpu_cache_lookups": int(match[3]),
                    "other_gpu_reads": int(match[4]),
                    "other_gpu_read_percent": float(match[5]),
                    "routed_experts": int(match[6]),
                }
        tier_counters = []
        for line in diagnostics:
            match = re.search(
                r"expert tiers: GPU (\d+) hits this request; since the start RAM "
                r"(\d+) blobs, files (\d+) blobs ([\d.]+) MB read",
                line,
            )
            if match:
                tier_counters.append({
                    "gpu_cache_hits_this_request": int(match[1]),
                    "resident_ram_blobs_cumulative": int(match[2]),
                    "file_blobs_cumulative": int(match[3]),
                    "file_expert_mb_cumulative": float(match[4]),
                })
        tier_counter = tier_counters[-1] if tier_counters else None
        ram_blobs_this_request = None
        if tier_counter is not None:
            ram_blobs_this_request = (
                tier_counter["resident_ram_blobs_cumulative"] - previous_ram_blobs
            )
            previous_ram_blobs = tier_counter["resident_ram_blobs_cumulative"]
        exchanges = [
            int(match.group(1))
            for line in diagnostics
            if (match := re.search(
                r"resident RAM: [\d.]+ GiB of experts in RAM, "
                r"(\d+) exchanged with the VRAM tier",
                line,
            ))
        ]
        cumulative_exchanges = exchanges[-1] if exchanges else None
        exchanges_this_request = (
            cumulative_exchanges - previous_exchanges if cumulative_exchanges is not None else None
        )
        if cumulative_exchanges is not None:
            previous_exchanges = cumulative_exchanges
        file_read_line = request["counter_lines"][-1]
        file_read_match = re.search(r"for ([\d.]+) MB of expert reads$", file_read_line)
        per_request.append({
            "request": request["name"],
            "profile_cache_metrics": cache_metrics,
            "tier_counters": tier_counter,
            "resident_ram_blobs_this_request": ram_blobs_this_request,
            "exchange_events_cumulative": cumulative_exchanges,
            "exchange_events_this_request": exchanges_this_request,
            "expert_file_read_mb_reported": (
                float(file_read_match[1]) if file_read_match else None
            ),
            "expert_file_read_note": (
                "The warmup counter includes startup-cumulative reads; subsequent request "
                "counters report their measured expert-read delta."
                if request["name"] == "warmup" else None
            ),
            "adaptive_lines": diagnostics,
            "profile_cache_and_tier_lines": [
                line for line in diagnostics
                if any(term in line.lower() for term in (
                    "expert tiers:", "cpu experts", "decode expert cache hit rate",
                    "resident ram:",
                ))
            ],
            "file_read_line": file_read_line,
        })
    return {
        "per_request": per_request,
        "all_relevant_engine_lines": lines,
        "synchronous_adaptation_rounds": None,
        "synchronous_adaptation_rounds_note": (
            "The unchanged request-end diagnostics expose cumulative exchange counts, but no "
            "per-request adaptation-round count or accepted-round histogram."
        ),
        "per_stage_swap_bytes": None,
        "per_stage_swap_bytes_note": (
            "The unchanged runtime emits aggregate exchange counts and no layer IDs or "
            "per-stage byte totals. No trace/profiling instrumentation or runtime change "
            "was authorized, so per-stage swap bytes are not inferred."
        ),
    }


def exact_cold_complement(results):
    control, adaptive = results
    control_gate = control["startup_gate"]
    adaptive_gate = adaptive["startup_gate"]
    for key in ("stage1_cache", "cold_complement"):
        if control_gate[key] != adaptive_gate[key]:
            raise ValueError(f"A/B actual {key} startup accounting differs")

    profile_artifact = next(
        artifact for artifact in control["prior_artifact_identity"]["artifacts"]
        if Path(artifact["resolved_path"]).name == "expert-profile.bin"
    )
    profile_path = Path(profile_artifact["resolved_path"])
    profile_data = profile_path.read_bytes()
    profile_sha = hashlib.sha256(profile_data).hexdigest()
    if profile_sha != profile_artifact["recorded_sha256"]:
        raise ValueError("expert profile differs from the recorded artifact identity")
    if profile_data[:4] != b"STRP":
        raise ValueError("expert profile does not have the STRP header")
    version, n_layers, n_experts, _, n_ranked = struct.unpack_from("<5I", profile_data, 4)
    if (version, n_layers, n_experts, n_ranked) != (1, 48, 512, 24576):
        raise ValueError("expert profile dimensions differ from the K35 startup")
    pairs = [
        struct.unpack_from("<HH", profile_data, 24 + i * 4)
        for i in range(n_ranked)
    ]
    stage_pairs = [pair for pair in pairs if 35 <= pair[0] < 48]
    cache_slots = control_gate["stage1_cache"]["slots"]
    cold_pairs = stage_pairs[cache_slots:]
    if len(stage_pairs) != control_gate["stage1_cache"]["profiled_pairs"]:
        raise ValueError("expert profile stage-1 pair count differs from startup")
    if len(cold_pairs) != control_gate["cold_complement"]["reported_unowned_pairs"]:
        raise ValueError("exact cold-pair set differs from startup ownership accounting")

    ledger_path = OUT.parent / "2026-10-16-cmp170hx-rtx3090-k37" / "ledger-chunk256.csv"
    layer_totals = {}
    with ledger_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            match = re.fullmatch(r"layer_(\d+)_expert_payload", row["component"])
            if match:
                layer = int(match[1])
                if layer in layer_totals:
                    raise ValueError(f"duplicate expert-payload ledger row for layer {layer}")
                layer_totals[layer] = int(row["bytes"])

    cold_counts = {}
    for layer, _expert in cold_pairs:
        cold_counts[layer] = cold_counts.get(layer, 0) + 1
    by_layer = {}
    for layer, count in sorted(cold_counts.items()):
        total = layer_totals.get(layer)
        if total is None or total % n_experts:
            raise ValueError(f"layer {layer} has no exact per-expert payload size in the ledger")
        by_layer[str(layer)] = count * (total // n_experts)
    payload_bytes = sum(by_layer.values())
    for arm in results:
        complement = arm["startup_gate"]["cold_complement"]
        if not complement["fully_page_locked"]:
            raise ValueError(f"{arm['arm']} did not fully page-lock the cold complement")
        for key in ("resident_gib", "pinned_gib"):
            if abs(complement[key] - payload_bytes / GIB) > 0.01:
                raise ValueError(f"{arm['arm']} rounded {key} disagrees with exact profile payload")

    return {
        "profile_sha256": profile_sha,
        "ledger": str(ledger_path.relative_to(OUT.parents[2])),
        "ledger_sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
        "stage1_profiled_pairs": len(stage_pairs),
        "stage1_gpu_cache_pairs": cache_slots,
        "cold_pairs": len(cold_pairs),
        "cold_pairs_by_layer": {str(k): v for k, v in sorted(cold_counts.items())},
        "cold_payload_bytes_by_layer": by_layer,
        "resident_expert_payload_bytes_exact": payload_bytes,
        "pinned_expert_payload_bytes_exact": payload_bytes,
        "pinned_bytes_basis": (
            "The runtime reported a mapped pinned complement and matching resident/pinned sizes; "
            "the exact logical payload is the profile-selected cold-pair sum using the recorded "
            "per-layer expert payload sizes."
        ),
    }


def main():
    control, control_memory = read_arm("control")
    adaptive, adaptive_memory = read_arm("adaptive")
    if normalize_adaptive_argv(control["argv"]) != normalize_adaptive_argv(adaptive["argv"]):
        raise ValueError("A/B engine argv differ beyond the adaptive schedule")
    if control["engine_sha256"] != adaptive["engine_sha256"]:
        raise ValueError("A/B engine binary hashes differ")
    if control["kquant_library_sha256"] != adaptive["kquant_library_sha256"]:
        raise ValueError("A/B MMQ library hashes differ")
    if control["prior_artifact_identity"] != adaptive["prior_artifact_identity"]:
        raise ValueError("A/B model/profile artifact identities differ")
    if control["environment_overrides"] != adaptive["environment_overrides"]:
        raise ValueError("A/B environment overrides differ")
    if control["scheduler"] != {"adapt_every": 0, "adapt_swaps": 0, "adapt_async": 0}:
        raise ValueError("control arm is not adaptive-off")
    if adaptive["scheduler"] != {"adapt_every": 4, "adapt_swaps": 96, "adapt_async": 0}:
        raise ValueError("adaptive arm does not match its approved schedule")
    complement_exact = exact_cold_complement((control, adaptive))

    short = {}
    for name in ("small", "medium", "large"):
        left = [r for r in control["requests"] if r["name"].endswith("-" + name)]
        right = [r for r in adaptive["requests"] if r["name"].endswith("-" + name)]
        if len(left) != 3 or len(right) != 3:
            raise ValueError(f"{name} does not have three measured repetitions per arm")
        if [r["prompt_ids_sha256"] for r in left] != [r["prompt_ids_sha256"] for r in right]:
            raise ValueError(f"{name} prompt IDs differ between arms")
        if [r["generated_ids"] for r in left] != [r["generated_ids"] for r in right]:
            raise ValueError(f"{name} generated token IDs differ between arms")
        left_times = [r["wall_seconds_to_done"] for r in left]
        right_times = [r["wall_seconds_to_done"] for r in right]
        short[name] = {
            "input_tokens": left[0]["input_tokens"],
            "control_median_prefill_ms": statistics.median(r["prefill_ms"] for r in left),
            "adaptive_median_prefill_ms": statistics.median(r["prefill_ms"] for r in right),
            "control_median_wall_seconds": statistics.median(left_times),
            "adaptive_median_wall_seconds": statistics.median(right_times),
            "adaptive_over_control_wall_ratio": (
                statistics.median(right_times) / statistics.median(left_times)
            ),
            "generated_ids_match": True,
            "mtp_accepted_offered": {
                "control": [r["mtp_accepted_offered"] for r in left],
                "adaptive": [r["mtp_accepted_offered"] for r in right],
            },
        }

    long = adaptive["requests"][-1]
    if long["name"] != "sanity-32k" or long["input_tokens"] != 32788:
        raise ValueError("adaptive arm lacks the one approved 32,788-token sanity prompt")
    if long["generation_cap"] != 192:
        raise ValueError("32K sanity prompt generation cap differs from its fixture")

    output = {
        "result": "passed",
        "description": (
            "Descriptive K35 A/B: ordinary CPU-computed resident-RAM misses versus "
            "synchronous adaptive RAM-to-GPU promotion; MMQ enabled in both."
        ),
        "arms": {"control": control_memory, "adaptive": adaptive_memory},
        "short_comparison": short,
        "adaptive_32k_sanity": {
            "input_tokens": long["input_tokens"],
            "generation_cap": long["generation_cap"],
            "prefill_ms": long["prefill_ms"],
            "wall_seconds_to_done": long["wall_seconds_to_done"],
            "output_tokens": long["output_tokens"],
            "mtp_accepted_offered": long["mtp_accepted_offered"],
            "answer": long["answer"],
            "prompt_ids_sha256": long["prompt_ids_sha256"],
        },
        "startup": {
            "control": control["startup_gate"],
            "adaptive": adaptive["startup_gate"],
            "RTX_later_reserve_mib": 2049,
            "CMP_first_reserve_mib": 1024,
        },
        "resident_complement_exact": complement_exact,
        "adaptive_metrics": {
            "control": extract_adaptive_metrics(control),
            "adaptive": extract_adaptive_metrics(adaptive),
        },
        "limits": [
            "Serial arms are descriptive, not randomized or interleaved.",
            "The 32K prompt is one sanity request, not a long-context performance claim.",
            "One-second samples are sampled extrema, not a hard OOM guarantee.",
            "Only CUDA was exercised; no default, runtime algorithm, split, or backend was changed.",
            "Adaptive per-stage swap bytes are not emitted by the unchanged runtime and are not inferred.",
        ],
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
