#!/usr/bin/env python3
"""Validate the K36 2048/512 run and summarize measured evidence."""
import json
from pathlib import Path
import statistics
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
MIB = 1 << 20
GIB = 1 << 30


def main():
    offline = json.loads((OUT / "offline.json").read_text())
    run_dir = OUT / "mixed2048-512"
    result = json.loads((run_dir / "results.json").read_text())
    if result["status"] != "passed" or result.get("engine_pid") is None:
        raise RuntimeError(f"GPU run did not pass: {result['status']}")
    requests = result["requests"]
    if [r["name"] for r in requests] != ["warmup", "short-1", "short-2", "short-3", "32k"]:
        raise RuntimeError("unexpected request schedule")
    if any(r["stage_chunks"] != [2048, 512] for r in requests):
        raise RuntimeError("per-stage chunks differ from requested values")
    if any(r["fresh"] != (32788 if r["name"] == "32k" else 1977) or r["reused"] != 0 for r in requests):
        raise RuntimeError("a request did not process its full fresh input")
    if not requests[0].get("first_stage_handoff", "").count("4 sub-chunks (512-token stage chunks)"):
        raise RuntimeError("larger handoff subdivision was not recorded")
    if any(not r["counter_lines"][-1].endswith("for 0.0 MB of expert reads") for r in requests[1:]):
        raise RuntimeError("a measured request reread expert source data")
    warmup_reads = requests[0]["counter_lines"][-1]
    if "for 77024.6 MB of expert reads" not in warmup_reads:
        raise RuntimeError(f"unexpected warmup expert-source read counter: {warmup_reads}")
    if result["engine_return_code"] != 0 or result["compute_processes_after"]:
        raise RuntimeError("engine/helper cleanup failed")
    if {entry["physical_index"] for entry in result["after_memory"]} != {0, 1}:
        raise RuntimeError("missing final GPU memory sample")

    monitor = [json.loads(line) for line in (run_dir / "monitor.jsonl").read_text().splitlines()]
    if not monitor or any(row["memory"]["MemAvailable"] < 5 * GIB for row in monitor):
        raise RuntimeError("host stop threshold was crossed")
    if any(row["other_compute"] for row in monitor):
        raise RuntimeError("unrelated compute process was observed")
    min_free = {
        str(gpu): min(
            next(item["free_bytes"] for item in row["gpu_memory_bytes"] if item["physical_index"] == gpu)
            for row in monitor
        )
        for gpu in (0, 1)
    }
    min_host = min(row["memory"]["MemAvailable"] for row in monitor)
    shorts = [r["prefill_ms"] for r in requests if r["name"].startswith("short-")]
    long_ms = next(r["prefill_ms"] for r in requests if r["name"] == "32k")

    prior_path = OUT.parent / "2026-10-17-cmp170hx-rtx3090-k37-mmq-chunks768-1024/chunk1024/results.json"
    prior = json.loads(prior_path.read_text())
    prior_shorts = [r["prefill_ms"] for r in prior["requests"] if r["name"].startswith("short-")]
    prior_32k = next(r["prefill_ms"] for r in prior["requests"] if r["name"] == "32k")
    checks = {
        "offline_sizing_k36_k37": "passed",
        "mandatory_prompt_init_headroom_k36": "passed",
        "startup_configuration_and_allocation_admission": "passed",
        "short_historical_output_ids_and_recall": "passed",
        "32788_token_historical_output_ids_and_recall": "passed",
        "full_fresh_prompt_input_no_reuse": "passed",
        "larger_handoff_consumed_in_512_token_subchunks": "passed",
        "post_warmup_measured_expert_source_reads": "zero",
        "host_start_at_least_6_gib": "passed",
        "host_stop_below_5_gib": "passed",
        "compute_process_cleanup": "passed",
        "cuda_build_targets_strata_strata_kq_and_split_memory": "passed",
        "split_memory_self_test": "passed",
        "per_stage_cpu_contract_tests": "passed",
        "hip_and_sycl": "not_tested",
        "checksums": "recorded in SHA256SUMS",
    }
    cleanup = {
        "engine_return_code": result["engine_return_code"],
        "pcie_helper_return_code": result["pcie_helper_return_code"],
        "compute_processes_after": result["compute_processes_after"],
        "final_free_memory_by_physical_index": result["after_memory"],
        "under_load_minimum_free_memory_by_physical_index": min_free,
        "host_memavailable_minimum_bytes": min_host,
    }
    summary = {
        "status": result["status"],
        "gpu_workload_started": True,
        "configuration": {
            "layer_split": 36,
            "stage_chunks": [2048, 512],
            "context_cells": 262144,
            "kv": "int8",
            "discretionary_gpu_reserve_mib": [0, 0],
            "external_gpu_free_abort_mib": None,
            "mandatory_prompt_init_headroom_mib": 512,
            "host_start_gib": 6,
            "host_stop_gib": 5,
        },
        "offline_sizing": offline["sizing"],
        "k36_admission": result["admission"],
        "k37_offline_comparison": "Both stages passed offline admission; K37 was not run.",
        "mixed_k36_short_prefill_ms": shorts,
        "mixed_k36_short_prefill_median_ms": statistics.median(shorts),
        "mixed_k36_32788_token_prefill_ms": long_ms,
        "warmup_expert_source_reads_mb": 77024.6,
        "prior_uniform_1024_k37_descriptive_only": {
            "short_prefill_ms": prior_shorts,
            "short_prefill_median_ms": statistics.median(prior_shorts),
            "single_32788_token_prefill_ms": prior_32k,
            "source": str(prior_path.relative_to(ROOT)),
            "not_a_contemporaneous_control": True,
        },
        "under_load_minimum_free_vram_bytes": min_free,
        "under_load_minimum_host_memavailable_bytes": min_host,
        "cleanup": cleanup,
        "checks": checks,
    }
    provenance = {
        "source_commit": result["source_commit"],
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "engine_sha256": offline["engine_sha256"],
        "kquant_library_sha256": offline["kquant_library_sha256"],
        "source_sha256": offline["source_sha256"],
        "offline_command": offline["command"],
        "environment_overrides": offline["environment_overrides"],
        "model_artifact_paths": {
            "pack": offline["command"][2],
            "native": offline["command"][4],
            "mtp": offline["command"][6],
        },
        "runner": "run.py; bounded K36 2048/512 sequence",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (OUT / "checks.json").write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n")
    (OUT / "cleanup.json").write_text(json.dumps(cleanup, indent=2, sort_keys=True) + "\n")
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(f"K36 passed: short median {statistics.median(shorts):.1f} ms, 32K {long_ms:.1f} ms; "
          f"minimum free VRAM {min_free}, host MemAvailable {min_host} bytes.")


if __name__ == "__main__":
    main()
