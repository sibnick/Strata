#!/usr/bin/env python3
"""Validate the offline admission decision and retain descriptive prior-run context."""
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
    preflight = json.loads((OUT / "mixed2048-1024/results.json").read_text())
    if offline["sizing"]["36"]["1"]["passes_mandatory_prompt_init_admission"]:
        raise RuntimeError("K36 RTX should fail mandatory 512 MiB prompt-init headroom")
    if preflight["status"] != "offline-infeasible" or preflight["gpu_workload_started"]:
        raise RuntimeError("preflight unexpectedly launched or passed a GPU workload")
    if preflight["preflight"]["compute_processes_before"]:
        raise RuntimeError("compute process was active at preflight")
    if preflight["preflight"]["host_memavailable_bytes"] < 6 * GIB:
        raise RuntimeError("host start guard was not met")
    if preflight["admission"]["1"]["margin_bytes"] != -297_172_204:
        raise RuntimeError("K36 RTX mandatory admission deficit changed")
    if preflight["preflight"]["engine_sha256"] != offline["engine_sha256"]:
        raise RuntimeError("preflight engine differs from offline sizing build")
    if preflight["preflight"]["engine_reserves_mib"] != [0, 0] or \
            preflight["preflight"]["external_gpu_free_abort_mib"] is not None:
        raise RuntimeError("GPU reserve policy differs from user authorization")

    prior_path = OUT.parent / "2026-10-17-cmp170hx-rtx3090-k37-mmq-chunks768-1024/chunk1024/results.json"
    prior = json.loads(prior_path.read_text())
    prior_shorts = [r["prefill_ms"] for r in prior["requests"] if r["name"].startswith("short-")]
    prior_32k = next(r["prefill_ms"] for r in prior["requests"] if r["name"] == "32k")

    gpu_rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,uuid,memory.free,memory.total",
         "--format=csv,noheader,nounits"], text=True
    ).splitlines()
    gpu_free = {}
    for line in gpu_rows:
        index, name, uuid, free, total = (part.strip() for part in line.split(","))
        gpu_free[index] = {
            "name": name, "uuid": uuid,
            "free_bytes": int(free) * MIB, "total_bytes": int(total) * MIB,
        }
    compute = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,name", "--format=csv,noheader"], text=True
    ).strip()
    if compute:
        raise RuntimeError(f"compute process remained after preflight: {compute}")
    cleanup = {
        "gpu_workload_started": False,
        "engine_pid": None,
        "pcie_helper_pid": None,
        "no_compute_processes_after_preflight": True,
        "compute_processes": [],
        "free_memory_after_preflight": gpu_free,
        "engine_return_code": None,
        "host_memavailable_bytes_at_preflight": preflight["preflight"]["host_memavailable_bytes"],
    }
    summary = {
        "status": "K36 GPU workload not started: mandatory admission failed",
        "gpu_workload_started": False,
        "configuration": {
            "layer_split": 36,
            "stage_chunks": [2048, 1024],
            "context_cells": 262144,
            "kv": "int8",
            "discretionary_gpu_reserve_mib": [0, 0],
            "external_gpu_free_abort_mib": None,
            "mandatory_prompt_init_headroom_mib": 512,
            "host_start_gib": 6,
            "host_stop_gib": 5,
        },
        "offline_sizing": offline["sizing"],
        "k36_admission": preflight["admission"],
        "k37_offline_comparison": "Both stages pass the same mandatory headroom check; K37 was not run.",
        "prior_uniform_1024_k37_descriptive_only": {
            "short_prefill_ms": prior_shorts,
            "short_prefill_median_ms": statistics.median(prior_shorts),
            "single_32788_token_prefill_ms": prior_32k,
            "source": str(prior_path.relative_to(ROOT)),
            "not_a_contemporaneous_control": True,
        },
        "mixed_k36_prefill_or_decode_timings": None,
        "mixed_k36_output_ids_or_handoff_runtime_verified": False,
        "under_load_minimum_free_vram": None,
        "idle_preflight_free_bytes": preflight["preflight"]["gpu_memory_by_physical_index"],
        "host_memavailable_bytes_at_preflight": preflight["preflight"]["host_memavailable_bytes"],
        "blocking_reason": "RTX 3090 is short 297,172,204 bytes of the mandatory 512 MiB prompt-init headroom.",
    }
    checks = {
        "cuda_build_targets_strata_strata_kq_and_split_memory": "passed",
        "split_memory_self_test": "passed",
        "per_stage_cpu_contract_tests": "passed",
        "source_ledger_k36_raw_free_memory_fit": "passed",
        "k36_mandatory_prompt_init_admission": "blocked_on_rtx3090",
        "k37_mandatory_prompt_init_admission": "passed_offline_only",
        "gpu_correctness_handoff_and_expert_read_tests": "not_run",
        "hip_and_sycl": "not_tested",
        "checksums": "recorded in SHA256SUMS",
    }
    provenance = {
        "source_commit": preflight["preflight"]["source_commit"],
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
        "preflight_runner": "run.py; stopped before engine/PCIe process creation",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (OUT / "checks.json").write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n")
    (OUT / "cleanup.json").write_text(json.dumps(cleanup, indent=2, sort_keys=True) + "\n")
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print("K36 preflight correctly blocked by mandatory RTX prompt-init headroom; no GPU workload or compute process.")


if __name__ == "__main__":
    main()
