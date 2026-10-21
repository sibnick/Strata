#!/usr/bin/env python3
"""Source-based sizing for a 2048/1024 per-stage prompt on K36 and K37."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
LEDGER = ROOT / "build/cuda-per-stage/strata-split-memory"
PACK = Path("/home/nikolay/projects/copilot-worktrees/170hx-3090/"
            "sibnick-upgraded-succotash/build/pack/ud-q4-k-xl")
NATIVE = Path("/home/nikolay/projects/Strata-data/models/UD-Q4_K_XL/UD-Q4_K_XL/"
              "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf")
MTP = Path("/home/nikolay/projects/Strata-data/mtp/rt")
ENVIRONMENT = {
    "STRATA_MMQ_KQ": "1",
    "STRATA_PREFILL_BATCH_GATHER": "1",
    "STRATA_SPLIT_RING": "96",
    "STRATA_FILL_AHEAD": "0",
    "STRATA_IO_PREFETCH": "0",
    "STRATA_PIN_RESERVE_GIB": "4",
    "STRATA_REFILL_BLOCKING": "1",
    "STRATA_RESIDENT_HEADROOM_GIB": "4",
    "STRATA_STAGE_PIN": "0",
    "STRATA_UNBUFFERED_LOAD": "1",
}
OWNED_PREFILL_ADMISSION_HEADROOM = 512 * (1 << 20)
SOURCES = (
    "src/program/generate.cpp",
    "src/prefill/prefill.cpp",
    "src/program/split_memory_ledger.cpp",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    for path in (LEDGER, PACK / "index.txt", NATIVE, MTP / "dense.bin"):
        if not path.exists():
            raise RuntimeError(f"missing sizing input: {path}")
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("STRATA_") and k not in ("CUDA_VISIBLE_DEVICES", "LD_PRELOAD")}
    env.update(ENVIRONMENT)
    command = [
        str(LEDGER), "--pack", str(PACK), "--native", str(NATIVE), "--mtp", str(MTP),
        "--context", "262144", "--chunk", "2048", "--prefill-stage-chunks", "2048,1024",
        "--additional-reserve-mib", "0",
    ]
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    if "K-quant prompt kernels from" not in result.stderr or "FP16 prompt path" in result.stderr:
        raise RuntimeError("sizing did not load the CUDA K-quant prompt kernels")
    (OUT / "ledger.csv").write_text(result.stdout)
    (OUT / "ledger.log").write_text(result.stderr)

    rows = {}
    for row in csv.DictReader(result.stdout.splitlines()):
        if row["split"] not in ("36", "37") or row["stage"] not in ("0", "1"):
            continue
        rows.setdefault(row["split"], {}).setdefault(row["stage"], {})[row["component"]] = int(row["bytes"])

    gpu_rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,uuid,memory.free,memory.total", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()
    physical = {}
    for row in gpu_rows:
        index, name, uuid, free_mib, total_mib = (part.strip() for part in row.split(","))
        physical[int(index)] = {
            "name": name, "uuid": uuid, "free_bytes": int(free_mib) * (1 << 20),
            "total_bytes": int(total_mib) * (1 << 20),
        }
    if "CMP 170HX" not in physical[1]["name"] or "RTX 3090" not in physical[0]["name"]:
        raise RuntimeError(f"unexpected physical GPU order: {physical}")
    stage_to_physical = {"0": 1, "1": 0}
    sizing = {}
    for boundary, stage_rows in rows.items():
        sizing[boundary] = {}
        for stage, components in stage_rows.items():
            gpu = physical[stage_to_physical[stage]]
            required = components["required_free_before_model_load"]
            required_with_prompt_headroom = required + OWNED_PREFILL_ADMISSION_HEADROOM
            sizing[boundary][stage] = {
                "chunk": 2048 if stage == "0" else 1024,
                "physical_gpu": stage_to_physical[stage],
                "gpu_name": gpu["name"],
                "free_bytes_before_run": gpu["free_bytes"],
                "total_bytes": gpu["total_bytes"],
                "owned_prefill_allocation_estimate": components["owned_prefill_allocation_estimate"],
                "required_free_before_model_load": required,
                "mandatory_prompt_init_headroom_bytes": OWNED_PREFILL_ADMISSION_HEADROOM,
                "required_with_mandatory_prompt_init_headroom": required_with_prompt_headroom,
                "admission_margin_bytes": gpu["free_bytes"] - required,
                "admission_margin_after_mandatory_headroom_bytes": gpu["free_bytes"] - required_with_prompt_headroom,
                "fits_current_idle_free_memory": gpu["free_bytes"] >= required,
                "passes_mandatory_prompt_init_admission": gpu["free_bytes"] >= required_with_prompt_headroom,
                "components": components,
            }
    data = {
        "command": command,
        "environment_overrides": ENVIRONMENT,
        "sizing_only": True,
        "gpu_workload_started": False,
        "context": 262144,
        "kv": "int8",
        "layer_boundaries": [36, 37],
        "per_stage_chunks": [2048, 1024],
        "cuda_reserve_mib": 0,
        "external_gpu_free_abort_mib": None,
        "stage_to_physical_gpu": stage_to_physical,
        "physical_gpu_snapshot": physical,
        "sizing": sizing,
        "engine_sha256": sha256(ROOT / "build/cuda-per-stage/strata"),
        "kquant_library_sha256": sha256(ROOT / "build/cuda-per-stage/libstrata_kq.so"),
        "source_sha256": {path: sha256(ROOT / path) for path in SOURCES},
    }
    (OUT / "offline.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    for boundary, stage_rows in sizing.items():
        for stage, values in stage_rows.items():
            print(f"K{boundary} stage{stage} {values['gpu_name']}: "
                  f"scratch={values['owned_prefill_allocation_estimate']} B, "
                  f"required={values['required_free_before_model_load']} B, "
                  f"required+512MiB={values['required_with_mandatory_prompt_init_headroom']} B, "
                  f"free={values['free_bytes_before_run']} B, "
                  f"margin={values['admission_margin_bytes']} B, "
                  f"margin-after-guard={values['admission_margin_after_mandatory_headroom_bytes']} B, "
                  f"raw-fits={values['fits_current_idle_free_memory']}, "
                  f"admission-passes={values['passes_mandatory_prompt_init_admission']}")


if __name__ == "__main__":
    main()
