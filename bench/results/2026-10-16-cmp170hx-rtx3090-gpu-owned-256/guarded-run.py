#!/usr/bin/env python3
"""Run the preserved K37/256 setup with a tokenized prompt and bounded safety polling."""
from __future__ import annotations

import hashlib
import os
import subprocess
import time
from pathlib import Path
from shlex import join


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
ENGINE = Path("/tmp/strata-gpu-owned-cuda/strata")
PACK = ROOT / "build" / "pack" / "ud-q4-k-xl"
MODEL = Path("/home/nikolay/projects/Strata-data/models/UD-Q4_K_XL/UD-Q4_K_XL/"
             "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf")
MTP = Path("/home/nikolay/projects/Strata-data/mtp/rt")
PROMPT_IDS = OUT / "prompt.ids"
ENGINE_LOG = OUT / "engine.txt"
GIB = 1 << 30
HOST_START_FLOOR = int(5.5 * GIB)
HOST_STOP_FLOOR = 5 * GIB
GPU_STOP_FLOOR_MIB = 1024
TIME_LIMIT_SECONDS = 1200


def available_host_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is absent from /proc/meminfo")


def gpu_memory() -> list[tuple[int, int, int]]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free,memory.used",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=5,
    )
    rows = []
    for line in result.stdout.splitlines():
        if line.strip():
            index, free, used = (int(value.strip()) for value in line.split(","))
            rows.append((index, free, used))
    if len(rows) < 2:
        raise RuntimeError(f"expected two visible GPUs, got {rows!r}")
    return rows


def main() -> int:
    if not ENGINE.is_file() or not MODEL.is_file() or not MTP.is_dir() or not PACK.is_dir():
        raise SystemExit("engine, model, MTP, or pack path is missing")
    prompt_bytes = PROMPT_IDS.read_bytes()
    ids = [int(token) for token in prompt_bytes.decode("ascii").strip().split(",") if token]
    if len(ids) != 256:
        raise SystemExit(f"expected 256 token IDs, found {len(ids)}")
    if available_host_bytes() < HOST_START_FLOOR:
        raise SystemExit("insufficient host headroom to start safely (need 5.5 GiB)")
    compute_apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
         "--format=csv,noheader"],
        check=True, capture_output=True, text=True, timeout=5,
    ).stdout.strip()
    if compute_apps:
        raise SystemExit(f"another compute process is using a GPU; refusing to start:\n{compute_apps}")

    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    for key in tuple(env):
        if key.startswith("STRATA_"):
            env.pop(key)
    env.update(
        STRATA_UNBUFFERED_LOAD="1",
        STRATA_RESIDENT_HEADROOM_GIB="4",
        STRATA_PIN_RESERVE_GIB="4",
        STRATA_TRACE="1",
        STRATA_SPLIT_RING="96",
        STRATA_STAGE_PIN="0",
        STRATA_FILL_AHEAD="0",
        STRATA_IO_PREFETCH="0",
        STRATA_REFILL_BLOCKING="1",
    )
    args = [
        str(ENGINE), "--pack", str(PACK), "--native", str(MODEL),
        "--gpu", "1,0", "--layer-split", "37", "--trim-stage-weights",
        "--max-context", "262144", "--kv", "int8",
        "--mtp", str(MTP), "--spec", "2", "--prefill", "256",
        "--no-prefill-borrow", "--mmap-experts", "--gpu-owned-experts",
        "--resident-budget-gib", "0.001", "--expert-profile",
        str(ROOT / "data" / "expert-profile.bin"),
        "--ple-io", "direct", "--ple-row-cache", "1024",
        "--vram-reserve-mib", "1024", "--vram-reserve-later-mib", "1024",
        "--pool-workers", "1", "--serve", "--prompt-cache", "0",
        "--conversation-cache-min-free-mib", "4096",
    ]
    print("COMMAND", join(args), flush=True)
    print("ENV_OVERRIDES", {key: env[key] for key in sorted(env) if key.startswith("STRATA_")},
          "CUDA_VISIBLE_DEVICES=unset", flush=True)
    print("PROMPT_IDS_SHA256", hashlib.sha256(prompt_bytes).hexdigest(),
          "tokens", len(ids), flush=True)
    print("GPU_INITIAL", gpu_memory(), flush=True)
    started = time.monotonic()
    minimum_host = available_host_bytes()
    minimum_gpu = {index: free for index, free, _ in gpu_memory()}
    reason = "completed"
    next_gpu_sample = 0.0
    with ENGINE_LOG.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            args, env=env, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, text=True,
        )
        try:
            process.stdin.write("GEN 4 " + ",".join(map(str, ids)) + "\nQUIT\n")
            process.stdin.flush()
            process.stdin.close()
        except BrokenPipeError:
            pass
        while process.poll() is None:
            elapsed = time.monotonic() - started
            host_free = available_host_bytes()
            minimum_host = min(minimum_host, host_free)
            if host_free < HOST_STOP_FLOOR:
                reason = "5 GiB host reaction threshold (4 GiB authorized floor)"
            elif elapsed > TIME_LIMIT_SECONDS:
                reason = f"{TIME_LIMIT_SECONDS}-second time limit"
            if elapsed >= next_gpu_sample:
                rows = gpu_memory()
                for index, free, _ in rows:
                    minimum_gpu[index] = min(minimum_gpu[index], free)
                print(f"SAMPLE {elapsed:.1f}s host_available={host_free} gpu(index,free_MiB,used_MiB)={rows}",
                      flush=True)
                if min(free for _, free, _ in rows) < GPU_STOP_FLOOR_MIB:
                    reason = "GPU free memory below 1024 MiB"
                next_gpu_sample = elapsed + 2.0
            if reason != "completed":
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                break
            time.sleep(0.1)
        return_code = process.wait()

    elapsed = time.monotonic() - started
    print("RESULT", return_code, reason, "minimum_host_available_bytes", minimum_host,
          "minimum_gpu_free_MiB", minimum_gpu, "elapsed_seconds", round(elapsed, 1), flush=True)
    if return_code != 0 or reason != "completed":
        return 1
    log_text = ENGINE_LOG.read_text(encoding="utf-8", errors="replace")
    required_logs = ("READY 262144 stop", "DONE ", "CUDA0 owns layers", "CUDA1 owns layers",
                     "reusable host staging buffer")
    missing = [item for item in required_logs if item not in log_text]
    if missing:
        print("ACCEPTANCE_LOG_MISSING", missing, flush=True)
        return 1
    if "copied cache complement" in log_text or "resident RAM mode:" in log_text:
        print("UNEXPECTED_CPU_COMPLEMENT_LOG", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
