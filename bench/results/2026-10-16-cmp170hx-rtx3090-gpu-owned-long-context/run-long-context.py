#!/usr/bin/env python3
"""Run guarded, full-prefill long-context requests on the preserved CUDA binary."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PROMPTS = OUT / "prompts"
ENGINE = Path("/tmp/strata-gpu-owned-cuda/strata")
PACK = Path(
    "/home/nikolay/projects/copilot-worktrees/170hx-3090/"
    "sibnick-upgraded-succotash/build/pack/ud-q4-k-xl"
)
MODEL = Path(
    "/home/nikolay/projects/Strata-data/models/UD-Q4_K_XL/UD-Q4_K_XL/"
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
)
MTP = Path("/home/nikolay/projects/Strata-data/mtp/rt")
PROFILE = Path(
    "/home/nikolay/projects/copilot-worktrees/170hx-3090/"
    "sibnick-upgraded-succotash/data/expert-profile.bin"
)
CONTEXT = 262144
GENERATION_CAP = 192
GIB = 1 << 30
HOST_START_FLOOR = 6 * GIB
HOST_REACTION_THRESHOLD = 5 * GIB
GPU_FREE_THRESHOLD_MIB = 1024
MONITOR_INTERVAL_SECONDS = 1.0
STARTUP_TIMEOUT_SECONDS = 1200
INITIAL_PREFILL_RATE = 131.6
ENV_OVERRIDES = {
    "STRATA_UNBUFFERED_LOAD": "1",
    "STRATA_RESIDENT_HEADROOM_GIB": "4",
    "STRATA_PIN_RESERVE_GIB": "4",
    "STRATA_TRACE": "1",
    "STRATA_SPLIT_RING": "96",
    "STRATA_STAGE_PIN": "0",
    "STRATA_FILL_AHEAD": "0",
    "STRATA_IO_PREFETCH": "0",
    "STRATA_REFILL_BLOCKING": "1",
}
MAIN_STAGES = ("8k", "32k", "128k", "near-max", "near-max-repeat")
LABELS = ("ORIGIN", "CENTER", "FINAL")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_meminfo() -> dict[str, int]:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in ("MemAvailable", "SwapTotal", "SwapFree"):
            values[key] = int(value.split()[0]) * 1024
    missing = {"MemAvailable", "SwapTotal", "SwapFree"} - values.keys()
    if missing:
        raise RuntimeError(f"/proc/meminfo is missing {sorted(missing)}")
    return values


def process_state(pid: int) -> dict[str, int]:
    result: dict[str, int] = {}
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith(("VmRSS:", "VmHWM:", "VmSwap:")):
            name, value = line.split(":", 1)
            result[name] = int(value.split()[0]) * 1024
    io = {}
    for line in Path(f"/proc/{pid}/io").read_text().splitlines():
        name, value = line.split(":", 1)
        io[name] = int(value.strip())
    for name in ("rchar", "syscr", "read_bytes", "write_bytes", "cancelled_write_bytes"):
        result[f"io_{name}"] = io.get(name, 0)
    return result


def gpu_state() -> list[dict[str, object]]:
    query = (
        "index,name,utilization.gpu,memory.used,memory.free,"
        "temperature.gpu,power.draw,clocks.current.graphics,clocks.current.sm"
    )
    completed = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=8,
    )
    rows = []
    for line in completed.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 9:
            raise RuntimeError(f"unexpected nvidia-smi GPU row: {line!r}")
        rows.append({
            "index": int(values[0]),
            "name": values[1],
            "utilization_gpu_percent": int(values[2]),
            "memory_used_mib": int(values[3]),
            "memory_free_mib": int(values[4]),
            "temperature_c": int(values[5]),
            "power_w": float(values[6]),
            "graphics_clock_mhz": int(values[7]),
            "sm_clock_mhz": int(values[8]),
        })
    if len(rows) != 2:
        raise RuntimeError(f"expected exactly two visible GPUs, got {rows!r}")
    return rows


def compute_processes() -> str:
    completed = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
         "--format=csv,noheader"],
        check=True, capture_output=True, text=True, timeout=8,
    )
    return completed.stdout.strip()


def prior_artifact_identity() -> dict:
    baseline = ROOT / "bench/results/2026-10-16-cmp170hx-rtx3090-k37/artifacts.json"
    baseline_sha = sha256(baseline)
    expected_sha = "de3b5214d60af2709d19baf8deed4f1c89e35006e26023882794437011d2ddd6"
    if baseline_sha != expected_sha:
        raise RuntimeError(f"preserved K37 artifact manifest hash changed: {baseline_sha}")
    manifest = json.loads(baseline.read_text(encoding="utf-8"))
    current = []
    for artifact in manifest["artifacts"]:
        recorded = artifact["path"]
        pack_prefix = "build/pack/ud-q4-k-xl/"
        if recorded.startswith(pack_prefix):
            path = PACK / recorded.removeprefix(pack_prefix)
        elif recorded == "data/expert-profile.bin":
            path = PROFILE
        elif recorded.startswith("data/"):
            path = ROOT / recorded
        elif Path(recorded).is_absolute():
            path = Path(recorded)
        else:
            continue
        stat = path.stat()
        if stat.st_size != artifact["bytes"]:
            raise RuntimeError(
                f"artifact size mismatch for {path}: {stat.st_size}, expected {artifact['bytes']}"
            )
        fresh_sha = sha256(path) if stat.st_size <= 16 * 1024 * 1024 else None
        if fresh_sha is not None and fresh_sha != artifact["sha256"]:
            raise RuntimeError(f"artifact hash mismatch for {path}: {fresh_sha}")
        current.append({
            "manifest_path": recorded,
            "resolved_path": str(path),
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "recorded_sha256": artifact["sha256"],
            "fresh_sha256": fresh_sha,
        })
    return {
        "manifest_path": str(baseline),
        "manifest_sha256": baseline_sha,
        "gguf_revision": manifest["gguf_revision"],
        "hash_method": manifest["hash_method"],
        "artifacts": current,
        "large_artifact_hashing": (
            "not freshly repeated for files over 16 MiB; size/mtime recorded and original "
            "manifest SHA-256 retained"
        ),
    }


def load_bundle(name: str) -> tuple[dict, list[int]]:
    metadata = json.loads((PROMPTS / f"{name}.json").read_text(encoding="utf-8"))
    id_text = (PROMPTS / f"{name}.ids").read_text(encoding="ascii").strip()
    ids = [int(token) for token in id_text.split(",") if token]
    if len(ids) != metadata["actual_input_tokens"]:
        raise ValueError(f"{name}: expected {metadata['actual_input_tokens']} IDs, found {len(ids)}")
    if len(ids) + GENERATION_CAP > CONTEXT:
        raise ValueError(f"{name}: input plus generation cap exceeds the model context")
    if hashlib.sha256((id_text + "\n").encode("ascii")).hexdigest() != metadata["prompt_ids_sha256"]:
        raise ValueError(f"{name}: prompt ID hash does not match its manifest")
    return metadata, ids


def command_for(mtp_enabled: bool) -> list[str]:
    args = [
        str(ENGINE), "--pack", str(PACK), "--native", str(MODEL),
        "--gpu", "1,0", "--layer-split", "37", "--trim-stage-weights",
        "--max-context", "262144", "--kv", "int8",
    ]
    if mtp_enabled:
        args.extend(("--mtp", str(MTP)))
    args.extend((
        "--spec", "2", "--prefill", "256", "--no-prefill-borrow",
        "--mmap-experts", "--resident-budget-gib", "0.001",
        "--expert-profile", str(PROFILE), "--ple-io", "direct",
        "--ple-row-cache", "1024", "--vram-reserve-mib", "1024",
        "--vram-reserve-later-mib", "1024", "--pool-workers", "1",
        "--serve", "--prompt-cache", "0",
        "--conversation-cache-min-free-mib", "4096",
    ))
    args.append("--gpu-owned-experts")
    return args


def safe_stop(process: subprocess.Popen, reason: str) -> int:
    if process.poll() is None:
        process.terminate()
        try:
            return process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
    return process.wait()


def start_pcie_logger(path: Path):
    try:
        proc = subprocess.Popen(
            ["nvidia-smi", "dmon", "--select", "t", "--delay", "1"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
    except OSError as exc:
        return None, None, f"nvidia-smi dmon unavailable: {exc}"

    def copy_output():
        with path.open("w", encoding="utf-8", buffering=1) as log:
            for line in proc.stdout:
                log.write(f"{time.time():.3f}\t{line.rstrip()}\n")

    thread = threading.Thread(target=copy_output, name="pcie-telemetry", daemon=True)
    thread.start()
    return proc, thread, None


def pcie_summary(path: Path, started: float, stopped: float) -> dict:
    samples: dict[str, dict[str, list[float]]] = {}
    if not path.exists():
        return {"available": False, "reason": "nvidia-smi dmon log was not created"}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        timestamp, separator, sample = line.partition("\t")
        if not separator:
            continue
        try:
            when = float(timestamp)
        except ValueError:
            continue
        if not started <= when <= stopped:
            continue
        columns = sample.split()
        if len(columns) != 3 or not columns[0].isdigit():
            continue
        index, rx, tx = columns
        if rx.lower() in ("n/a", "-") or tx.lower() in ("n/a", "-"):
            continue
        values = samples.setdefault(index, {"rx_mib_s": [], "tx_mib_s": []})
        values["rx_mib_s"].append(float(rx))
        values["tx_mib_s"].append(float(tx))
    return {
        "available": bool(samples),
        "units": "MB/s as printed by nvidia-smi dmon",
        "scope": "whole-device throughput; other graphics processes may contribute",
        "per_gpu": {
            index: {
                metric: {
                    "sample_count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                }
                for metric, values in metrics.items() if values
            }
            for index, metrics in samples.items()
        },
        "reason": None if samples else "no parseable PCIe throughput samples in this interval",
    }


class Session:
    def __init__(self, mode: str, mtp_enabled: bool):
        self.mode = mode
        self.mtp_enabled = mtp_enabled
        self.args = command_for(mtp_enabled)
        self.env = os.environ.copy()
        self.env.pop("CUDA_VISIBLE_DEVICES", None)
        for key in tuple(self.env):
            if key.startswith("STRATA_"):
                self.env.pop(key)
        self.env.update(ENV_OVERRIDES)
        self.events_path = OUT / f"{mode}-progress.log"
        self.engine_path = OUT / f"{mode}-engine.log"
        self.monitor_path = OUT / f"{mode}-monitor.jsonl"
        self.pcie_path = OUT / f"{mode}-pcie.log"
        self.results_path = OUT / f"{mode}-results.json"
        self.reader_queue: queue.Queue[str | None] = queue.Queue()
        self.results: dict = {
            "mode": mode,
            "mtp_enabled": mtp_enabled,
            "started_utc": utc_now(),
            "status": "starting",
            "arguments": self.args,
            "environment_policy": {
                "CUDA_VISIBLE_DEVICES": "unset",
                "STRATA_*": "clear inherited values, then set exactly as in the prior comparison manifest",
                "STRATA_overrides": ENV_OVERRIDES,
                "other_inherited_environment": "retained; not serialized",
            },
            "guards": {
                "host_start_floor_bytes": HOST_START_FLOOR,
                "host_reaction_threshold_bytes": HOST_REACTION_THRESHOLD,
                "gpu_free_reaction_threshold_mib": GPU_FREE_THRESHOLD_MIB,
                "monitor_interval_seconds": MONITOR_INTERVAL_SECONDS,
                "startup_timeout_seconds": STARTUP_TIMEOUT_SECONDS,
                "request_timeout_policy": (
                    "max(1800 s, 4 * stage_token_count / preceding_measured_prefill_tokens_per_s + 300 s); "
                    "no disk-progress timeout"
                ),
                "memory_monitoring_is_sampled_not_a_hard_limit": True,
            },
            "requests": [],
        }
        self.last_sample = 0.0
        self.samples: list[dict] = []
        self.global_min_host = None
        self.global_min_gpu = {}
        self.global_max_rss = 0
        self.global_max_swap = 0
        self.global_max_process_swap = 0
        self.started_epoch = time.time()
        self.pcie_process = None
        self.pcie_thread = None
        self.engine_process = None
        self.reader_thread = None
        self.failure_reason = None

    def write_event(self, message: str) -> None:
        line = f"{utc_now()} {message}"
        print(line, flush=True)
        with self.events_path.open("a", encoding="utf-8") as log:
            log.write(line + "\n")
            log.flush()

    def save_results(self) -> None:
        temp = self.results_path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(self.results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp.replace(self.results_path)

    def sample(self, stage: str, request_start: float | None = None) -> dict:
        now = time.monotonic()
        mem = read_meminfo()
        state = {}
        if self.engine_process and self.engine_process.poll() is None:
            try:
                state = process_state(self.engine_process.pid)
            except FileNotFoundError:
                state = {}
        gpus = gpu_state()
        active_apps = compute_processes()
        self_pid = self.engine_process.pid if self.engine_process else None
        other_apps = []
        for line in active_apps.splitlines():
            values = [value.strip() for value in line.split(",")]
            if values and values[0].isdigit() and int(values[0]) != self_pid:
                other_apps.append(line)
        sample = {
            "utc": utc_now(),
            "elapsed_seconds": round(now - self.started_monotonic, 3),
            "stage": stage,
            "request_elapsed_seconds": None if request_start is None else round(now - request_start, 3),
            "host_available_bytes": mem["MemAvailable"],
            "swap_total_bytes": mem["SwapTotal"],
            "swap_free_bytes": mem["SwapFree"],
            "process": state,
            "gpus": gpus,
            "other_compute_processes": other_apps,
        }
        self.samples.append(sample)
        self.global_min_host = mem["MemAvailable"] if self.global_min_host is None \
            else min(self.global_min_host, mem["MemAvailable"])
        self.global_max_swap = max(self.global_max_swap, mem["SwapTotal"] - mem["SwapFree"])
        if "VmRSS" in state:
            self.global_max_rss = max(self.global_max_rss, state["VmRSS"])
            self.global_max_process_swap = max(self.global_max_process_swap, state.get("VmSwap", 0))
        for gpu in gpus:
            index = str(gpu["index"])
            free = int(gpu["memory_free_mib"])
            self.global_min_gpu[index] = min(self.global_min_gpu.get(index, free), free)
        with self.monitor_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(sample, sort_keys=True) + "\n")
            log.flush()

        if mem["MemAvailable"] < HOST_REACTION_THRESHOLD:
            return sample
        if min(int(gpu["memory_free_mib"]) for gpu in gpus) < GPU_FREE_THRESHOLD_MIB:
            return sample
        return sample

    def check_pressure(self, sample: dict) -> str | None:
        if sample["host_available_bytes"] < HOST_REACTION_THRESHOLD:
            return (
                f"host MemAvailable {sample['host_available_bytes']} below "
                f"{HOST_REACTION_THRESHOLD}-byte reaction threshold"
            )
        low = [
            (gpu["index"], gpu["memory_free_mib"])
            for gpu in sample["gpus"] if gpu["memory_free_mib"] < GPU_FREE_THRESHOLD_MIB
        ]
        if low:
            return f"GPU free memory below {GPU_FREE_THRESHOLD_MIB} MiB: {low}"
        other_pids = sample.get("other_compute_processes", [])
        if other_pids:
            return f"another GPU compute process appeared: {other_pids}"
        return None

    def drain_reader(self, engine_log) -> str | None:
        try:
            line = self.reader_queue.get_nowait()
        except queue.Empty:
            return None
        if line is None:
            return ""
        engine_log.write(line)
        engine_log.flush()
        return line.rstrip("\n")

    def next_line(self, engine_log, stage: str, request_start: float | None, deadline: float,
                  last_progress_print: float) -> tuple[str | None, str | None, float]:
        now = time.monotonic()
        if now >= self.last_sample + MONITOR_INTERVAL_SECONDS:
            sample = self.sample(stage, request_start)
            self.last_sample = now
            if reason := self.check_pressure(sample):
                return None, reason, now
        if now >= deadline:
            return None, f"bounded timeout expired after {now - (request_start or self.started_monotonic):.1f}s", now
        if now - last_progress_print >= 15:
            sample = self.samples[-1] if self.samples else {}
            host = sample.get("host_available_bytes", "sampling")
            free = [
                f'{gpu["index"]}:{gpu["memory_free_mib"]}'
                for gpu in sample.get("gpus", [])
            ]
            stage_elapsed = "" if request_start is None else f" request={now - request_start:.0f}s"
            self.write_event(f"PROGRESS stage={stage}{stage_elapsed} host_available={host} gpu_free_MiB={free}")
            last_progress_print = now
        line = self.drain_reader(engine_log)
        if line is not None:
            return line, None, last_progress_print
        try:
            line = self.reader_queue.get(timeout=0.25)
        except queue.Empty:
            return None, None, last_progress_print
        if line is None:
            return "", None, last_progress_print
        engine_log.write(line)
        engine_log.flush()
        return line.rstrip("\n"), None, last_progress_print

    def start(self) -> bool:
        existing = [
            path for path in (
                self.events_path, self.engine_path, self.monitor_path, self.pcie_path, self.results_path
            ) if path.exists()
        ]
        if existing:
            raise FileExistsError(
                f"refusing to overwrite prior run evidence: {', '.join(map(str, existing))}"
            )
        if not ENGINE.is_file() or not PACK.is_dir() or not MODEL.is_file() or not PROFILE.is_file():
            raise FileNotFoundError("engine, pack, native model, or expert profile is missing")
        if self.mtp_enabled and not MTP.is_dir():
            raise FileNotFoundError(f"MTP directory is missing: {MTP}")
        if read_meminfo()["MemAvailable"] < HOST_START_FLOOR:
            raise RuntimeError(
                f"refusing to start: host MemAvailable is below {HOST_START_FLOOR} bytes"
            )
        if active := compute_processes():
            raise RuntimeError(f"refusing to start while another GPU compute process is active:\n{active}")
        identity = {
            "run_mode": self.mode,
            "source_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True,
            ).stdout.strip(),
            "engine_sha256": sha256(ENGINE),
            "cmake_cache_sha256": sha256(Path("/tmp/strata-gpu-owned-cuda/CMakeCache.txt")),
            "build_directory": "/tmp/strata-gpu-owned-cuda",
            "build_type": "Release",
            "cuda_compiler": "/usr/local/cuda-13.4/bin/nvcc",
            "cuda_compiler_version": "13.4.92",
            "cuda_architectures": "sm_80;sm_86",
            "candidate_build_source_commit": "0327ba279ba50a34894f6aac16e5fe3c21106611",
            "artifact_identity": prior_artifact_identity(),
            "gpu_initial": gpu_state(),
            "driver_version": subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                check=True, capture_output=True, text=True, timeout=8,
            ).stdout.strip().splitlines(),
            "compute_processes_before_start": "none",
        }
        self.results["identity"] = identity
        self.results["started_utc"] = utc_now()
        with self.events_path.open("w", encoding="utf-8") as log:
            log.write(f"{utc_now()} COMMAND {' '.join(self.args)}\n")
            log.write(f"{utc_now()} IDENTITY {json.dumps(identity, sort_keys=True)}\n")
            log.write(f"{utc_now()} ENV_OVERRIDES {json.dumps(ENV_OVERRIDES, sort_keys=True)}\n")
            log.flush()
        self.started_monotonic = time.monotonic()
        self.pcie_process, self.pcie_thread, pcie_error = start_pcie_logger(self.pcie_path)
        self.results["pcie_start_error"] = pcie_error
        self.engine_process = subprocess.Popen(
            self.args, cwd=ROOT, env=self.env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        self.spawn_snapshot = process_state(self.engine_process.pid)

        def read_stdout():
            for line in self.engine_process.stdout:
                self.reader_queue.put(line)
            self.reader_queue.put(None)

        self.reader_thread = threading.Thread(target=read_stdout, name="engine-output", daemon=True)
        self.reader_thread.start()
        self.engine_started_epoch = time.time()
        self.engine_started_monotonic = time.monotonic()
        with self.engine_path.open("w", encoding="utf-8", buffering=1) as engine_log:
            startup_deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
            last_progress = time.monotonic()
            ready = False
            startup_exit = False
            while not ready:
                line, reason, last_progress = self.next_line(
                    engine_log, "startup", None, startup_deadline, last_progress
                )
                if reason:
                    self.failure_reason = reason
                    break
                if line == "":
                    startup_exit = True
                    break
                if line and line.startswith("READY "):
                    if line != f"READY {CONTEXT} stop":
                        self.failure_reason = f"unexpected engine context line: {line}"
                        break
                    ready = True
                    self.ready_epoch = time.time()
                    self.ready_monotonic = time.monotonic()
                    self.write_event(
                        f"READY {line} startup_seconds={self.ready_monotonic - self.engine_started_monotonic:.1f}"
                    )
        if not ready:
            self.failure_reason = self.failure_reason or "engine exited before READY"
            code = safe_stop(self.engine_process, self.failure_reason)
            self.results["startup"] = {
                "return_code": code,
                "failure_reason": self.failure_reason,
            }
            return False
        self.start_snapshot = process_state(self.engine_process.pid)
        self.results["startup"] = {
            "seconds": round(self.ready_monotonic - self.engine_started_monotonic, 3),
            "process_io_at_ready": self.start_snapshot,
        }
        self.results["startup"]["process_io_delta_to_ready"] = {
            key: self.start_snapshot[key] - self.spawn_snapshot[key]
            for key in self.spawn_snapshot if key.startswith("io_")
        }
        self.results["startup"]["process_io_at_spawn"] = self.spawn_snapshot
        self.results["status"] = "running"
        self.save_results()
        return True

    def run_request(self, name: str, metadata: dict, ids: list[int],
                    preceding_rate: float) -> tuple[dict | None, float]:
        if len(ids) + GENERATION_CAP > CONTEXT:
            raise ValueError(f"{name}: request plus output cap exceeds model context")
        rate = max(preceding_rate, 1.0)
        request_timeout = max(1800.0, (len(ids) / rate) * 4.0 + 300.0)
        request_start = time.monotonic()
        request_start_epoch = time.time()
        before_io = process_state(self.engine_process.pid)
        engine_log_start = self.engine_path.stat().st_size
        request = {
            "name": name,
            "started_utc": utc_now(),
            "input_tokens": len(ids),
            "generation_cap": GENERATION_CAP,
            "context_cells_reserved": len(ids) + GENERATION_CAP,
            "prompt_text_sha256": metadata["prompt_text_sha256"],
            "prompt_ids_sha256": metadata["prompt_ids_sha256"],
            "expected_answers": metadata["expected"],
            "needle_offsets_token": metadata["needle_offsets_token"],
            "preceding_stage_prefill_tokens_per_second": preceding_rate,
            "timeout_seconds": round(request_timeout, 3),
            "prefix_reuse_expected": 0,
            "process_io_before": before_io,
        }
        self.write_event(
            f"REQUEST_START stage={name} tokens={len(ids)} generation_cap={GENERATION_CAP} "
            f"deadline={request_timeout:.0f}s prompt_ids_sha256={metadata['prompt_ids_sha256']}"
        )
        tokens = []
        last_progress = time.monotonic()
        deadline = request_start + request_timeout
        write_failure: list[BaseException] = []
        request_line = f"GEN {GENERATION_CAP} " + ",".join(map(str, ids)) + "\n"

        def write_request():
            try:
                self.engine_process.stdin.write(request_line)
                self.engine_process.stdin.flush()
            except BaseException as exc:
                write_failure.append(exc)

        writer = threading.Thread(target=write_request, name=f"request-writer-{name}", daemon=True)
        writer.start()
        done_fields = None
        first_token_elapsed = None
        done_observed_monotonic = None
        done_observed_epoch = None
        with self.engine_path.open("a", encoding="utf-8", buffering=1) as engine_log:
            while done_fields is None:
                line, reason, last_progress = self.next_line(
                    engine_log, name, request_start, deadline, last_progress
                )
                if reason:
                    self.failure_reason = reason
                    break
                if line == "":
                    self.failure_reason = "engine closed stdout before DONE"
                    break
                if line is None:
                    if write_failure:
                        self.failure_reason = f"could not write request: {write_failure[0]}"
                        break
                    if self.engine_process.poll() is not None:
                        self.failure_reason = f"engine exited with code {self.engine_process.returncode} before DONE"
                        break
                    continue
                if line.startswith("T "):
                    try:
                        tokens.append(int(line.split()[1]))
                    except (IndexError, ValueError) as exc:
                        self.failure_reason = f"invalid generated token line {line!r}: {exc}"
                        break
                    if first_token_elapsed is None:
                        first_token_elapsed = time.monotonic() - request_start
                elif line.startswith("ERR "):
                    self.failure_reason = f"engine rejected the request: {line}"
                    break
                elif line.startswith("DONE "):
                    fields = line.split()
                    if len(fields) < 16:
                        self.failure_reason = f"short DONE line: {line}"
                        break
                    done_fields = fields
                    done_observed_monotonic = time.monotonic()
                    done_observed_epoch = time.time()
        writer.join(timeout=15)
        if writer.is_alive():
            self.failure_reason = "request input writer remained blocked after engine response"
        if done_fields is None or self.failure_reason:
            code = safe_stop(self.engine_process, self.failure_reason or "request did not complete")
            request["status"] = "failed"
            request["failure_reason"] = self.failure_reason
            request["return_code_after_stop"] = code
            request["generated_tokens_so_far"] = tokens
            request["finished_utc"] = utc_now()
            request["process_io_after"] = process_state(self.engine_process.pid) \
                if Path(f"/proc/{self.engine_process.pid}").exists() else None
            self.results["requests"].append(request)
            self.save_results()
            self.write_event(f"REQUEST_FAILED stage={name} reason={self.failure_reason}")
            return None, preceding_rate

        # The engine emits its storage/expert counters just after DONE. Drain them before
        # the next request so process-cumulative and request-delta reads stay attributable.
        ancillary_deadline = time.monotonic() + 3.0
        ancillary_idle_since = None
        with self.engine_path.open("a", encoding="utf-8", buffering=1) as engine_log:
            while time.monotonic() < ancillary_deadline:
                try:
                    line = self.reader_queue.get(timeout=0.25)
                except queue.Empty:
                    if ancillary_idle_since is None:
                        ancillary_idle_since = time.monotonic()
                    if time.monotonic() - ancillary_idle_since >= 0.5:
                        break
                    continue
                if line is None:
                    break
                engine_log.write(line)
                engine_log.flush()
                ancillary_idle_since = None

        finished_epoch = done_observed_epoch
        finished = done_observed_monotonic
        produced = int(done_fields[1])
        input_count = int(done_fields[2])
        prompt_ms = float(done_fields[3])
        decode_ms = float(done_fields[4])
        finish = done_fields[5]
        accepted = int(done_fields[6])
        offered = int(done_fields[7])
        reused = int(done_fields[8])
        fresh = int(done_fields[14])
        if produced != len(tokens):
            self.failure_reason = f"DONE reports {produced} tokens but captured {len(tokens)}"
        elif input_count != len(ids):
            self.failure_reason = f"DONE reports {input_count} prompt tokens but sent {len(ids)}"
        elif reused != 0 or fresh != len(ids):
            self.failure_reason = f"prompt reuse was not zero (reused={reused}, fresh={fresh})"
        elif finish != "length" and finish != "stop":
            self.failure_reason = f"unexpected generation finish reason {finish}"
        elif produced >= GENERATION_CAP:
            self.failure_reason = "generation hit its cap; answer completion is not established"
        else:
            self.failure_reason = None

        decoded = ""
        if self.failure_reason is None:
            sys.path.insert(0, str(ROOT / "tools"))
            from strata_tokenizer import Tokenizer
            vocab = json.loads((PACK / "tokenizer" / "vocab.json").read_text(encoding="utf-8"))
            ordered = [None] * len(vocab)
            for token, token_id in vocab.items():
                ordered[token_id] = token
            merges = (PACK / "tokenizer" / "merges.txt").read_text(encoding="utf-8").splitlines()
            token_types = json.loads(
                (PACK / "tokenizer" / "token_type.json").read_text(encoding="utf-8")
            )
            config = json.loads((PACK / "tokenizer" / "tokenizer.json").read_text(encoding="utf-8"))
            tokenizer = Tokenizer(
                ordered, merges, token_types, config["pre"], config["special_ids"]
            )
            eos_ids = {248044, 248046}
            decoded = tokenizer.decode(tokens[:-1] if tokens and tokens[-1] in eos_ids else tokens)
            expected = metadata["expected"]
            for label in LABELS:
                pattern = (
                    r"(?im)^\s*(?:[-*]\s*)?" + re.escape(label)
                    + r"\s*[:=]\s*`?" + re.escape(expected[label]) + r"`?\s*$"
                )
                if not re.search(pattern, decoded):
                    self.failure_reason = (
                        f"retrieval check failed for {label}={expected[label]}; "
                        "answer did not contain the exact labeled code"
                    )
                    break
        request["finished_utc"] = utc_now()
        request["elapsed_seconds"] = round(finished - request_start, 3)
        request["ttft_seconds"] = (
            None if first_token_elapsed is None else round(first_token_elapsed, 3)
        )
        request["status"] = "passed" if self.failure_reason is None else "failed"
        request["done_line"] = " ".join(done_fields)
        request["generated_token_count"] = produced
        request["generated_token_ids"] = tokens
        request["generated_token_ids_sha256"] = hashlib.sha256(
            (",".join(map(str, tokens)) + "\n").encode("ascii")
        ).hexdigest()
        request["answer"] = decoded
        request["prompt_ms"] = prompt_ms
        request["prefill_tokens_per_second"] = len(ids) / (prompt_ms / 1000.0) if prompt_ms else None
        request["decode_ms"] = decode_ms
        request["decode_tokens_per_second"] = produced / (decode_ms / 1000.0) if decode_ms else None
        request["mtp_drafts_accepted"] = accepted
        request["mtp_drafts_offered"] = offered
        request["mtp_acceptance_rate"] = accepted / offered if offered else None
        request["prompt_tokens_reused"] = reused
        request["prompt_tokens_read"] = fresh
        request["process_io_after"] = process_state(self.engine_process.pid)
        request["process_io_delta"] = {
            key: request["process_io_after"][key] - before_io[key]
            for key in before_io if key.startswith("io_")
        }
        request_tail = self.engine_path.read_bytes()[engine_log_start:].decode("utf-8", errors="replace")
        request_lines = request_tail.splitlines()
        request_lines = [
            line for line in request_lines
            if "expert tiers: GPU " in line or "file tier I/O this request:" in line
        ]
        request["engine_reported_request_expert_io"] = request_lines[-2:]
        request["pcie"] = pcie_summary(self.pcie_path, request_start_epoch, finished_epoch)
        request["minimum_host_available_bytes"] = min(
            (
                sample["host_available_bytes"] for sample in self.samples
                if sample["request_elapsed_seconds"] is not None
                and sample["stage"] == name
            ),
            default=None,
        )
        request["minimum_gpu_free_mib"] = {
            str(index): min(
                sample["gpus"][index]["memory_free_mib"]
                for sample in self.samples
                if sample["request_elapsed_seconds"] is not None
                and sample["stage"] == name
            )
            for index in range(2)
        }
        request["maximum_process_rss_bytes"] = max(
            (
                sample["process"].get("VmRSS", 0) for sample in self.samples
                if sample["request_elapsed_seconds"] is not None
                and sample["stage"] == name
            ),
            default=None,
        )
        self.results["requests"].append(request)
        self.save_results()
        output_name = f"mtp-off-{name}" if self.mode == "mtp-off" else name
        token_path = OUT / "generated" / f"{output_name}.ids"
        answer_path = OUT / "generated" / f"{output_name}.txt"
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(",".join(map(str, tokens)) + "\n", encoding="ascii")
        answer_path.write_text(decoded, encoding="utf-8")
        prefill_rate = request["prefill_tokens_per_second"] or preceding_rate
        self.write_event(
            f"REQUEST_{request['status'].upper()} stage={name} tokens={len(ids)} "
            f"prefill={prefill_rate:.2f} tok/s decode={request['decode_tokens_per_second']} tok/s "
            f"generated={produced} reuse={reused} MTP={accepted}/{offered}"
        )
        if self.failure_reason:
            request["failure_reason"] = self.failure_reason
            self.save_results()
            return None, prefill_rate
        return request, prefill_rate

    def finish(self) -> None:
        if self.engine_process is not None and self.engine_process.poll() is None:
            try:
                self.engine_process.stdin.close()
            except BrokenPipeError:
                pass
            try:
                self.engine_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                safe_stop(self.engine_process, "process did not exit after EOF")
        if self.pcie_process is not None and self.pcie_process.poll() is None:
            self.pcie_process.terminate()
            try:
                self.pcie_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.pcie_process.kill()
                self.pcie_process.wait()
        if self.pcie_thread is not None:
            self.pcie_thread.join(timeout=5)
        if self.engine_process is not None:
            self.results["engine_return_code"] = self.engine_process.returncode
        if self.engine_path.exists():
            engine_lines = self.engine_path.read_text(encoding="utf-8", errors="replace").splitlines()
            self.results.setdefault("startup", {})["expert_ownership_and_residency_lines"] = [
                line for line in engine_lines
                if "gpu-owned-experts:" in line
                and any(word in line for word in ("owns layers", "resident", "staging buffer"))
            ]
        ended = time.time()
        self.results["finished_utc"] = utc_now()
        self.results["elapsed_seconds"] = round(ended - self.started_epoch, 3)
        self.results["minimum_host_available_bytes"] = self.global_min_host
        self.results["minimum_gpu_free_mib"] = self.global_min_gpu
        self.results["maximum_engine_process_rss_bytes"] = self.global_max_rss
        self.results["maximum_engine_process_swap_bytes"] = self.global_max_process_swap
        self.results["maximum_system_swap_used_bytes"] = self.global_max_swap
        self.results["pcie_whole_run"] = pcie_summary(self.pcie_path, self.started_epoch, ended)
        self.results["status"] = (
            "failed" if self.failure_reason or self.results.get("engine_return_code", 0) != 0
            else "passed"
        )
        if self.failure_reason:
            self.results["failure_reason"] = self.failure_reason
        self.save_results()
        self.write_event(
            f"RUN_{self.results['status'].upper()} elapsed={self.results['elapsed_seconds']}s "
            f"engine_exit={self.results.get('engine_return_code')} "
            f"min_host={self.global_min_host} min_gpu={self.global_min_gpu}"
        )


def run(mode: str) -> int:
    mtp_enabled = mode != "mtp-off"
    names = MAIN_STAGES if mode == "main" else ("8k",)
    session = Session(mode, mtp_enabled)
    first, _ = load_bundle(names[0])
    if any(first["expected"].get(label) is None for label in LABELS):
        raise ValueError("prompt manifest does not contain all expected answers")
    if mode == "mtp-off":
        on_ids_path = OUT / "generated" / "8k.ids"
        if not on_ids_path.exists():
            raise RuntimeError("run the MTP-on stages first; their 8k token IDs are required")
    preceding_rate = INITIAL_PREFILL_RATE
    success = False
    try:
        if not session.start():
            return 1
        for name in names:
            metadata, ids = load_bundle(name)
            request, preceding_rate = session.run_request(name, metadata, ids, preceding_rate)
            if request is None:
                break
            if mode == "mtp-off":
                on_ids_path = OUT / "generated" / "8k.ids"
                off_ids = [
                    int(x) for x in on_ids_path.read_text(encoding="ascii").strip().split(",") if x
                ]
                if request["generated_token_ids"] != off_ids:
                    session.failure_reason = "MTP-on and MTP-off output token IDs differ"
                    session.results["mtp_output_parity"] = False
                    session.save_results()
                    break
                session.results["mtp_output_parity"] = True
                session.results["mtp_on_token_ids_sha256"] = sha256(on_ids_path)
                session.save_results()
        success = session.failure_reason is None and all(
            request["status"] == "passed" for request in session.results["requests"]
        )
    except Exception as exc:
        session.failure_reason = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        session.finish()
    return 0 if success and session.results["status"] == "passed" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("main", "mtp-off"), required=True)
    args = parser.parse_args()
    return run(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
