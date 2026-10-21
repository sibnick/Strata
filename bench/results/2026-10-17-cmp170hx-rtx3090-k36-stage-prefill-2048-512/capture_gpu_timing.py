#!/usr/bin/env python3
"""One guarded K36 2048/512 prompt-timing capture; requires the API to be stopped."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import queue
import random
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
CAPTURE = OUT / "gpu-utilization-capture-ple-phases"
GIB = 1 << 30
MIB = 1 << 20
TOKEN_COUNT = 1977
OUTPUT_CAP = 32
SHORT_RUN = OUT.parent / "2026-10-17-cmp170hx-rtx3090-k37-mmq-batched-gather" / "run.py"


def utc():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def mem_available():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is missing")


def gpu_rows():
    raw = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,memory.free",
        "--format=csv,noheader,nounits",
    ], text=True)
    rows = {}
    for line in raw.splitlines():
        index, name, free_mib = (part.strip() for part in line.split(",", 2))
        rows[int(index)] = {"name": name, "free_bytes": int(free_mib) * MIB}
    return rows


def compute_processes():
    return subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader",
    ], text=True).strip()


def load_synthetic_ids(tokenizer_dir):
    sys.path.insert(0, str(ROOT / "tools"))
    vocab = json.loads((tokenizer_dir / "vocab.json").read_text(encoding="utf-8"))
    types = json.loads((tokenizer_dir / "token_type.json").read_text())
    ordinary = [token_id for token_id, kind in enumerate(types) if kind == 1]
    if len(vocab) != len(types) or len(ordinary) < TOKEN_COUNT:
        raise RuntimeError("tokenizer inventory is incomplete")
    seed = int(time.time())
    rng = random.Random(seed)
    ids = [rng.choice(ordinary) for _ in range(TOKEN_COUNT)]
    digest = hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()
    return seed, ids, digest


def load_historical_fixture():
    spec = importlib.util.spec_from_file_location("k36_timing_short_fixture", SHORT_RUN)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load historical short-fixture runner: {SHORT_RUN}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    meta, ids, reference = module.bundle(["large"])[0]
    if len(ids) != TOKEN_COUNT or meta["generation_cap"] != OUTPUT_CAP:
        raise RuntimeError("historical short fixture is not exactly 1977 tokens / cap32")
    return meta, ids, reference


def process_cpu(pid):
    task_dir = Path(f"/proc/{pid}/task")
    ticks = runnable = threads = 0
    for stat in task_dir.glob("*/stat"):
        try:
            raw = stat.read_text()
            fields = raw[raw.rfind(")") + 2:].split()
            threads += 1
            runnable += fields[0] == "R"
            ticks += int(fields[11]) + int(fields[12])
        except (FileNotFoundError, ProcessLookupError, IndexError, ValueError):
            continue
    return {"threads": threads, "runnable_threads": runnable, "cpu_ticks": ticks}


def gpu_sample():
    raw = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=index,memory.free,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ], text=True)
    result = {}
    for line in raw.splitlines():
        index, free_mib, util, power = (part.strip() for part in line.split(",", 3))
        result[int(index)] = {
            "free_bytes": int(free_mib) * MIB,
            "utilization_percent": int(util),
            "power_w": float(power),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-fixture", choices=("historical-large", "synthetic"),
                        default="historical-large")
    parser.add_argument("--capture-dir", type=Path, default=CAPTURE,
                        help="new evidence directory (relative paths are under this result directory)")
    cli_args = parser.parse_args()
    capture_dir = cli_args.capture_dir
    if not capture_dir.is_absolute():
        capture_dir = OUT / capture_dir
    if capture_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing evidence: {capture_dir}")
    config = json.loads((OUT / "api/strata-api.json").read_text())
    sizing = json.loads((OUT / "offline.json").read_text())["sizing"]["36"]
    engine_args = config["args"]
    for flag, expected in (
        ("--gpu", "1,0"), ("--layer-split", "36"), ("--max-context", "262144"),
        ("--kv", "int8"), ("--prefill-stage-chunks", "2048,512"),
        ("--vram-reserve-mib", "0"), ("--vram-reserve-later-mib", "0"),
    ):
        try:
            actual = engine_args[engine_args.index(flag) + 1]
        except (ValueError, IndexError) as exc:
            raise RuntimeError(f"API config lacks {flag}") from exc
        if actual != expected:
            raise RuntimeError(f"API config has {flag}={actual}, expected {expected}")
    for flag in ("--serve", "--gpu-owned-experts", "--no-prefill-borrow"):
        if flag not in engine_args:
            raise RuntimeError(f"API config lacks {flag}")
    if mem_available() < 6 * GIB:
        raise RuntimeError("host MemAvailable is below the 6 GiB startup threshold")
    other_compute = compute_processes()
    if other_compute:
        raise RuntimeError(f"GPU compute process exists; refusing to interfere: {other_compute}")
    free = gpu_rows()
    admission = {}
    for stage, physical, expected in (
        ("0", 1, "NVIDIA CMP 170HX"), ("1", 0, "NVIDIA GeForce RTX 3090"),
    ):
        if free[physical]["name"] != expected:
            raise RuntimeError(f"physical GPU {physical} is {free[physical]['name']}, expected {expected}")
        need = sizing[stage]["required_with_mandatory_prompt_init_headroom"]
        admission[stage] = {
            "physical_gpu": physical, "free_bytes": free[physical]["free_bytes"],
            "required_including_mandatory_512_mib_headroom_bytes": need,
            "margin_bytes": free[physical]["free_bytes"] - need,
        }
        if free[physical]["free_bytes"] < need:
            raise RuntimeError(f"physical GPU {physical} fails mandatory prompt-init admission")

    fixture_reference = None
    seed = None
    if cli_args.prompt_fixture == "historical-large":
        prompt_meta, ids, fixture_reference = load_historical_fixture()
        prompt_hash = prompt_meta["prompt_ids_sha256"]
    else:
        seed, ids, prompt_hash = load_synthetic_ids(Path(config["tokenizer"]))
    capture_dir.mkdir()
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("STRATA_") and k not in ("CUDA_VISIBLE_DEVICES", "LD_PRELOAD")}
    env.update({str(k): str(v) for k, v in config["env"].items()})
    env["STRATA_PREFILL_TIMING"] = "1"
    command = [config["exe"], *config["args"], "--stats"]
    dmon_file = (capture_dir / "dmon-samples.txt").open("w", buffering=1)
    dmon = subprocess.Popen(
        ["nvidia-smi", "dmon", "-i", "0,1", "-d", "1", "-s", "putcm", "-o", "T"],
        stdout=dmon_file, stderr=subprocess.STDOUT, text=True,
    )
    engine = None
    reader = None
    lines = queue.Queue()
    samples = []
    output_lines = []
    generated_ids = []
    started = time.monotonic()
    request_started = None
    result = {
        "started_utc": utc(),
        "runtime": {"gpu": "1,0", "layer_split": 36, "stage_chunks": [2048, 512],
                    "max_context": 262144, "kv": "int8", "speculative_generation": "unchanged"},
        "environment": config["env"],
        "instrumentation": {"STRATA_PREFILL_TIMING": "1"},
        "prompt_tokens": TOKEN_COUNT, "generation_cap": OUTPUT_CAP,
        "prompt_fixture": cli_args.prompt_fixture, "prompt_seed": seed,
        "prompt_ids_sha256": prompt_hash,
        "admission": admission, "initial_mem_available_bytes": mem_available(),
        "initial_gpu_memory": free, "engine_pid": None, "request_started_utc": None,
        "request_finished_utc": None, "phase_lines": [], "request_result": None,
    }

    def append_line(line):
        if (line.startswith("strata prefill timing:") or line.startswith("strata prefill timing (peer GPU")
                or line.startswith("DONE ") or line.startswith("ERR ")
                or line.startswith("REUSED ")
                or line.startswith("strata PLE ")
                or line.startswith("ple io: ")
                or "file tier I/O this request:" in line):
            output_lines.append(line)

    def read_output():
        for line in engine.stdout:
            lines.put(line.rstrip("\n"))
        lines.put(None)

    try:
        engine = subprocess.Popen(
            command, cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        result["engine_pid"] = engine.pid
        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        ready = False
        request_done = False
        done_line = None
        last_gpu_sample = 0.0
        startup_deadline = time.monotonic() + 1200
        request_deadline = None
        while not request_done:
            deadline = request_deadline if ready else startup_deadline
            if time.monotonic() >= deadline:
                break
            now = time.monotonic()
            if mem_available() < 5 * GIB:
                raise RuntimeError("host MemAvailable fell below the 5 GiB stop threshold")
            if now - last_gpu_sample >= 1.0:
                snap = gpu_sample()
                stats = process_cpu(engine.pid)
                sample = {
                    "utc": utc(), "elapsed_seconds": now - started,
                    "request_elapsed_seconds": None if request_started is None else now - request_started,
                    "mem_available_bytes": mem_available(), "gpu": snap, "engine": stats,
                }
                samples.append(sample)
                last_gpu_sample = now
            try:
                line = lines.get(timeout=0.2)
            except queue.Empty:
                continue
            if line is None:
                raise RuntimeError("engine exited before request completion")
            append_line(line)
            if ready and line.startswith("T "):
                generated_ids.append(int(line.split()[1]))
            if line.startswith("ERR "):
                raise RuntimeError(line)
            if not ready:
                if line == "READY 262144 stop":
                    ready = True
                    request_started = time.monotonic()
                    request_deadline = request_started + 180
                    result["request_started_utc"] = utc()
                    engine.stdin.write(f"GEN {OUTPUT_CAP} " + ",".join(map(str, ids)) + "\n")
                    engine.stdin.flush()
                continue
            if line.startswith("DONE "):
                done_line = line
                request_done = True
                result["request_finished_utc"] = utc()
                break
        if not ready:
            raise TimeoutError("engine did not become ready within 20 minutes")
        if not request_done:
            raise TimeoutError("bounded request did not finish before the capture deadline")
        done_fields = done_line.split()
        if len(done_fields) < 16 or int(done_fields[1]) != len(generated_ids) or \
                int(done_fields[1]) > OUTPUT_CAP or int(done_fields[2]) != TOKEN_COUNT:
            raise RuntimeError(f"engine completion did not match the bounded request: {done_line}")
        if int(done_fields[8]) != 0 or int(done_fields[14]) != TOKEN_COUNT:
            raise RuntimeError(f"prompt was not fully fresh: {done_line}")
        output_match = None
        if fixture_reference is not None:
            output_match = generated_ids == fixture_reference["generated_ids"]
            if not output_match:
                raise RuntimeError("historical prompt output IDs differ from the exact recall reference")
        engine.stdin.close()
        engine.wait(timeout=30)
        reader.join(timeout=5)
        result["request_result"] = done_line
        result["generated_count"] = len(generated_ids)
        result["output_ids_match_historical"] = output_match
    finally:
        if engine is not None and engine.poll() is None:
            engine.terminate()
            try:
                engine.wait(timeout=10)
            except subprocess.TimeoutExpired:
                engine.kill()
                engine.wait()
        if dmon.poll() is None:
            dmon.terminate()
            try:
                dmon.wait(timeout=5)
            except subprocess.TimeoutExpired:
                dmon.kill()
                dmon.wait()
        dmon_file.close()
        result["finished_utc"] = utc()
        result["engine_return_code"] = None if engine is None else engine.poll()
        result["dmon_return_code"] = dmon.returncode
        result["sample_count"] = len(samples)
        if samples:
            result["min_host_available_bytes"] = min(s["mem_available_bytes"] for s in samples)
            result["min_gpu_free_bytes"] = {
                str(gpu): min(s["gpu"][gpu]["free_bytes"] for s in samples)
                for gpu in (0, 1)
            }
        result["phase_lines"] = output_lines
        result["other_compute_after"] = compute_processes()
        (capture_dir / "monitor.jsonl").write_text(
            "".join(json.dumps(sample, sort_keys=True) + "\n" for sample in samples))
        (capture_dir / "capture.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if result["request_result"] is None or result["engine_return_code"] != 0:
        raise RuntimeError("capture did not complete cleanly")
    if "strata prefill timing:" not in "\n".join(output_lines):
        raise RuntimeError("STRATA_PREFILL_TIMING output is missing")
    timing_text = "\n".join(output_lines)
    for marker in ("strata PLE prep timing:", "strata PLE gather timing:", "strata PLE upload timing:"):
        if marker not in timing_text:
            raise RuntimeError(f"PLE phase breakdown is missing: {marker}")
    if "ple io: " not in timing_text:
        raise RuntimeError("PLE row-cache and direct-read summary is missing")
    if result["other_compute_after"]:
        raise RuntimeError(f"GPU compute process remained after capture: {result['other_compute_after']}")
    dmon_text = (capture_dir / "dmon-samples.txt").read_text()
    if "rxpci" not in dmon_text or "pclk" not in dmon_text or len(dmon_text.splitlines()) < 4:
        raise RuntimeError("timestamped dmon GPU/PCIe/clock samples are missing")
    if result["dmon_return_code"] not in (-15, 0):
        raise RuntimeError(f"dmon exited unexpectedly: {result['dmon_return_code']}")
    artifacts = ("capture.json", "monitor.jsonl", "dmon-samples.txt")
    (capture_dir / "SHA256SUMS").write_text("".join(
        f"{hashlib.sha256((capture_dir / name).read_bytes()).hexdigest()}  {name}\n"
        for name in artifacts
    ))
    print(json.dumps({
        "request": result["request_result"], "phase_lines": result["phase_lines"],
        "admission": admission, "sample_count": len(samples),
        "min_host_available_bytes": min(s["mem_available_bytes"] for s in samples),
        "min_gpu_free_bytes": {
            str(gpu): min(s["gpu"][gpu]["free_bytes"] for s in samples)
            for gpu in (0, 1) if samples
        },
        "other_compute_after": result["other_compute_after"],
    }, indent=2))


if __name__ == "__main__":
    main()
