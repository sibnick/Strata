#!/usr/bin/env python3
"""Profile one CMP Q4_K MMQ kernel from a bounded historical K36 request."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
from urllib.error import URLError
from urllib.request import urlopen

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
CONFIG = OUT / "api/strata-api.json"
CAPTURE = OUT / "gpu-utilization-capture-ncu-cmp-sm80-512mib"
PRIOR_CAPTURE = OUT / "gpu-utilization-capture-cuda-timeline-cmp-sm80/capture.json"
PROXY = OUT / "nsys_engine_proxy.py"
SHORT_RUN = OUT.parent / "2026-10-17-cmp170hx-rtx3090-k37-mmq-batched-gather/run.py"
NCU = Path("/opt/nvidia/nsight-compute/2026.3.1/ncu")
TOKEN_COUNT = 1977
OUTPUT_CAP = 32
GIB = 1 << 30
MIB = 1 << 20
HOST_LIMIT_BYTES = 512 * MIB
METRICS = (
    "gpu__time_duration.avg,"
    "sm__throughput.avg.pct_of_peak_sustained_elapsed,"
    "sm__warps_active.avg.pct_of_peak_sustained_active,"
    "dram__throughput.avg.pct_of_peak_sustained_elapsed"
)
KERNEL_REGEX = r"^void mul_mat_q<\(ggml_type\)12, \(int\)128, \(bool\)0>.*"


def utc_now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def mem_available():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is missing")


def host_memory_snapshot():
    fields = {
        "MemAvailable": "mem_available_bytes",
        "MemFree": "mem_free_bytes",
        "Cached": "cached_bytes",
        "Buffers": "buffers_bytes",
        "Slab": "slab_bytes",
        "SReclaimable": "sreclaimable_bytes",
        "PageTables": "page_tables_bytes",
        "Unevictable": "unevictable_bytes",
    }
    result = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in fields:
            result[fields[key]] = int(value.strip().split()[0]) * 1024
    if "mem_available_bytes" not in result:
        raise RuntimeError("MemAvailable is missing")
    return result


def process_tree(root_pid):
    children = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = entry.joinpath("stat").read_text()
            fields = stat[stat.rfind(")") + 2:].split()
            children.setdefault(int(fields[1]), []).append(int(entry.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, IndexError):
            continue
    result, pending = [], [root_pid]
    while pending:
        parent = pending.pop()
        result.append(parent)
        pending.extend(children.get(parent, []))
    return result


def process_memory_snapshot(root_pid):
    snapshot = {}
    for pid in process_tree(root_pid):
        base = Path(f"/proc/{pid}")
        try:
            status = base.joinpath("status").read_text()
            cmdline = base.joinpath("cmdline").read_bytes().replace(b"\0", b" ").decode(
                errors="replace"
            ).strip()
            rss = next(
                int(line.split()[1]) * 1024
                for line in status.splitlines() if line.startswith("VmRSS:")
            )
            pss = None
            for line in base.joinpath("smaps_rollup").read_text().splitlines():
                if line.startswith("Pss:"):
                    pss = int(line.split()[1]) * 1024
                    break
            snapshot[str(pid)] = {
                "cmdline": cmdline[:240], "rss_bytes": rss, "pss_bytes": pss,
            }
        except (FileNotFoundError, ProcessLookupError, PermissionError, StopIteration, ValueError):
            continue
    return snapshot


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gpu_rows():
    raw = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.free",
        "--format=csv,noheader,nounits",
    ], text=True, timeout=8)
    rows = {}
    for line in raw.splitlines():
        index, uuid, name, free_mib = (part.strip() for part in line.split(",", 3))
        rows[int(index)] = {
            "uuid": uuid, "name": name, "free_bytes": int(free_mib) * MIB,
        }
    return rows


def compute_processes():
    return subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader",
    ], text=True, timeout=8).strip()


def api_is_listening():
    try:
        with urlopen("http://127.0.0.1:8080/health", timeout=2):
            return True
    except (URLError, TimeoutError):
        with socket.socket() as probe:
            return probe.connect_ex(("127.0.0.1", 8080)) == 0


def load_fixture():
    spec = importlib.util.spec_from_file_location("k36_ncu_short_fixture", SHORT_RUN)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load historical fixture helper: {SHORT_RUN}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metadata, ids, reference = module.bundle(["large"])[0]
    if len(ids) != TOKEN_COUNT or metadata["generation_cap"] != OUTPUT_CAP:
        raise RuntimeError("historical fixture is not exactly 1977 tokens / cap32")
    return metadata, ids, reference


def parse_done(line, reference):
    fields = line.split()
    if len(fields) < 16:
        raise RuntimeError(f"malformed engine completion: {line}")
    result = {
        "generated_token_count": int(fields[1]),
        "prompt_token_count": int(fields[2]),
        "prefill_ms": float(fields[3]),
        "decode_ms": float(fields[4]),
        "mtp_accepted_offered": [int(fields[6]), int(fields[7])],
        "fresh_prompt_tokens": int(fields[14]),
    }
    if result["generated_token_count"] != len(reference["generated_ids"]):
        raise RuntimeError("generated token count differs from the historical fixture")
    if result["prompt_token_count"] != TOKEN_COUNT or result["fresh_prompt_tokens"] != TOKEN_COUNT:
        raise RuntimeError("prompt was not exactly 1977 fresh tokens")
    return result


def stop_owned_process(process, timeout=10):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def main():
    if CAPTURE.exists():
        raise RuntimeError(f"refusing to overwrite evidence directory: {CAPTURE}")
    if not NCU.is_file() or not os.access(NCU, os.X_OK):
        raise RuntimeError(f"Nsight Compute is unavailable: {NCU}")
    if api_is_listening():
        raise RuntimeError("port 8080 is active; this capture never stops or restarts the API")
    if mem_available() < HOST_LIMIT_BYTES:
        raise RuntimeError("host MemAvailable is below the 512 MiB startup threshold")
    active = compute_processes()
    if active:
        raise RuntimeError(f"GPU compute process exists; refusing to overlap: {active}")

    config = json.loads(CONFIG.read_text())
    engine = Path(config["exe"])
    if not engine.is_file():
        raise RuntimeError(f"tested engine binary is missing: {engine}")
    prior_identity = json.loads(PRIOR_CAPTURE.read_text())["capture_identity"]
    engine_sha256 = sha256_file(engine)
    kquant_sha256 = sha256_file(engine.parent / "libstrata_kq.so")
    if engine_sha256 != prior_identity["engine_sha256"]:
        raise RuntimeError("engine binary differs from the prior tested K36 capture")
    if kquant_sha256 != prior_identity["kquant_library_sha256"]:
        raise RuntimeError("K-quant library differs from the prior tested K36 capture")
    engine_args = config["args"]
    for flag, expected in (
        ("--gpu", "1,0"), ("--layer-split", "36"), ("--max-context", "262144"),
        ("--kv", "int8"), ("--prefill-stage-chunks", "2048,512"),
        ("--vram-reserve-mib", "0"), ("--vram-reserve-later-mib", "0"),
    ):
        try:
            actual = engine_args[engine_args.index(flag) + 1]
        except (ValueError, IndexError) as exc:
            raise RuntimeError(f"runtime config lacks {flag}") from exc
        if actual != expected:
            raise RuntimeError(f"runtime config has {flag}={actual}, expected {expected}")
    for flag in ("--serve", "--gpu-owned-experts", "--no-prefill-borrow"):
        if flag not in engine_args:
            raise RuntimeError(f"runtime config lacks {flag}")

    initial_gpu = gpu_rows()
    expected_gpus = {
        0: ("GPU-01f295eb-2ee0-838a-ba5b-db97f075a5f7", "NVIDIA GeForce RTX 3090"),
        1: ("GPU-79d96de3-4783-ba96-d95c-fe5e0364528e", "NVIDIA CMP 170HX"),
    }
    for physical, expected in expected_gpus.items():
        if (initial_gpu[physical]["uuid"], initial_gpu[physical]["name"]) != expected:
            raise RuntimeError(f"physical GPU identity mismatch at index {physical}")

    sizing = json.loads((OUT / "offline.json").read_text())["sizing"]["36"]
    admission = {}
    for stage, physical in (("0", 1), ("1", 0)):
        required = sizing[stage]["required_with_mandatory_prompt_init_headroom"]
        available = initial_gpu[physical]["free_bytes"]
        admission[stage] = {
            "physical_gpu": physical,
            "free_bytes": available,
            "required_with_mandatory_prompt_init_headroom_bytes": required,
            "margin_bytes": available - required,
        }
        if available < required:
            raise RuntimeError(f"physical GPU {physical} fails mandatory prompt-init admission")

    prompt, expected_ids, reference = load_fixture()
    if mem_available() < HOST_LIMIT_BYTES or compute_processes():
        raise RuntimeError("fresh host/CUDA admission changed while preparing the fixture")
    CAPTURE.mkdir()
    config_path = CAPTURE / "engine-config.json"
    ids_path = CAPTURE / "generated-ids.tmp"
    report_path = CAPTURE / "cmp-q4k-mmq"
    log_path = CAPTURE / "ncu.log"
    monitor_path = CAPTURE / "monitor.jsonl"
    sanitized_config = {"exe": config["exe"], "args": engine_args, "cwd": config["cwd"]}
    config_path.write_text(json.dumps(sanitized_config, sort_keys=True) + "\n")
    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith("STRATA_") and key not in ("CUDA_VISIBLE_DEVICES", "LD_PRELOAD")
    }
    env.update({str(key): str(value) for key, value in config["env"].items()})
    env["CUDA_VISIBLE_DEVICES"] = "1,0"
    command = [
        str(NCU), "--target-processes", "all", "--devices", "0",
        "--kernel-name-base", "demangled", "--kernel-name", f"regex:{KERNEL_REGEX}",
        "--launch-count", "1", "--metrics", METRICS,
        "--replay-mode", "kernel", "--cache-control", "none",
        "--clock-control", "base", "--export", str(report_path),
        sys.executable, str(PROXY), "--config", str(config_path),
        "--token-ids-output", str(ids_path),
    ]
    result = {
        "status": "starting",
        "started_utc": utc_now(),
        "runtime": {
            "gpu_order": "1,0", "layer_split": 36, "stage_chunks": [2048, 512],
            "max_context": 262144, "kv": "int8", "mtp_spec": "unchanged",
            "ple_and_cache_settings": "unchanged",
            "engine_sha256": engine_sha256,
            "kquant_library_sha256": kquant_sha256,
            "source_revision": prior_identity["source_revision_premerge"],
        },
        "fixture": {
            "name": "historical-large", "prompt_tokens": TOKEN_COUNT,
            "generation_cap": OUTPUT_CAP, "prompt_ids_sha256": prompt["prompt_ids_sha256"],
            "fresh_input": None, "output_ids_match_historical": None,
            "expected_generated_ids_count": len(reference["generated_ids"]),
        },
        "profiler": {
            "name": "Nsight Compute", "version": subprocess.check_output(
                [str(NCU), "--version"], text=True, stderr=subprocess.STDOUT, timeout=10
            ).strip(),
            "device_filter": "CUDA device 0 (CMP 170HX, physical GPU 1)",
            "kernel_filter": KERNEL_REGEX, "launch_count": 1,
            "metrics": METRICS.split(","), "replay_mode": "kernel",
            "cache_control": "none", "clock_control": "base",
            "report_file": report_path.name + ".ncu-rep",
        },
        "memory_policy": {
            "engine_reserves_mib": [0, 0], "external_gpu_free_abort": False,
            "host_start_mib": 512, "host_stop_mib": 512,
            "mandatory_cuda_admission": True, "mandatory_prompt_init_headroom_mib": 512,
            "cache_flush": False,
        },
        "admission": admission,
        "initial_host_memory": host_memory_snapshot(),
        "initial_gpu_memory": initial_gpu,
        "timestamps": {},
        "request_result": None,
        "ncu_command": command,
        "compute_processes_after": None,
    }
    process = None
    engine_pid = None
    monitor = []
    messages = queue.Queue()
    start = time.monotonic()
    last_sample = 0.0
    last_guard_check = 0.0
    phase = "model-startup"

    def read_output():
        assert process is not None and process.stdout is not None
        for raw in process.stdout:
            messages.put(raw.rstrip("\n"))
        messages.put(None)

    def sample():
        nonlocal last_sample, last_guard_check
        now = time.monotonic()
        if now - last_guard_check >= 0.1:
            last_guard_check = now
            if mem_available() < HOST_LIMIT_BYTES:
                raise RuntimeError("host MemAvailable fell below the 512 MiB stop threshold")
        if now - last_sample < 1:
            return
        last_sample = now
        gpu = gpu_rows()
        row = {
            "utc": utc_now(), "elapsed_seconds": now - start, "phase": phase,
            "host_memory": host_memory_snapshot(),
            "gpu_free_bytes": {str(i): gpu[i]["free_bytes"] for i in (0, 1)},
            "engine_pid": engine_pid,
            "process_memory": process_memory_snapshot(os.getpid()),
        }
        monitor.append(row)

    try:
        with log_path.open("w", buffering=1) as log:
            process = subprocess.Popen(
                command, cwd=ROOT, env=env, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            )
            reader = threading.Thread(target=read_output, daemon=True)
            reader.start()
            ready = False
            done = False
            deadline = time.monotonic() + 1200
            request_started = False
            while not done:
                if time.monotonic() >= deadline:
                    raise TimeoutError("bounded Nsight Compute request exceeded 20 minutes")
                sample()
                try:
                    line = messages.get(timeout=0.2)
                except queue.Empty:
                    if process.poll() is not None:
                        raise RuntimeError(f"Nsight Compute exited with status {process.returncode}")
                    continue
                if line is None:
                    raise RuntimeError("Nsight Compute proxy exited before request completion")
                log.write(line + "\n")
                if line.startswith("ERR "):
                    raise RuntimeError(line)
                if line.startswith("ENGINE_PID "):
                    engine_pid = int(line.split()[1])
                elif line == "READY 262144 stop":
                    if ready or engine_pid is None:
                        raise RuntimeError("unexpected READY or missing engine PID")
                    ready = True
                    phase = "profiled-short-request"
                    result["timestamps"]["ready_utc"] = utc_now()
                    process.stdin.write(f"GEN {OUTPUT_CAP} " + ",".join(map(str, expected_ids)) + "\n")
                    process.stdin.flush()
                    request_started = True
                    result["timestamps"]["request_sent_utc"] = utc_now()
                elif line.startswith("DONE "):
                    if not request_started:
                        raise RuntimeError("DONE arrived before the bounded request")
                    result["request_result"] = parse_done(line, reference)
                    result["fixture"]["fresh_input"] = True
                    result["timestamps"]["done_utc"] = utc_now()
                    process.stdin.write("QUIT\n")
                    process.stdin.flush()
                    done = True
            phase = "profiler-finalization"
            try:
                process.wait(timeout=300)
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError("Nsight Compute report finalization exceeded five minutes") from exc
            reader.join(timeout=10)
            if process.returncode != 0:
                raise RuntimeError(f"Nsight Compute returned {process.returncode}")
            if not ids_path.is_file():
                raise RuntimeError("engine proxy did not return generated token IDs")
            generated_ids = json.loads(ids_path.read_text())
            ids_path.unlink()
            result["fixture"]["generated_ids_count"] = len(generated_ids)
            result["fixture"]["output_ids_match_historical"] = generated_ids == reference["generated_ids"]
            if not result["fixture"]["output_ids_match_historical"]:
                raise RuntimeError("generated IDs differ from historical fixture")
            if not Path(str(report_path) + ".ncu-rep").is_file():
                raise RuntimeError("Nsight Compute report is missing")
            result["status"] = "passed"
    except BaseException as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if process is not None and process.poll() is None:
            stop_owned_process(process)
        if engine_pid is not None and Path(f"/proc/{engine_pid}").exists():
            try:
                command_line = Path(f"/proc/{engine_pid}/cmdline").read_bytes().split(b"\0")
                if str(engine).encode() in command_line:
                    os.kill(engine_pid, signal.SIGTERM)
                    for _ in range(100):
                        if not Path(f"/proc/{engine_pid}").exists():
                            break
                        time.sleep(0.1)
                    if Path(f"/proc/{engine_pid}").exists():
                        os.kill(engine_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        result["finished_utc"] = utc_now()
        result["min_host_memavailable_bytes"] = min(
            (row["host_memory"]["mem_available_bytes"] for row in monitor),
            default=host_memory_snapshot()["mem_available_bytes"],
        )
        result["min_gpu_free_bytes"] = {
            str(i): min((row["gpu_free_bytes"][str(i)] for row in monitor),
                        default=gpu_rows()[i]["free_bytes"])
            for i in (0, 1)
        }
        result["compute_processes_after"] = compute_processes()
        result["api_listener_after"] = api_is_listening()
        result["ncu_return_code"] = None if process is None else process.poll()
        result["monitor_samples"] = len(monitor)
        try:
            config_path.unlink()
        except FileNotFoundError:
            pass
        monitor_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in monitor)
        )
        (CAPTURE / "capture.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        if log_path.exists():
            artifacts = (
                "capture.json", "monitor.jsonl", "ncu.log",
                report_path.name + ".ncu-rep",
            )
            present = [name for name in artifacts if (CAPTURE / name).is_file()]
            (CAPTURE / "SHA256SUMS").write_text("".join(
                f"{sha256_file(CAPTURE / name)}  {name}\n" for name in present
            ))
    if result["status"] != "passed":
        raise RuntimeError("Nsight Compute capture did not pass")
    if result["compute_processes_after"] or result["api_listener_after"]:
        raise RuntimeError("compute process remains or API listener unexpectedly started")
    print(json.dumps({
        "status": result["status"], "request_result": result["request_result"],
        "output_ids_match_historical": result["fixture"]["output_ids_match_historical"],
        "profiler": result["profiler"], "admission": admission,
        "min_host_memavailable_bytes": result["min_host_memavailable_bytes"],
        "min_gpu_free_bytes": result["min_gpu_free_bytes"],
        "api_listener_after": result["api_listener_after"],
        "compute_processes_after": result["compute_processes_after"],
    }, indent=2))


if __name__ == "__main__":
    main()
