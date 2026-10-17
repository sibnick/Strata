#!/usr/bin/env python3
"""Capture one historical short K36 request after model startup using Nsight Systems."""
import argparse
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
from urllib.request import urlopen
from datetime import datetime, timezone
from urllib.error import URLError

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
CONFIG = OUT / "api/strata-api.json"
CAPTURE = OUT / "gpu-utilization-capture-cuda-timeline"
PROXY = OUT / "nsys_engine_proxy.py"
SHORT_RUN = OUT.parent / "2026-10-17-cmp170hx-rtx3090-k37-mmq-batched-gather/run.py"
TOKEN_COUNT = 1977
OUTPUT_CAP = 32
GIB = 1 << 30
MIB = 1 << 20


def utc():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def mem_available():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is missing")


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


def api_health():
    try:
        with urlopen("http://127.0.0.1:8080/health", timeout=2) as response:
            return json.loads(response.read())
    except (URLError, TimeoutError, json.JSONDecodeError):
        return None


def restore_api():
    health = api_health()
    if health is not None:
        if health.get("status") == "ok" and health.get("loaded") is True:
            return {"already_running": True, "health": health}
        raise RuntimeError("loopback API endpoint is present but not healthy")
    if compute_processes():
        raise RuntimeError("refusing to restart API while GPU compute is active")
    if mem_available() < 6 * GIB:
        raise RuntimeError("host MemAvailable is below the 6 GiB API restart threshold")
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", 8080)) == 0:
            raise RuntimeError("port 8080 is occupied; refusing to disrupt its listener")

    api_process = subprocess.Popen(
        ["bash", str(OUT / "api/start-api.sh")], cwd=ROOT,
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    for _ in range(1200):
        health = api_health()
        if health is not None and health.get("status") == "ok" and health.get("loaded") is True:
            return {"supervisor_pid": api_process.pid, "health": health}
        if api_process.poll() is not None:
            raise RuntimeError(f"API start process exited with status {api_process.returncode}")
        if mem_available() < 5 * GIB:
            raise RuntimeError("host MemAvailable fell below 5 GiB while restoring API")
        time.sleep(1)
    raise RuntimeError("API did not become loaded/healthy within the startup deadline")


def stop_owned_nsys_agent(session):
    stopped = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            args = entry.joinpath("cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
        except (FileNotFoundError, ProcessLookupError, PermissionError, UnicodeDecodeError):
            continue
        if not args or Path(args[0]).name != "nsys" or "--start-agent" not in args:
            continue
        try:
            marker = args.index("--session-name")
        except ValueError:
            continue
        session_args = []
        for arg in args[marker + 1:]:
            if arg.startswith("--"):
                break
            session_args.append(arg)
        if " ".join(session_args) != session:
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        for _ in range(50):
            if not entry.exists():
                break
            time.sleep(0.1)
        if entry.exists():
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        stopped.append(pid)
    return stopped


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


def load_fixture():
    spec = importlib.util.spec_from_file_location("k36_nsys_short_fixture", SHORT_RUN)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load historical fixture helper: {SHORT_RUN}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metadata, ids, reference = module.bundle(["large"])[0]
    if len(ids) != TOKEN_COUNT or metadata["generation_cap"] != OUTPUT_CAP:
        raise RuntimeError("historical fixture is not exactly 1977 tokens / cap32")
    return metadata, ids, reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, default=CAPTURE)
    cli = parser.parse_args()
    capture_dir = cli.capture_dir if cli.capture_dir.is_absolute() else OUT / cli.capture_dir
    if capture_dir.exists():
        raise RuntimeError(f"refusing to overwrite evidence directory: {capture_dir}")
    if not shutil_which("nsys"):
        raise RuntimeError("nsys is not installed")
    config = json.loads(CONFIG.read_text())
    if api_health() is not None:
        raise RuntimeError("loopback port 8080 already serves an API; refusing to stop or profile it")
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
        raise RuntimeError("host MemAvailable is below the required 6 GiB startup threshold")
    active = compute_processes()
    if active:
        raise RuntimeError(f"GPU compute process exists; refusing to interfere: {active}")
    initial_gpu = gpu_rows()
    expected_gpus = {
        0: ("GPU-01f295eb-2ee0-838a-ba5b-db97f075a5f7", "NVIDIA GeForce RTX 3090"),
        1: ("GPU-79d96de3-4783-ba96-d95c-fe5e0364528e", "NVIDIA CMP 170HX"),
    }
    sizing = json.loads((OUT / "offline.json").read_text())["sizing"]["36"]
    admission = {}
    for stage, physical in ((0, 1), (1, 0)):
        uuid, name = expected_gpus[physical]
        if (initial_gpu[physical]["uuid"], initial_gpu[physical]["name"]) != (uuid, name):
            raise RuntimeError(f"physical GPU identity mismatch at index {physical}")
        required = sizing[str(stage)]["required_with_mandatory_prompt_init_headroom"]
        free = initial_gpu[physical]["free_bytes"]
        admission[str(stage)] = {
            "physical_gpu": physical, "free_bytes": free,
            "required_including_mandatory_512_mib_prompt_init_headroom_bytes": required,
            "margin_bytes": free - required,
        }
        if free < required:
            raise RuntimeError(f"physical GPU {physical} fails mandatory prompt-init admission")
    prompt, expected_ids, reference = load_fixture()
    if mem_available() < 6 * GIB or compute_processes():
        raise RuntimeError("fresh host/CUDA admission changed while preparing the fixture")

    capture_dir.mkdir()
    ids_path = capture_dir / "generated-ids.tmp"
    profile_output = capture_dir / "cuda-timeline"
    session = f"K36 timeline {os.getpid()}"
    sanitized_config = capture_dir / "engine-config.json"
    sanitized_config.write_text(json.dumps({
        "exe": config["exe"], "args": config["args"], "cwd": config["cwd"],
    }))
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("STRATA_") and key not in ("CUDA_VISIBLE_DEVICES", "LD_PRELOAD")}
    env.update({str(key): str(value) for key, value in config["env"].items()})
    profiler_command = [
        "nsys", "profile", "--start-later=true", f"--session-new={session}",
        "--trace=cuda,cublas,osrt", "--sample=process-tree", "--cpuctxsw=process-tree",
        "--cuda-event-trace=false", "--osrt-threshold=10000", "--stats=true",
        "--show-output=true", "--discard-environment=true", f"--output={profile_output}",
        sys.executable, str(PROXY), "--config", str(sanitized_config),
        "--token-ids-output", str(ids_path),
    ]
    stdout_path = capture_dir / "proxy-control.log"
    stderr_path = capture_dir / "nsys.log"
    stdout_file = stdout_path.open("w", buffering=1)
    stderr_file = stderr_path.open("w", buffering=1)
    monitor = []
    messages = queue.Queue()
    profiled = None
    engine_pid = None
    reader = None
    dmon = None
    dmon_file = None
    capture_started = False
    capture_stopped = False
    timeline_start_mono = None
    result = {
        "started_utc": utc(),
        "runtime": {"gpu": "1,0", "layer_split": 36, "stage_chunks": [2048, 512],
                    "max_context": 262144, "kv": "int8", "mtp_spec": "unchanged"},
        "environment": config["env"],
        "profiler": {
            "name": "Nsight Systems", "version": subprocess.check_output(
                ["nsys", "--version"], text=True, stderr=subprocess.STDOUT, timeout=10
            ).strip(),
            "trace": ["cuda", "cublas", "osrt"],
            "cpu_thread_scheduling": "process-tree", "cpu_sampling": "process-tree",
            "collection_starts_after_model_ready": True,
            "collection_stops_after_done": True,
            "cuda_event_trace": False, "osrt_threshold_ns": 10000,
            "stats_enabled": True,
        },
        "request": {
            "fixture": "historical-large", "prompt_tokens": TOKEN_COUNT,
            "generation_cap": OUTPUT_CAP, "prompt_ids_sha256": prompt["prompt_ids_sha256"],
            "input_fresh": None, "output_ids_match_historical": None,
            "expected_output_token_count": len(reference["generated_ids"]),
        },
        "memory_policy": {
            "engine_reserves_mib": [0, 0], "external_gpu_free_abort": False,
            "host_start_bytes": 6 * GIB, "host_stop_bytes": 5 * GIB,
            "mandatory_cuda_admission": True,
            "mandatory_prompt_init_headroom_mib": 512,
            "cache_flush": False,
        },
        "admission": admission, "initial_mem_available_bytes": mem_available(),
        "initial_gpu_memory": initial_gpu,
        "timestamps": {}, "request_result": None,
        "engine_pid": None, "profile_cli_pid": None,
    }

    def read_proxy():
        for line in profiled.stdout:
            messages.put(line.rstrip("\n"))
        messages.put(None)

    def sample():
        snap = gpu_rows()
        return {
            "utc": utc(), "elapsed_seconds": time.monotonic() - start_mono,
            "request_elapsed_seconds": (
                None if request_start_mono is None
                else time.monotonic() - request_start_mono
            ),
            "mem_available_bytes": mem_available(),
            "gpu": {
                str(i): {
                    "free_bytes": snap[i]["free_bytes"],
                    "name": snap[i]["name"],
                } for i in (0, 1)
            },
            "engine_process": process_cpu(engine_pid) if engine_pid is not None else None,
        }

    start_mono = time.monotonic()
    request_start_mono = None
    last_sample = 0.0
    try:
        dmon_file = (capture_dir / "dmon-samples.txt").open("w", buffering=1)
        dmon = subprocess.Popen(
            ["nvidia-smi", "dmon", "-i", "0,1", "-d", "1", "-s", "putcm", "-o", "T"],
            stdout=dmon_file, stderr=subprocess.STDOUT, text=True,
        )
        profiled = subprocess.Popen(
            profiler_command, cwd=ROOT, env=env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=stderr_file, text=True, bufsize=1,
        )
        result["profile_cli_pid"] = profiled.pid
        reader = threading.Thread(target=read_proxy, daemon=True)
        reader.start()
        ready = False
        first_token = False
        done = False
        deadline = time.monotonic() + 1200
        while not done:
            if time.monotonic() >= deadline:
                raise TimeoutError("bounded Nsight request exceeded its deadline")
            now = time.monotonic()
            available = mem_available()
            if available < 5 * GIB:
                raise RuntimeError("host MemAvailable fell below the 5 GiB stop threshold")
            if now - last_sample >= 1.0:
                monitor.append(sample())
                last_sample = now
            try:
                line = messages.get(timeout=0.2)
            except queue.Empty:
                if profiled.poll() is not None:
                    raise RuntimeError(f"Nsight exited early with status {profiled.returncode}")
                continue
            if line is None:
                raise RuntimeError("Nsight proxy exited before request completion")
            if line.startswith("ERR "):
                raise RuntimeError(line)
            if line.startswith("ENGINE_PID "):
                engine_pid = int(line.split()[1])
                result["engine_pid"] = engine_pid
            elif line == "READY 262144 stop":
                if ready:
                    raise RuntimeError("proxy emitted duplicate READY")
                if engine_pid is None:
                    raise RuntimeError("proxy did not report its engine PID")
                ready = True
                start = subprocess.run(
                    ["nsys", "start", f"--session={session}"], check=True,
                    capture_output=True, text=True, timeout=30,
                )
                capture_started = True
                timeline_start_mono = time.monotonic()
                request_start_mono = timeline_start_mono
                result["timestamps"]["nsys_start_command_completed_utc"] = utc()
                result["timestamps"]["gen_sent_utc"] = utc()
                result["timestamps"]["gen_sent_elapsed_ns"] = time.monotonic_ns() - int(timeline_start_mono * 1e9)
                profiled.stdin.write(f"GEN {OUTPUT_CAP} " + ",".join(map(str, expected_ids)) + "\n")
                profiled.stdin.flush()
            elif line == "FIRST_TOKEN":
                if not ready or first_token:
                    raise RuntimeError("unexpected/duplicate first-token marker")
                first_token = True
                result["timestamps"]["first_token_seen_utc"] = utc()
                result["timestamps"]["first_token_elapsed_ns"] = time.monotonic_ns() - int(timeline_start_mono * 1e9)
            elif line.startswith("DONE "):
                if not ready or not first_token:
                    raise RuntimeError("DONE arrived before first token")
                done = True
                result["request_result"] = line
                result["timestamps"]["done_seen_utc"] = utc()
                result["timestamps"]["done_elapsed_ns"] = time.monotonic_ns() - int(timeline_start_mono * 1e9)
                request_start_mono = None
                fields = line.split()
                if len(fields) < 16 or int(fields[1]) != len(reference["generated_ids"]) or \
                        int(fields[2]) != TOKEN_COUNT or int(fields[8]) != 0 or \
                        int(fields[14]) != TOKEN_COUNT:
                    raise RuntimeError(f"request completion/freshness check failed: {line}")
                result["request"]["input_fresh_tokens"] = int(fields[14])
                result["request"]["input_fresh"] = True
                result["request"]["prefill_ms"] = float(fields[3])
                result["request"]["decode_ms"] = float(fields[4])
                result["request"]["mtp_accepted_offered"] = [int(fields[6]), int(fields[7])]
                stop = subprocess.run(
                    ["nsys", "stop", f"--session={session}"], check=True,
                    capture_output=True, text=True, timeout=180,
                )
                capture_stopped = True
                result["profiler"]["stop_output"] = stop.stdout.strip()
                profiled.stdin.write("QUIT\n")
                profiled.stdin.flush()
                break
        try:
            profiled.wait(timeout=120)
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError("Nsight did not exit after the profiled child stopped") from exc
        reader.join(timeout=10)
        if profiled.returncode != 0:
            raise RuntimeError(f"Nsight profile returned {profiled.returncode}")
        if not ids_path.exists():
            raise RuntimeError("proxy did not return output-token IDs")
        generated_ids = json.loads(ids_path.read_text())
        ids_path.unlink()
        result["request"]["generated_count"] = len(generated_ids)
        result["request"]["output_ids_match_historical"] = generated_ids == reference["generated_ids"]
        if not result["request"]["output_ids_match_historical"]:
            raise RuntimeError("generated IDs differ from historical output")
        result["profiler"]["report_file"] = profile_output.name + ".nsys-rep"
        report = Path(str(profile_output) + ".nsys-rep")
        if not report.is_file() or report.stat().st_size <= 0:
            raise RuntimeError(f"Nsight report missing or empty: {report}")
        export = subprocess.run(
            ["nsys", "export", "--type=sqlite", f"--output={capture_dir / 'timeline-export.sqlite'}", str(report)],
            check=True, capture_output=True, text=True, timeout=120,
        )
        result["profiler"]["export"] = export.stdout.strip()
        result["min_host_available_bytes"] = min(s["mem_available_bytes"] for s in monitor)
        result["min_gpu_free_bytes"] = {
            str(i): min(s["gpu"][str(i)]["free_bytes"] for s in monitor) for i in (0, 1)
        }
        result["monitor_sample_count"] = len(monitor)
        result["engine_return_code"] = profiled.returncode
        result["other_compute_after"] = compute_processes()
    except Exception as exc:
        result["capture_error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if profiled is not None and profiled.poll() is None:
            if capture_started and not capture_stopped:
                try:
                    subprocess.run(
                        ["nsys", "stop", f"--session={session}"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30,
                    )
                    capture_stopped = True
                except (OSError, subprocess.SubprocessError) as exc:
                    result["profiler"]["cleanup_stop_error"] = f"{type(exc).__name__}: {exc}"
            try:
                profiled.stdin.write("QUIT\n")
                profiled.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
            try:
                profiled.wait(timeout=15)
            except subprocess.TimeoutExpired:
                profiled.terminate()
                try:
                    profiled.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    profiled.kill()
                    profiled.wait()
        if engine_pid is not None and Path(f"/proc/{engine_pid}").exists():
            try:
                command_line = Path(f"/proc/{engine_pid}/cmdline").read_bytes().split(b"\0")
                if config["exe"].encode() in command_line:
                    os.kill(engine_pid, signal.SIGTERM)
                    for _ in range(100):
                        if not Path(f"/proc/{engine_pid}").exists():
                            break
                        time.sleep(0.1)
                    if Path(f"/proc/{engine_pid}").exists():
                        os.kill(engine_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if dmon is not None and dmon.poll() is None:
            dmon.terminate()
            try:
                dmon.wait(timeout=5)
            except subprocess.TimeoutExpired:
                dmon.kill()
                dmon.wait()
        if dmon_file is not None:
            dmon_file.close()
        result["profiler"]["agent_pids_stopped"] = stop_owned_nsys_agent(session)
        stdout_file.close()
        stderr_file.close()
        result["finished_utc"] = utc()
        result["capture_stopped"] = capture_stopped
        result["compute_processes_after"] = compute_processes()
        (capture_dir / "monitor.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in monitor))
        (capture_dir / "capture.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        try:
            sanitized_config.unlink()
        except FileNotFoundError:
            pass
        try:
            result["api_start"] = restore_api()
            result["api_start_utc"] = utc()
        except Exception as exc:
            result["api_restore_error"] = f"{type(exc).__name__}: {exc}"
        (capture_dir / "capture.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not result.get("request", {}).get("output_ids_match_historical"):
        raise RuntimeError("capture failed exact output validation")
    if result.get("compute_processes_after"):
        raise RuntimeError(f"compute process remained after profiling: {result['compute_processes_after']}")
    report = Path(str(profile_output) + ".nsys-rep")
    if not report.exists():
        raise RuntimeError("Nsight Systems report was not saved")
    if "api_restore_error" in result:
        raise RuntimeError(f"API restore failed: {result['api_restore_error']}")
    artifacts = (
        "capture.json", "monitor.jsonl", "dmon-samples.txt", "proxy-control.log",
        "nsys.log", "timeline-export.sqlite", report.name,
    )
    (capture_dir / "SHA256SUMS").write_text("".join(
        f"{hashlib.sha256((capture_dir / name).read_bytes()).hexdigest()}  {name}\n"
        for name in artifacts
    ))
    print(json.dumps({
        "request": result["request"], "timestamps": result["timestamps"],
        "profiler": result["profiler"], "min_host_available_bytes": result["min_host_available_bytes"],
        "min_gpu_free_bytes": result["min_gpu_free_bytes"],
        "compute_processes_after": result["compute_processes_after"],
        "api_start": result["api_start"],
    }, indent=2))


def shutil_which(name):
    from shutil import which
    return which(name) is not None


if __name__ == "__main__":
    main()
