#!/usr/bin/env python3
"""Build, size, and run the guarded CMP 170HX + RTX 3090 K36 local API."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BUILD = Path(os.environ.get("STRATA_170HX_BUILD", ROOT / "build/cuda-170hx-3090")).resolve()
RUNTIME = BUILD / "runner-runtime"
PACK = Path(os.environ.get("STRATA_PACK", ROOT / "build/pack/ud-q4-k-xl")).resolve()
DATA = Path(os.environ.get("STRATA_DATA_DIR", Path.home() / "projects/Strata-data")).resolve()
NATIVE = Path(os.environ.get(
    "STRATA_NATIVE",
    DATA / "models/UD-Q4_K_XL/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
)).resolve()
MTP = Path(os.environ.get("STRATA_MTP", DATA / "mtp/rt")).resolve()
PROFILE = Path(os.environ.get("STRATA_EXPERT_PROFILE", ROOT / "data/expert-profile.bin")).resolve()
DRAFT_VOCAB = ROOT / "data/draft_vocab_cyrillic.bin"
HOST = "127.0.0.1"
PORT = 8080
GIB = 1 << 30
MIB = 1 << 20
HOST_START = 6 * GIB
HOST_STOP = 5 * GIB
PROMPT_INIT_HEADROOM = 512 * MIB
PIDFILE = RUNTIME / "api-supervisor.pid"
CONFIG = RUNTIME / "strata-api.json"
LOG = RUNTIME / "server.log"
ENGINE_LOG = RUNTIME / "engine.log"
stop_signal = None

ENVIRONMENT = {
    "STRATA_MMQ_KQ": "1",
    "STRATA_KQ_SPLIT": "1",
    "STRATA_PREFILL_BATCH_GATHER": "1",
    "STRATA_SIDE_MIXER": "2",
    "STRATA_ATTN_DAILY": "1",
    "STRATA_GR_UP75": "1",
    "STRATA_QSA_ROWS": "1",
    "STRATA_B6_MMVQ_ROWS": "1",
    "STRATA_SPLIT_RING": "96",
    "STRATA_FILL_AHEAD": "0",
    "STRATA_IO_PREFETCH": "0",
    "STRATA_PIN_RESERVE_GIB": "4",
    "STRATA_REFILL_BLOCKING": "1",
    "STRATA_RESIDENT_HEADROOM_GIB": "4",
    "STRATA_STAGE_PIN": "0",
    "STRATA_UNBUFFERED_LOAD": "1",
}


def run(command, *, capture=False):
    return subprocess.run(
        [str(part) for part in command],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=capture,
    )


def engine_path():
    return BUILD / "strata"


def ledger_path():
    return BUILD / "strata-split-memory"


def cmake_build():
    command = [
        "cmake", "-S", ROOT, "-B", BUILD,
        "-DCMAKE_BUILD_TYPE=Release",
        "-DSTRATA_ENABLE_CUDA=ON",
        "-DCMAKE_CUDA_ARCHITECTURES=80;86",
        "-DSTRATA_MMQ_KQUANTS=ON",
        "-DSTRATA_MMQ_TAB=OFF",
        "-DSTRATA_BUILD_TESTS=OFF",
    ]
    nvcc = os.environ.get("CUDACXX") or shutil.which("nvcc")
    if not nvcc:
        for candidate in (Path("/usr/local/cuda/bin/nvcc"), Path("/usr/local/cuda-13.4/bin/nvcc")):
            if candidate.is_file():
                nvcc = str(candidate)
                break
    if nvcc:
        command.append(f"-DCMAKE_CUDA_COMPILER={nvcc}")
    run(command)
    jobs = int(os.environ.get("STRATA_BUILD_JOBS", "2"))
    if jobs < 1 or jobs > 128:
        raise RuntimeError("STRATA_BUILD_JOBS must be between 1 and 128")
    run(["cmake", "--build", BUILD, "--parallel", jobs, "--target",
         "strata", "strata_kq", "strata-split-memory"])


def check_paths():
    required = {
        "engine": engine_path(),
        "split-memory ledger": ledger_path(),
        "pack index": PACK / "index.txt",
        "pack tokenizer vocab": PACK / "tokenizer/vocab.json",
        "pack tokenizer merges": PACK / "tokenizer/merges.txt",
        "native GGUF shard": NATIVE,
        "MTP dense weights": MTP / "dense.bin",
        "MTP expert weights": MTP / "experts.bin",
        "expert profile": PROFILE,
        "Cyrillic MTP draft vocabulary": DRAFT_VOCAB,
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not path.exists()]
    if missing:
        raise RuntimeError("required runner input is missing:\n  " + "\n  ".join(missing) +
                           "\nSet STRATA_PACK, STRATA_NATIVE, STRATA_MTP, or STRATA_EXPERT_PROFILE as needed.")


def validate_engine_flags(help_text):
    required = ("--prefill-stage-chunks", "--mtp-draft-vocab", "--gpu-owned-experts")
    missing = [flag for flag in required if flag not in help_text]
    if missing:
        raise RuntimeError(
            f"{engine_path()} is stale or was built without runner options: {', '.join(missing)}. "
            "Run ./run-170hx-3090.sh build before check or start."
        )


def check_engine_flags():
    result = run([engine_path(), "--help"], capture=True)
    validate_engine_flags(result.stdout + result.stderr)


def memory_available():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is missing from /proc/meminfo")


def gpu_rows():
    result = run([
        "nvidia-smi", "--query-gpu=index,name,memory.free,memory.total",
        "--format=csv,noheader,nounits",
    ], capture=True)
    rows = {}
    for line in result.stdout.splitlines():
        index, name, free_mib, total_mib = (item.strip() for item in line.split(",", 3))
        rows[int(index)] = {
            "name": name,
            "free_bytes": int(free_mib) * MIB,
            "total_bytes": int(total_mib) * MIB,
        }
    expected = {0: "RTX 3090", 1: "CMP 170HX"}
    for index, name in expected.items():
        if index not in rows or name not in rows[index]["name"]:
            raise RuntimeError(f"expected physical GPU {index} to be {name}; got {rows.get(index)}")
    return rows


def compute_processes():
    result = run([
        "nvidia-smi", "--query-compute-apps=pid,name", "--format=csv,noheader",
    ], capture=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def api_is_listening():
    with socket.socket() as probe:
        return probe.connect_ex((HOST, PORT)) == 0


def api_health():
    try:
        with urllib.request.urlopen(f"http://{HOST}:{PORT}/health", timeout=1) as response:
            health = json.load(response)
    except (OSError, ValueError, urllib.error.URLError):
        return None
    if not isinstance(health, dict) or health.get("status") != "ok" or health.get("service") != "strata":
        return None
    return health


def build_config():
    args = [
        "--pack", str(PACK),
        "--native", str(NATIVE),
        "--gpu", "1,0",
        "--layer-split", "36",
        "--trim-stage-weights",
        "--max-context", "262144",
        "--kv", "int8",
        "--mtp", str(MTP),
        "--spec", "4",
        "--prefill", "2048",
        "--prefill-stage-chunks", "2048,512",
        "--no-prefill-borrow",
        "--mmap-experts",
        "--resident-budget-gib", "0.001",
        "--expert-profile", str(PROFILE),
        "--ple-io", "direct",
        "--ple-row-cache", "1024",
        "--mtp-draft-vocab", str(DRAFT_VOCAB),
        "--vram-reserve-mib", "0",
        "--vram-reserve-later-mib", "0",
        "--pool-workers", "1",
        "--serve",
        "--conversation-cache-min-free-mib", "128",
        "--conversation-cache-mib", "16384",
        "--conversation-cache-slots", "32",
        "--gpu-owned-experts",
    ]
    return {
        "exe": str(engine_path()),
        "args": args,
        "cwd": str(ROOT),
        "tokenizer": str(PACK / "tokenizer"),
        "log": str(ENGINE_LOG),
        "model_name": "qwen3.8-flash-next-unsloth-ud-q4_k_xl",
        "host": HOST,
        "port": PORT,
        "env": ENVIRONMENT,
    }


def sizing_admission():
    RUNTIME.mkdir(parents=True, exist_ok=True)
    command = [
        ledger_path(),
        "--pack", PACK,
        "--native", NATIVE,
        "--mtp", MTP,
        "--draft-vocab", DRAFT_VOCAB,
        "--context", "262144",
        "--chunk", "2048",
        "--prefill-stage-chunks", "2048,512",
        "--additional-reserve-mib", "0",
    ]
    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith("STRATA_") and key not in ("CUDA_VISIBLE_DEVICES", "LD_PRELOAD")
    }
    env.update(ENVIRONMENT)
    result = subprocess.run(
        [str(part) for part in command],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    rows = list(csv.DictReader(result.stdout.splitlines()))
    gpu = gpu_rows()
    stage_physical = {"0": 1, "1": 0}
    selected = {}
    for row in rows:
        if row["split"] == "36" and row["stage"] in stage_physical and \
                row["component"] == "required_free_before_model_load":
            selected[row["stage"]] = int(row["bytes"])
    if set(selected) != {"0", "1"}:
        raise RuntimeError("split-memory ledger did not report both K36 GPU stages")

    report = {
        "command": [str(part) for part in command],
        "draft_vocabulary": str(DRAFT_VOCAB),
        "host_mem_available_bytes": memory_available(),
        "host_start_floor_bytes": HOST_START,
        "host_stop_floor_bytes": HOST_STOP,
        "prompt_init_headroom_bytes": PROMPT_INIT_HEADROOM,
        "stage_admission": {},
    }
    for stage, physical in stage_physical.items():
        required = selected[stage] + PROMPT_INIT_HEADROOM
        free = gpu[physical]["free_bytes"]
        report["stage_admission"][stage] = {
            "physical_gpu": physical,
            "gpu_name": gpu[physical]["name"],
            "free_bytes": free,
            "required_with_prompt_init_headroom_bytes": required,
            "margin_bytes": free - required,
            "passes": free >= required,
        }
    (RUNTIME / "sizing.csv").write_text(result.stdout)
    (RUNTIME / "sizing.log").write_text(result.stderr)
    (RUNTIME / "admission.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["host_mem_available_bytes"] < HOST_START:
        raise RuntimeError(
            f"host MemAvailable is {report['host_mem_available_bytes'] / GIB:.2f} GiB; "
            "startup requires at least 6 GiB"
        )
    failures = [stage for stage, value in report["stage_admission"].items() if not value["passes"]]
    if failures:
        detail = "; ".join(
            f"stage {stage} {report['stage_admission'][stage]['gpu_name']}: "
            f"free={report['stage_admission'][stage]['free_bytes']} B, "
            f"required={report['stage_admission'][stage]['required_with_prompt_init_headroom_bytes']} B"
            for stage in failures
        )
        raise RuntimeError("mandatory K36 CUDA admission fails with Cyrillic draft vocabulary: " + detail)
    return report, result


def preflight():
    check_paths()
    check_engine_flags()
    if memory_available() < HOST_START:
        raise RuntimeError("host MemAvailable is below the required 6 GiB startup threshold")
    if api_is_listening():
        raise RuntimeError("127.0.0.1:8080 is already in use; refusing to replace a running service")
    active = compute_processes()
    if active:
        raise RuntimeError("NVIDIA compute process already exists; refusing to overlap: " + ", ".join(active))
    report, ledger_result = sizing_admission()
    print(ledger_result.stderr, end="")
    print(ledger_result.stdout, end="")
    for stage, values in report["stage_admission"].items():
        print(f"K36 stage {stage} {values['gpu_name']}: free {values['free_bytes']} B, "
              f"required incl. 512 MiB headroom {values['required_with_prompt_init_headroom_bytes']} B, "
              f"margin {values['margin_bytes']} B")
    config = build_config()
    CONFIG.write_text(json.dumps(config, indent=2) + "\n")
    return config


def process_args(pid):
    try:
        data = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return ""
    return data


def supervisor_valid(pid):
    args = process_args(pid)
    return bool(args) and str(Path(__file__).resolve()) in args and "supervise" in args


def on_signal(signum, _frame):
    global stop_signal
    stop_signal = signum


def stop_child(child, reason):
    if child.poll() is not None:
        return
    print(f"[170hx-3090] stopping API: {reason}", flush=True)
    child.send_signal(signal.SIGTERM)
    try:
        child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        print("[170hx-3090] server did not exit after SIGTERM; sending SIGKILL to its verified child", flush=True)
        child.kill()
        child.wait()


def supervise():
    child = None
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, on_signal)
    try:
        available = memory_available()
        if available < HOST_START:
            raise RuntimeError("host MemAvailable is below the required 6 GiB startup threshold")
        env = os.environ.copy()
        for key in list(env):
            if key.startswith("STRATA_") or key in ("CUDA_VISIBLE_DEVICES", "LD_PRELOAD"):
                del env[key]
        env.update(ENVIRONMENT)
        command = [
            sys.executable, "-m", "serve.server", "--engine", "strata",
            "--config", str(CONFIG), "--host", HOST, "--port", str(PORT),
        ]
        print(f"[170hx-3090] launching loopback API; host MemAvailable {available} bytes", flush=True)
        child = subprocess.Popen(command, cwd=ROOT, env=env)
        while child.poll() is None:
            if stop_signal is not None:
                stop_child(child, f"received signal {stop_signal}")
                break
            available = memory_available()
            if available < HOST_STOP:
                stop_child(child, f"host MemAvailable {available} bytes is below 5 GiB")
                return 75
            time.sleep(1)
        return child.wait()
    finally:
        if child is not None and child.poll() is None:
            stop_child(child, "supervisor exit")
        try:
            if PIDFILE.read_text().strip() == str(os.getpid()):
                PIDFILE.unlink()
        except FileNotFoundError:
            pass


def start():
    if PIDFILE.exists():
        text = PIDFILE.read_text().strip()
        if text.isdigit() and supervisor_valid(int(text)):
            raise RuntimeError(f"runner already active as PID {text}; use stop/status")
    preflight()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    log = LOG.open("a", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "-B", str(Path(__file__).resolve()), "supervise"],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log.close()
    PIDFILE.write_text(f"{process.pid}\n")
    time.sleep(2)
    if process.poll() is not None:
        PIDFILE.unlink(missing_ok=True)
        details = "\n".join(
            f"--- {path} ---\n{tail_file(path)}"
            for path in (ENGINE_LOG, LOG) if path.exists()
        )
        message = f"API supervisor exited during startup (code {process.returncode})."
        if details:
            message += "\n" + details
        raise RuntimeError(message)
    health = api_health()
    if health and health.get("loaded") is True:
        print(f"K36 API ready at http://{HOST}:{PORT}; supervisor PID {process.pid}; log: {LOG}")
    else:
        print(f"K36 startup is still in progress; supervisor PID {process.pid}; log: {LOG}")
        print("The API is not ready yet. Check with status; readiness requires /health loaded=true.")


def tail_file(path, lines=30):
    try:
        return "\n".join(Path(path).read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError as exc:
        return f"could not read log: {exc}"


def stop():
    if not PIDFILE.exists():
        raise RuntimeError(f"no runner PID file found at {PIDFILE}")
    text = PIDFILE.read_text().strip()
    if not text.isdigit() or not supervisor_valid(int(text)):
        raise RuntimeError(f"PID file does not identify this runner; refusing to signal PID {text!r}")
    pid = int(text)
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline:
        if not supervisor_valid(pid):
            PIDFILE.unlink(missing_ok=True)
            print(f"Stopped K36 API supervisor PID {pid}")
            return
        time.sleep(0.25)
    raise RuntimeError(f"runner PID {pid} did not exit; inspect {LOG}; no unrelated process was signaled")


def status():
    if PIDFILE.exists():
        pid = PIDFILE.read_text().strip()
        active = pid.isdigit() and supervisor_valid(int(pid))
    else:
        active = False
        pid = ""
    print(f"supervisor: {'running PID ' + pid if active else 'stopped'}")
    health = api_health()
    if health and health.get("loaded") is True:
        print(f"API: ready at http://{HOST}:{PORT} (model loaded)")
    elif health:
        print(f"API: responding at http://{HOST}:{PORT}, but model is not loaded")
    else:
        print("API: not ready (health endpoint unavailable)")
    print(f"host MemAvailable: {memory_available() / GIB:.2f} GiB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("build", "check", "start", "stop", "status", "supervise"),
                        default="start")
    args = parser.parse_args()
    if args.command == "build":
        cmake_build()
        print(f"Built CUDA targets in {BUILD}")
    elif args.command == "check":
        preflight()
        print(f"Admission passed; report: {RUNTIME / 'admission.json'}")
    elif args.command == "start":
        start()
    elif args.command == "stop":
        stop()
    elif args.command == "status":
        status()
    elif args.command == "supervise":
        return supervise()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"run-170hx-3090: ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
