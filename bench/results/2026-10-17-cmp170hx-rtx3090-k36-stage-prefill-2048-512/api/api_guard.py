#!/usr/bin/env python3
"""Run the local K36 API with the authorized host-memory start/stop guards."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

API_DIR = Path(__file__).resolve().parent
CONFIG = API_DIR / "strata-api.json"
PIDFILE = API_DIR / "server.pid"
ROOT = API_DIR.parents[3]
GIB = 1 << 30
HOST_START_BYTES = 6 * GIB
HOST_STOP_BYTES = 5 * GIB
PORT = 8080
HOST = "127.0.0.1"
stop_signal = None


def mem_available_bytes():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is missing from /proc/meminfo")


def compute_processes():
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,name", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def config_check():
    cfg = json.loads(CONFIG.read_text())
    args = cfg["args"]
    required = {
        "--gpu": "1,0",
        "--layer-split": "36",
        "--max-context": "262144",
        "--kv": "int8",
        "--prefill": "2048",
        "--prefill-stage-chunks": "2048,512",
        "--vram-reserve-mib": "0",
        "--vram-reserve-later-mib": "0",
    }
    for flag, expected in required.items():
        try:
            actual = args[args.index(flag) + 1]
        except (ValueError, IndexError) as exc:
            raise RuntimeError(f"API config lacks {flag}") from exc
        if actual != expected:
            raise RuntimeError(f"API config has {flag}={actual}, expected {expected}")
    for flag in ("--serve", "--gpu-owned-experts", "--no-prefill-borrow"):
        if flag not in args:
            raise RuntimeError(f"API config lacks {flag}")
    if cfg.get("host") != HOST or cfg.get("port") != PORT:
        raise RuntimeError("API config must bind only to 127.0.0.1:8080")
    for path in (cfg["exe"], cfg["tokenizer"] + "/vocab.json",
                 cfg["tokenizer"] + "/merges.txt"):
        if not Path(path).exists():
            raise RuntimeError(f"API input does not exist: {path}")
    for flag in ("--pack", "--native", "--mtp", "--expert-profile"):
        try:
            path = args[args.index(flag) + 1]
        except (ValueError, IndexError) as exc:
            raise RuntimeError(f"API config lacks {flag}") from exc
        if not Path(path).exists():
            raise RuntimeError(f"API input does not exist: {path}")
    if "--api-key" in cfg or cfg.get("api_key"):
        raise RuntimeError("local API config should not carry an API key")
    return cfg


def on_signal(signum, _frame):
    global stop_signal
    stop_signal = signum


def stop_server(server, reason):
    if server.poll() is not None:
        return
    print(f"[api-guard] stopping API server: {reason}", flush=True)
    server.send_signal(signal.SIGTERM)
    try:
        server.wait(timeout=30)
    except subprocess.TimeoutExpired:
        print("[api-guard] server did not stop after SIGTERM; sending SIGKILL", flush=True)
        server.kill()
        server.wait()


def main():
    if mem_available_bytes() < HOST_START_BYTES:
        raise RuntimeError("host MemAvailable is below the required 6 GiB startup threshold")
    active = compute_processes()
    if active:
        raise RuntimeError(f"GPU compute process already exists; refusing to interfere: {active}")
    cfg = config_check()
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("STRATA_") or key in ("CUDA_VISIBLE_DEVICES", "LD_PRELOAD"):
            del env[key]
    env.update({str(key): str(value) for key, value in cfg["env"].items()})
    command = [
        sys.executable, "-m", "serve.server", "--engine", "strata",
        "--config", str(CONFIG), "--host", HOST, "--port", str(PORT),
    ]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, on_signal)
    print(f"[api-guard] host MemAvailable {mem_available_bytes()} bytes; startup threshold {HOST_START_BYTES}", flush=True)
    print(f"[api-guard] launching loopback OpenAI/Anthropic API on http://{HOST}:{PORT}", flush=True)
    server = subprocess.Popen(command, cwd=ROOT, env=env)
    try:
        while server.poll() is None:
            if stop_signal is not None:
                stop_server(server, f"received signal {stop_signal}")
                break
            available = mem_available_bytes()
            if available < HOST_STOP_BYTES:
                stop_server(server, f"host MemAvailable {available} bytes is below 5 GiB")
                return 75
            time.sleep(1)
        return server.wait()
    finally:
        if server.poll() is None:
            stop_server(server, "supervisor exit")
        try:
            if PIDFILE.read_text().strip() == str(os.getpid()):
                PIDFILE.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[api-guard] ERROR: {type(exc).__name__}: {exc}", flush=True)
        raise
