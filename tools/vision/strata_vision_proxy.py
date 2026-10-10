#!/usr/bin/env python3
"""tools/vision/strata_vision_proxy.py - strata-vision's stdin/stdout protocol, three encoders behind it.

A drop-in for "exe" in strata-<model>.json: serve/server.py starts it with the same argv it would give strata-vision
and passes three extra keys from the config's vision section (remote_url, intel_exe, fallback_exe).  It speaks the
same line protocol as strata-vision - "READY <n_embd>", then per stdin line "ENC <image> <output>" ->
"OK <n_tokens> <nx> <ny> <ms>" or "ERR <message>", and QUIT - and picks the first tier that answers:
    1. a daemon on another PC (--remote-url, tools/vision/remote_daemon.py), checked with GET /health at start;
    2. a local build with a GPU (--intel-exe), started with --gpu;
    3. a local CPU build (--fallback-exe, CUDA_VISIBLE_DEVICES=-1).
If the daemon stops answering a picture, the same ENC line goes to a local worker and its reply is passed through;
a local worker that dies is started again once.  Which tier is in use, and every fallback, is noted on stderr.
Unknown arguments are ignored with a note (a plain strata-vision stops on them).
"""
import argparse
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import urllib.request

HEALTH_S = 1.5        # the start must not wait on a PC that is off: fall back fast, the local encoder is close by
REMOTE_ENCODE_S = 120.0    # one picture over the wire, shorter than serve/server.py's VISION_ENCODE_S
READY_S = 300.0            # a local worker loading its model files, as long as serve/server.py waits


def note(message):
    print(f"strata-vision-proxy: {message}", file=sys.stderr, flush=True)


class ReadTimeout(Exception):
    pass


def read_line(pipe, timeout):
    """One line from a pipe, waiting at most `timeout` s.  Raises ReadTimeout; "" means the pipe closed."""
    got = queue.Queue()

    def reader():
        try:
            got.put(pipe.readline())
        except (OSError, ValueError):               # the process was killed under the read
            got.put("")

    threading.Thread(target=reader, daemon=True).start()
    try:
        return got.get(timeout=timeout)
    except queue.Empty:
        raise ReadTimeout(f"it said nothing for {timeout:.0f} s") from None


def parse_args():
    """strata-vision's arguments plus --remote-url/--intel-exe/--fallback-exe; unknown ones are ignored."""
    cfg = argparse.Namespace(mmproj=None, model=None, gpu=False, threads=None, max_tokens=None, min_tokens=None,
                             remote_url=None, intel_exe=None, fallback_exe=None)
    value_flags = ("--mmproj", "--model", "--threads", "--max-tokens", "--min-tokens",
                   "--remote-url", "--intel-exe", "--fallback-exe")
    argv, i = sys.argv[1:], 0
    while i < len(argv):
        a = argv[i]
        if a == "--gpu":
            cfg.gpu = True
            i += 1
            continue
        if a in value_flags:
            if i + 1 >= len(argv):
                note(f"{a} needs a value")
                sys.exit(2)
            value, i = argv[i + 1], i + 2
        else:
            note(f"unknown argument {a} (ignored)")
            i += 2 if i + 1 < len(argv) and not argv[i + 1].startswith("-") else 1
            continue
        dest = a[2:].replace("-", "_")
        if dest in ("threads", "max_tokens", "min_tokens"):
            try:
                setattr(cfg, dest, int(value))
            except ValueError:
                note(f"{a} needs a number (ignored)")
        else:
            setattr(cfg, dest, value)
    return cfg


def parse_enc(line):
    """'ENC <image> <out>' -> (img, out), or None.  Paths may contain spaces, so the image path ends at the last
    space before the output path (as in strata_vision.cpp)."""
    if not line.startswith("ENC "):
        return None
    rest = line[4:]
    sp = rest.rfind(" ")
    if sp <= 0 or sp == len(rest) - 1:
        return None
    return rest[:sp], rest[sp + 1:]


class Worker:
    """A local strata-vision this proxy starts: tier 2 with --gpu, tier 3 on the CPU (no CUDA context)."""

    def __init__(self, exe, cfg, gpu):
        self.exe, self.gpu, self.ready, self.proc = exe, gpu, "", None
        cmd = [exe, "--mmproj", cfg.mmproj, "--model", cfg.model]
        if gpu:
            cmd.append("--gpu")
        for flag, value in (("--threads", cfg.threads), ("--max-tokens", cfg.max_tokens),
                            ("--min-tokens", cfg.min_tokens)):
            if value is not None:
                cmd += [flag, str(value)]
        env = dict(os.environ)
        if not gpu:               # as in strata_vision.cpp: a CUDA build must not open a context on the CPU
            env["CUDA_VISIBLE_DEVICES"] = "-1"
        elif exe == cfg.intel_exe:
            for icd in ("/usr/share/vulkan/icd.d/intel_icd.json", "/usr/share/vulkan/icd.d/intel_hasvk_icd.json"):
                if os.path.exists(icd):
                    env["VK_DRIVER_FILES"] = icd
                    env["VK_ICD_FILENAMES"] = icd
                    break
            env.setdefault("GGML_VK_VISIBLE_DEVICES", "0")
        try:                      # cwd is ours: serve/server.py runs the proxy and any worker in the same scratch dir,
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                         encoding="utf-8", bufsize=1, cwd=os.getcwd(), env=env)  # and its ENC lines are relative
            self.ready = read_line(self.proc.stdout, READY_S).rstrip("\r\n")
        except (OSError, ReadTimeout) as e:
            note(f"{exe} did not start: {e}")
            self.proc = None
            return
        if not self.ready.startswith("READY"):
            note(f"{exe} did not get ready: {self.ready or 'it closed its output'}")
            try:
                self.proc.kill()
            except OSError:
                pass
            self.proc = None


def start_local(cfg):
    """Tier 2 (--intel-exe), then tier 3 (--fallback-exe).  -> a ready Worker, or None."""
    for exe, gpu in ((cfg.intel_exe, True), (cfg.fallback_exe, False)):
        if not exe or not os.path.isfile(exe) or not os.access(exe, os.X_OK):
            continue
        note(f"starting a local encoder: {exe} ({'gpu' if gpu else 'cpu'})")
        worker = Worker(exe, cfg, gpu)
        if worker.proc is not None:
            return worker
    return None


def remote_health(cfg):
    """GET <remote_url>/health, at most 1.5 s.  -> n_embd, or None when the daemon is not answering."""
    try:
        with urllib.request.urlopen(cfg.remote_url.rstrip("/") + "/health", timeout=HEALTH_S) as r:
            if r.status != 200:
                return None
            info = json.loads(r.read().decode("utf-8", "replace"))
        n = int(info.get("n_embd", 0))
    except (OSError, ValueError):
        return None
    return n if n > 0 else None


def remote_encode(cfg, data, t0):
    """POST the image bytes to the daemon, at most 120 s.  -> (.sve bytes, tokens, nx, ny, ms), or None on any
    failure (connection refused, timeout, non-200, missing X- headers)."""
    req = urllib.request.Request(cfg.remote_url.rstrip("/") + "/encode", data=data, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=REMOTE_ENCODE_S) as r:
            if r.status != 200:
                note(f"the remote encoder answered HTTP {r.status}")
                return None
            sve, headers = r.read(), r.headers
    except OSError as e:
        note(f"the remote encoder did not answer: {e}")
        return None
    tokens, nx, ny = (headers.get(k) for k in ("X-Tokens", "X-Grid-Nx", "X-Grid-Ny"))
    if not tokens or not nx or not ny:
        note("the remote encoder left out the X-Tokens/X-Grid headers")
        return None
    ms = headers.get("X-Duration-Ms") or f"{(time.monotonic() - t0) * 1000:.0f}"      # else measure it here
    return sve, tokens, nx, ny, ms


worker = None    # the local worker, if any; which tier is live is decided in main()


def stop_worker():
    """QUIT to the local worker, wait 5 s, terminate and kill if it stays."""
    global worker
    to_quit, worker = worker, None
    if to_quit is None:
        return
    try:
        to_quit.proc.stdin.write("QUIT\n")
        to_quit.proc.stdin.flush()
        to_quit.proc.wait(timeout=5)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        try:
            to_quit.proc.terminate()
            to_quit.proc.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            try:
                to_quit.proc.kill()
            except OSError:
                pass


def on_term(signum, frame):
    """SIGTERM is a QUIT line: the server is going away, so the worker goes with it."""
    stop_worker()
    os._exit(0)


def forward(cfg, line):
    """One ENC line to the local worker; if it died (EOF), restart once and retry."""
    global worker
    for attempt in (1, 2):
        if worker is None:
            worker = start_local(cfg)
            if worker is None:
                print("ERR no vision encoder could be started", flush=True)
                return
        try:
            worker.proc.stdin.write(line + "\n")
            worker.proc.stdin.flush()
            reply = worker.proc.stdout.readline().rstrip("\r\n")
        except (OSError, ValueError):               # the worker's pipes are gone
            reply = ""
        if reply:
            print(reply, flush=True)
            return
        note(f"the local encoder stopped ({'restarting it' if attempt == 1 else 'no retry left'})")
        worker = None
    print("ERR the vision encoder stopped", flush=True)


def handle_remote(cfg, line, img, out):
    """One picture through the remote daemon; on any failure the same ENC line goes to a local worker."""
    global worker
    try:
        with open(img, "rb") as f:                # paths are relative to cwd, as serve/server.py writes them
            data = f.read()
    except OSError:
        print(f"ERR cannot read the image {img}", flush=True)
        return
    t0 = time.monotonic()
    got = remote_encode(cfg, data, t0)
    if got is not None:
        sve, tokens, nx, ny, ms = got
        try:
            with open(out, "wb") as f:
                f.write(sve)
        except OSError:
            print(f"ERR cannot write {out}", flush=True)
            return
        print(f"OK {tokens} {nx} {ny} {ms}", flush=True)
        return
    note("the remote encoder failed this picture; trying a local one")
    if worker is None:
        worker = start_local(cfg)
        if worker is None:
            print("ERR no vision encoder could be started", flush=True)
            return
    forward(cfg, line)


def main():
    global worker
    cfg = parse_args()
    if not cfg.mmproj or not cfg.model:
        note("usage: strata-vision-proxy --mmproj <mmproj.gguf> --model <model.gguf> [--gpu] [--threads N] "
             "[--max-tokens N] [--min-tokens N] [--remote-url URL] [--intel-exe EXE] [--fallback-exe EXE]")
        return 2

    tier = 0
    if cfg.remote_url:
        n = remote_health(cfg)
        if n:
            tier = 1
            print(f"READY {n}", flush=True)
            note(f"tier 1: the remote encoder at {cfg.remote_url} (n_embd {n})")
    if tier == 0:
        worker = start_local(cfg)
        if worker is not None:
            tier = 2 if worker.gpu else 3
            print(worker.ready, flush=True)      # echo the worker's READY line on our own stdout
            note(f"tier {tier}: the local encoder {worker.exe}")
    if tier == 0:
        print("ERR no vision encoder could be started", flush=True)
        return 1

    signal.signal(signal.SIGTERM, on_term)
    while True:
        line = sys.stdin.readline()
        if not line:
            stop_worker()
            return 0
        line = line.rstrip("\r\n")
        if line == "QUIT":
            stop_worker()
            return 0
        enc = parse_enc(line)
        if enc is None:
            print("ERR expected: ENC <image> <output>", flush=True)
            continue
        if tier == 1:
            handle_remote(cfg, line, *enc)
        else:
            forward(cfg, line)


if __name__ == "__main__":
    sys.exit(main())
