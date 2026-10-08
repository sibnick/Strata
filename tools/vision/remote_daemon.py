#!/usr/bin/env python3
"""tools/vision/remote_daemon.py - strata-vision on a second PC, behind a small HTTP server.

Starts the resident image encoder (tools/vision/strata_vision.cpp) once and answers two requests:
    GET  /health   -> 200 {"status": "ready", "device": <name>, "n_embd": <n>}
    POST /encode   -> the image's raw bytes as the body; the answer is the .sve file strata-vision wrote, with
                      X-Tokens, X-Grid-Nx, X-Grid-Ny and X-Duration-Ms headers taken from its OK line.
                      A body over 64 MiB or a missing Content-Length is refused; anything else is a 404.
The encodes are serialized with a lock: the encoder is single-threaded, one ENC line at a time.

On the GTX 1060 PC (Ubuntu), from the Strata directory:
    python3 tools/vision/remote_daemon.py --mmproj <mmproj.gguf> --model <model.gguf> --port 8085
strata-vision is started with --gpu (that is what this PC is for); --no-gpu runs it on its CPU, --exe points at
another build.  The main PC reaches this through "remote_url" in its config (see tools/vision/strata_vision_proxy.py):
the picture goes over the network, the embeddings come back, and a machine without a spare GPU never runs the encoder
itself.  There is no authentication, so keep the daemon inside the LAN or behind a VPN.
"""
import argparse
import json
import os
import queue
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

IMAGE_MAX = 64 << 20            # a POST /encode body over this is refused before any of it is read
READY_S = 600.0                 # the model files can take minutes on a cold disk
REPLY_S = 300.0                 # one ENC line, as long as what serve/server.py waits for


def note(message):
    print(f"remote-daemon: {message}", file=sys.stderr, flush=True)


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
        raise ReadTimeout(f"the encoder said nothing for {timeout:.0f} s") from None


class Encoder:
    """The resident strata-vision process: one ENC line in, one OK/ERR line out.  Every request goes through a
    lock, because the encoder is single-threaded."""

    def __init__(self, proc, directory, n_embd):
        self.proc = proc
        self.dir = directory
        self.n_embd = n_embd
        self.lock = threading.Lock()
        self.seq = 0

    def encode(self, data):
        """The image's bytes -> ("OK", (tokens, nx, ny, ms, sve bytes)) or ("ERR", message)."""
        with self.lock:
            self.seq += 1
            img = os.path.join(self.dir, f"{self.seq:06d}.img")    # absolute paths: the encoder's cwd is our dir
            out = os.path.join(self.dir, f"{self.seq:06d}.sve")
            try:
                try:
                    with open(img, "wb") as f:
                        f.write(data)
                    self.proc.stdin.write(f"ENC {img} {out}\n")
                    self.proc.stdin.flush()
                    line = read_line(self.proc.stdout, REPLY_S).rstrip("\r\n")
                except ReadTimeout as e:
                    line = "ERR " + str(e)
                except (OSError, ValueError):                       # the encoder's pipes are gone
                    line = "ERR the vision encoder stopped"
            finally:
                try:
                    os.remove(img)
                except OSError:
                    pass
            if not line.startswith("OK"):
                return ("ERR", line[4:] if line.startswith("ERR") else "the vision encoder stopped")
            parts = line.split()
            if len(parts) != 5:
                return ("ERR", "the vision encoder answered a strange reply")
            try:
                with open(out, "rb") as f:
                    sve = f.read()
            except OSError as e:
                return ("ERR", f"cannot read the embeddings file ({e})")
            finally:
                try:
                    os.remove(out)
                except OSError:
                    pass
            return ("OK", (parts[1], parts[2], parts[3], parts[4], sve))

    def close(self):
        """QUIT, then terminate, then kill; the temp dir goes either way."""
        try:
            self.proc.stdin.write("QUIT\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=5)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    self.proc.kill()
                except OSError:
                    pass
        shutil.rmtree(self.dir, ignore_errors=True)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    encoder = None              # set in main(), before the server starts
    device = ""

    def do_GET(self):
        if self.path.split("?", 1)[0] != "/health":
            return self._send(404, json.dumps({"error": "not found"}).encode())
        body = json.dumps({"status": "ready", "device": self.device, "n_embd": self.encoder.n_embd}).encode()
        self._send(200, body, "application/json")

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/encode":
            return self._send(404, json.dumps({"error": "not found"}).encode())
        try:
            length = int(self.headers.get("Content-Length"))
        except (TypeError, ValueError):
            return self._send(400, json.dumps({"error": "POST /encode needs a Content-Length header"}).encode())
        if length > IMAGE_MAX:
            return self._send(413, json.dumps({"error": "the image is over 64 MiB"}).encode())
        kind, info = self.encoder.encode(self.rfile.read(length))
        if kind != "OK":
            return self._send(500, json.dumps({"error": info}).encode())
        tokens, nx, ny, ms, sve = info
        self._send(200, sve, "application/octet-stream",
                   {"X-Tokens": tokens, "X-Grid-Nx": nx, "X-Grid-Ny": ny, "X-Duration-Ms": ms})

    def do_PUT(self):
        self._send(404, json.dumps({"error": "not found"}).encode())

    do_DELETE = do_HEAD = do_OPTIONS = do_PUT

    def _send(self, code, body, content_type="application/json", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="strata-vision behind HTTP, for a second PC with the GPU")
    ap.add_argument("--exe", default=os.path.join(script_dir, "..", "..", "engine", "strata-vision"))
    ap.add_argument("--mmproj", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8085)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--device", default=socket.gethostname(), help="the name reported in /health")
    ap.add_argument("--threads", type=int)
    ap.add_argument("--max-tokens", type=int)
    ap.add_argument("--min-tokens", type=int)
    ap.add_argument("--no-gpu", action="store_true", help="start strata-vision on the CPU (it gets --gpu by default)")
    args = ap.parse_args()

    cmd = [args.exe, "--mmproj", args.mmproj, "--model", args.model]
    if not args.no_gpu:
        cmd.append("--gpu")
    for flag, value in (("--threads", args.threads), ("--max-tokens", args.max_tokens),
                        ("--min-tokens", args.min_tokens)):
        if value is not None:
            cmd += [flag, str(value)]
    # the encoder's cwd and the place ENC lines name its files: a fresh temp dir without spaces (#480)
    tmp = tempfile.mkdtemp(prefix="strata-vision-daemon-")
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8",
                                bufsize=1, cwd=tmp)
    except OSError as e:
        note(f"cannot start {args.exe}: {e}")
        shutil.rmtree(tmp, ignore_errors=True)
        return 1
    line = ""
    try:
        line = read_line(proc.stdout, READY_S).rstrip("\r\n")
    except ReadTimeout as e:
        note(str(e))
    if not line.startswith("READY"):
        note(f"the encoder did not get ready: {line or 'it closed its output'}")
        proc.kill()
        shutil.rmtree(tmp, ignore_errors=True)
        return 1
    try:
        n_embd = int(line.split()[1])
    except (IndexError, ValueError):
        note(f"the encoder said {line!r} instead of READY <n_embd>")
        proc.kill()
        shutil.rmtree(tmp, ignore_errors=True)
        return 1

    encoder = Encoder(proc, tmp, n_embd)
    Handler.encoder = encoder
    Handler.device = args.device
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    note(f"listening on http://{args.host}:{args.port} (device {args.device}, n_embd {n_embd})")

    stopping = threading.Event()

    def on_signal(signum, frame):
        if not stopping.is_set():                     # serve_forever runs in this thread: shut down from a side one
            stopping.set()
            threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    try:
        server.serve_forever()
    finally:
        encoder.close()                              # QUIT, terminate, the temp dir goes
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
