#!/usr/bin/env python3
"""tools/vision/test_vision_proxy.py - the three-tier vision cascade, tested without a GPU or downloads.

What it covers (every stub lives in a temporary directory; nothing is written into the repo):
  1. Tier 1: tools/vision/remote_daemon.py in front of a stub strata-vision; a proxy started with --remote-url
     answers READY from GET /health and passes every ENC through POST /encode, writing the .sve the daemon sent.
  2. Tier 2: no daemon; --intel-exe starts the local worker (the proxy gives it --gpu).
  3. Tier 3: --intel-exe is a missing path; --fallback-exe runs without --gpu and with CUDA_VISIBLE_DEVICES=-1
     (the stub prints the environment it was started with on stderr, which the test reads).
  4. Mid-flight fallback: the daemon dies between two pictures; the next ENC still answers OK through a local worker.
  5. No tier at all: "ERR no vision encoder could be started" on stdout and exit code 1.
  6. serve/server.py Vision(): the config keys remote_url/intel_exe/fallback_exe become --remote-url/--intel-exe/
     --fallback-exe (the exes absolutized); without those keys the argv is exactly what a plain strata-vision gets.

The stub speaks the strata-vision line protocol ("READY <n_embd>", "ENC <img> <out>" -> "OK ...", QUIT) and writes
a valid .sve file - int32 {0x31455653, n_tokens, nx, ny, n_embd} then float32 rows, as tools/vision/strata_vision.cpp
defines it - whose floats are seeded by its STRATA_STUB_TAG, so a test can tell which binary produced the bytes.

Run from the Strata directory:  python3 tools/vision/test_vision_proxy.py
"""
import io
import json
import os
import queue
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VISION = ROOT / "tools" / "vision"
PROXY, DAEMON = VISION / "strata_vision_proxy.py", VISION / "remote_daemon.py"
sys.path.insert(0, str(ROOT))                       # so case 6 can import serve.server offline
import serve.server as server                       # noqa: E402

SVE_MAGIC = 0x31455653                              # 'SVE1', as in tools/vision/strata_vision.cpp
EMBD, TOKENS, GRID_NX, GRID_NY = 8, 6, 3, 2         # what the stub answers with
READY_WAIT, ENCODE_WAIT = 120.0, 30.0               # generous: a stub answers at once

_STUB_BODY = r"""
import os
import struct
import sys

EMBD = int(os.environ.get("STRATA_STUB_EMBD", "8"))
TAG = os.environ.get("STRATA_STUB_TAG", "plain")
SEED = sum(TAG.encode("utf-8")) % 9973
CUDA = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")        # a marker for the environment the stub saw
print("stub-run tag=%s gpu=%s cuda=%s" % (TAG, "--gpu" in sys.argv[1:], CUDA), file=sys.stderr, flush=True)
print("READY %d" % EMBD, flush=True)
while True:
    line = sys.stdin.readline()
    if not line or line.strip() == "QUIT":
        break
    parts = line.strip().split(" ")
    if len(parts) != 3 or parts[0] != "ENC":
        print("ERR expected: ENC <image> <output>", flush=True)
        continue
    n, nx, ny = 6, 3, 2
    vals = [((SEED * 7919 + i) % 997) / 13.0 for i in range(n * EMBD)]
    with open(parts[2], "wb") as f:
        f.write(struct.pack("<5i", 0x31455653, n, nx, ny, EMBD) + struct.pack("<%df" % (n * EMBD), *vals))
    print("OK %d %d %d 42" % (n, nx, ny), flush=True)
sys.exit(0)
"""


def make_stub(directory):
    """An executable stand-in for strata-vision in a temp dir; its floats and stderr say which tag it ran with."""
    path = os.path.join(str(directory), "stub-vision")
    with open(path, "w", encoding="utf-8") as f:
        f.write("#!" + sys.executable + "\n" + _STUB_BODY)
    os.chmod(path, 0o755)                           # the proxy only starts an exe that is +x
    return path


def sve_bytes(tag, embd=EMBD):
    """The exact bytes the stub with this tag writes: header plus deterministic floats seeded by the tag."""
    seed = sum(tag.encode("utf-8")) % 9973
    vals = [((seed * 7919 + i) % 997) / 13.0 for i in range(TOKENS * embd)]
    return struct.pack("<5i", SVE_MAGIC, TOKENS, GRID_NX, GRID_NY, embd) + struct.pack("<%df" % (TOKENS * embd), *vals)


def free_port():
    """A port nothing is listening on: bind 0, read it back, close."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def read_line(pipe, timeout):
    """One line from a pipe, waiting at most `timeout` s (the same trick the proxy itself uses)."""
    got = queue.Queue()

    def reader():
        try:
            got.put(pipe.readline())
        except (OSError, ValueError):
            got.put("")

    threading.Thread(target=reader, daemon=True).start()
    try:
        line = got.get(timeout=timeout)
    except queue.Empty:
        raise AssertionError("nothing was answered within %.0f s" % timeout) from None
    return line.rstrip("\r\n")


class VisionProxyCascade(unittest.TestCase):
    """The proxy and the daemon, against stubs in a temp dir."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="strata-vision-test-"))
        self.stub = make_stub(self.tmp)
        self.procs, self.logs = [], []

    def tearDown(self):
        for p in self.procs:
            if p.poll() is None:
                p.kill()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            for stream in (p.stdin, p.stdout, p.stderr):
                try:
                    if stream:
                        stream.close()
                except OSError:
                    pass
        for f in self.logs:
            f.close()
        shutil.rmtree(str(self.tmp), ignore_errors=True)

    # helpers -----------------------------------------------------------------------------------------------

    def spawn(self, cmd, cwd=None, env=None, log=None):
        proc = subprocess.Popen(cmd, cwd=str(cwd) if cwd else None, env=env, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=log or subprocess.PIPE, text=True)
        self.procs.append(proc)
        return proc

    def start_proxy(self, extra_args, cwd=None, tag=None):
        env = dict(os.environ)
        if tag:
            env["STRATA_STUB_TAG"] = tag
        return self.spawn([sys.executable, str(PROXY)] + extra_args, cwd=cwd, env=env)

    def start_daemon(self, port, tag):
        env = dict(os.environ)
        env["STRATA_STUB_TAG"] = tag
        log = open(str(self.tmp / ("daemon-%d.log" % port)), "w", encoding="utf-8")
        self.logs.append(log)
        proc = self.spawn([sys.executable, str(DAEMON), "--exe", self.stub, "--mmproj", "m.gguf", "--model",
                           "t.gguf", "--port", str(port), "--device", "testbox"], env=env, log=log)
        return proc, log

    def wait_health(self, port, timeout=10.0):
        """GET /health, retrying a refused connection for up to `timeout` s (the daemon is still starting)."""
        url = "http://127.0.0.1:%d/health" % port
        deadline, last = time.time() + timeout, None
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=1.0) as r:
                    return json.loads(r.read().decode("utf-8", "replace"))
            except OSError as e:
                last = e
                time.sleep(0.05)
        raise AssertionError(f"the daemon on port {port} never answered /health ({last})")

    def expect_ready(self, proc):
        self.assertEqual(read_line(proc.stdout, READY_WAIT), "READY %d" % EMBD)

    def send(self, proc, line):
        proc.stdin.write(line + "\n")
        proc.stdin.flush()

    def quit_proxy(self, proc):
        try:
            proc.stdin.write("QUIT\n")
            proc.stdin.flush()
        except OSError:
            pass
        return proc.wait(timeout=10)

    def scratch(self, name="scratch"):
        d = self.tmp / name
        d.mkdir()
        (d / "pic.png").write_bytes(b"\x89PNG a fake picture for the stub")
        return d

    # the cases ---------------------------------------------------------------------------------------------

    def test_tier1_remote_daemon(self):
        """--remote-url and an answering daemon: READY from /health, ENC through POST /encode."""
        port = free_port()
        daemon, log = self.start_daemon(port, "remote")
        info = self.wait_health(port)
        self.assertEqual(info["status"], "ready")
        self.assertEqual(info["device"], "testbox")
        self.assertEqual(info["n_embd"], EMBD)
        cwd = self.scratch()
        proc = self.start_proxy(["--mmproj", "m.gguf", "--model", "t.gguf",
                                 "--remote-url", "http://127.0.0.1:%d" % port], cwd=cwd)
        self.expect_ready(proc)
        self.send(proc, "ENC pic.png out.sve")
        self.assertEqual(read_line(proc.stdout, ENCODE_WAIT), "OK %d %d %d 42" % (TOKENS, GRID_NX, GRID_NY))
        self.assertEqual((cwd / "out.sve").read_bytes(), sve_bytes("remote"))   # the daemon's body, byte for byte
        self.assertEqual(self.quit_proxy(proc), 0)
        self.wait_daemon(daemon)
        with open(log.name, encoding="utf-8") as f:
            self.assertIn("listening", f.read())

    def wait_daemon(self, proc):
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)

    def test_tier2_local_gpu_worker(self):
        """No daemon: --intel-exe is started with --gpu and answers the ENC line."""
        cwd = self.scratch()
        proc = self.start_proxy(["--mmproj", "m.gguf", "--model", "t.gguf", "--intel-exe", self.stub],
                                cwd=cwd, tag="intel")
        self.expect_ready(proc)
        self.send(proc, "ENC pic.png out.sve")
        self.assertEqual(read_line(proc.stdout, ENCODE_WAIT), "OK %d %d %d 42" % (TOKENS, GRID_NX, GRID_NY))
        self.assertEqual((cwd / "out.sve").read_bytes(), sve_bytes("intel"))
        self.assertEqual(self.quit_proxy(proc), 0)
        err = proc.stderr.read()
        self.assertIn("stub-run tag=intel gpu=True", err)          # the worker really was started with --gpu

    def test_tier3_cpu_fallback_env(self):
        """--intel-exe missing: --fallback-exe runs without --gpu and with CUDA_VISIBLE_DEVICES=-1."""
        cwd = self.scratch()
        missing = str(self.tmp / "no-such-vision")
        proc = self.start_proxy(["--mmproj", "m.gguf", "--model", "t.gguf", "--intel-exe", missing,
                                 "--fallback-exe", self.stub], cwd=cwd, tag="cpu")
        self.expect_ready(proc)
        self.send(proc, "ENC pic.png out.sve")
        self.assertEqual(read_line(proc.stdout, ENCODE_WAIT), "OK %d %d %d 42" % (TOKENS, GRID_NX, GRID_NY))
        self.assertEqual((cwd / "out.sve").read_bytes(), sve_bytes("cpu"))
        self.assertEqual(self.quit_proxy(proc), 0)
        self.assertIn("stub-run tag=cpu gpu=False cuda=-1", proc.stderr.read())

    def test_mid_flight_fallback_after_the_daemon_dies(self):
        """The daemon dies between two pictures: the second ENC still answers OK, through a local worker."""
        port = free_port()
        daemon, _ = self.start_daemon(port, "remote2")
        self.wait_health(port)
        cwd = self.scratch()
        proc = self.start_proxy(["--mmproj", "m.gguf", "--model", "t.gguf",
                                 "--remote-url", "http://127.0.0.1:%d" % port, "--intel-exe", self.stub],
                                cwd=cwd, tag="after")
        self.expect_ready(proc)
        self.send(proc, "ENC pic.png out1.sve")
        self.assertEqual(read_line(proc.stdout, ENCODE_WAIT), "OK %d %d %d 42" % (TOKENS, GRID_NX, GRID_NY))
        self.assertEqual((cwd / "out1.sve").read_bytes(), sve_bytes("remote2"))
        daemon.kill()                          # the other PC goes away mid-conversation
        daemon.wait(timeout=10)
        self.send(proc, "ENC pic.png out2.sve")
        self.assertEqual(read_line(proc.stdout, ENCODE_WAIT), "OK %d %d %d 42" % (TOKENS, GRID_NX, GRID_NY))
        self.assertEqual((cwd / "out2.sve").read_bytes(), sve_bytes("after"))   # the local worker took the picture
        self.assertEqual(self.quit_proxy(proc), 0)
        self.assertIn("the remote encoder failed this picture; trying a local one", proc.stderr.read())

    def test_no_tier_available(self):
        """Nothing to talk to: an ERR line on stdout and exit code 1."""
        port = free_port()                            # nothing is listening there
        proc = self.start_proxy(["--mmproj", "m.gguf", "--model", "t.gguf",
                                 "--remote-url", "http://127.0.0.1:%d" % port])
        out, _err = proc.communicate(timeout=30)
        self.assertIn("ERR no vision encoder could be started", out)
        self.assertEqual(proc.returncode, 1)


class ServeVisionForwarding(unittest.TestCase):
    """serve/server.py Vision(): the tier keys become proxy arguments; without them nothing changes."""

    class FakeProc:
        def __init__(self):
            self.stdout = io.StringIO("READY %d\n" % EMBD)

        def poll(self):
            return None

        def kill(self):
            pass

    def _argv_for(self, cfg):
        captured = []

        def fake_popen(what, args, **kw):
            captured.append(list(args))
            return ServeVisionForwarding.FakeProc()

        real_popen, real_contain = server.popen, server.contain
        server.popen, server.contain = fake_popen, lambda proc: None
        v = None
        try:
            v = server.Vision(cfg, lazy=True)         # lazy: nothing starts until the first use, as serve does
            v._start()                                # the first use: popen is the fake, and READY comes back
        finally:
            server.popen, server.contain = real_popen, real_contain
            if v is not None:
                shutil.rmtree(str(v.dir), ignore_errors=True)
        self.assertEqual(len(captured), 1)
        return captured[0]

    def test_forwards_the_tier_keys(self):
        argv = self._argv_for({"exe": "proxy", "mmproj": "m.gguf", "model": "t.gguf",
                               "remote_url": "http://h:8085", "intel_exe": "engine/strata-vision-intel",
                               "fallback_exe": "engine/strata-vision"})
        self.assertEqual(argv[:5], ["proxy", "--mmproj", "m.gguf", "--model", "t.gguf"])
        pairs = list(zip(argv, argv[1:]))
        self.assertIn(("--remote-url", "http://h:8085"), pairs)
        self.assertIn(("--intel-exe", os.path.abspath("engine/strata-vision-intel")), pairs)
        self.assertIn(("--fallback-exe", os.path.abspath("engine/strata-vision")), pairs)

    def test_without_the_keys_the_argv_is_unchanged(self):
        argv = self._argv_for({"exe": "proxy", "mmproj": "m.gguf", "model": "t.gguf"})
        self.assertEqual(argv, ["proxy", "--mmproj", "m.gguf", "--model", "t.gguf"])


if __name__ == "__main__":
    unittest.main()
