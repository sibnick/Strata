#!/usr/bin/env python3
"""Sanitize one engine-control exchange while Nsight traces the engine child."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--token-ids-output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    engine = subprocess.Popen(
        [config["exe"], *config["args"]], cwd=config["cwd"], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    print(f"ENGINE_PID {engine.pid}", flush=True)
    ids = []
    request_sent = False
    first_token_sent = False
    try:
        for raw in engine.stdout:
            line = raw.rstrip("\n")
            if line.startswith("READY "):
                print(line, flush=True)
                command = sys.stdin.readline()
                if not command.startswith("GEN "):
                    raise RuntimeError("expected the single bounded GEN command")
                engine.stdin.write(command)
                engine.stdin.flush()
                request_sent = True
            elif request_sent and line.startswith("T "):
                fields = line.split(maxsplit=2)
                if len(fields) < 2:
                    raise RuntimeError("malformed generated-token line")
                ids.append(int(fields[1]))
                if not first_token_sent:
                    print("FIRST_TOKEN", flush=True)
                    first_token_sent = True
            elif line.startswith("DONE "):
                print(line, flush=True)
                break
            elif line.startswith("ERR "):
                print("ERR engine reported a request failure", flush=True)
                break

        if not request_sent:
            raise RuntimeError("engine exited before READY")
        if not first_token_sent:
            raise RuntimeError("engine produced no token")
        args.token_ids_output.write_text(json.dumps(ids) + "\n")
        if not sys.stdin.readline().startswith("QUIT"):
            raise RuntimeError("expected QUIT after the bounded request")
        engine.stdin.close()
        code = engine.wait(timeout=30)
        if code != 0:
            raise RuntimeError(f"engine exited with status {code}")
        return 0
    finally:
        if engine.poll() is None:
            engine.terminate()
            try:
                engine.wait(timeout=10)
            except subprocess.TimeoutExpired:
                engine.kill()
                engine.wait()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERR proxy failure: {type(exc).__name__}: {exc}", flush=True)
        raise
