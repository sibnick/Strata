#!/usr/bin/env python3
"""Guarded <=2K profiling using the preserved model/config, never long prompts."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
PRIOR = OUT.parent / "2026-10-16-cmp170hx-rtx3090-gpu-owned-long-context"
ENGINE = ROOT / "build/cuda-short-profile/strata"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load("guarded_long_context", PRIOR / "run-long-context.py")
prepare = load("prepare_long_context", PRIOR / "prepare-long-context.py")


def prompts() -> list[tuple[dict, list[int]]]:
    prepare.EXPECTED = {"ORIGIN": "714", "CENTER": "928", "FINAL": "365"}
    prepare.GENERATION_CAP = 32
    tokenizer = prepare.load_tokenizer()
    bundle = []
    for name, target, seed in (("small", 448, 83), ("medium", 960, 97), ("large", 1920, 109)):
        body, ids, metadata = prepare.make_prompt(tokenizer, name, target, seed)
        if len(ids) > 2048:
            raise ValueError(f"{name} input exceeds 2K: {len(ids)}")
        folder = OUT / "prompts"
        folder.mkdir(exist_ok=True)
        for suffix, content in (
            (".txt", body + "\n"),
            (".ids", ",".join(map(str, ids)) + "\n"),
            (".json", json.dumps(metadata, indent=2, sort_keys=True) + "\n"),
        ):
            path = folder / (name + suffix)
            if path.exists() and path.read_text() != content:
                raise RuntimeError(f"refusing to change existing prompt {path}")
            if not path.exists():
                path.write_text(content)
        bundle.append((metadata, ids))
    return bundle


def validate_done(line: str, ids: list[int], generated: list[int], tokenizer) -> dict:
    fields = line.split()
    if len(fields) < 16:
        raise ValueError(f"unexpected DONE: {line}")
    if int(fields[1]) != len(generated) or int(fields[2]) != len(ids):
        raise ValueError("DONE token count mismatch")
    if int(fields[8]) != 0 or int(fields[14]) != len(ids):
        raise ValueError("request did not fully prefill without reuse")
    if fields[5] != "stop" or len(generated) >= 32:
        raise ValueError("answer did not complete within 32-token cap")
    answer = tokenizer.decode(generated[:-1] if generated and generated[-1] in {248044, 248046} else generated)
    expected = "\n".join(f"{key}: {value}" for key, value in prepare.EXPECTED.items())
    if answer.strip() != expected:
        raise ValueError(f"retrieval failed: {answer!r} != {expected!r}")
    return {
        "done": line, "answer": answer, "generated_ids": generated,
        "input_tokens": len(ids), "output_tokens": len(generated),
        "prefill_ms": float(fields[3]), "decode_ms": float(fields[4]),
        "prefill_tokens_per_second": len(ids) * 1000 / float(fields[3]),
        "reused": int(fields[8]), "fresh": int(fields[14]),
    }


def run(mode: str) -> None:
    bundle = prompts()
    if not ENGINE.is_file():
        raise FileNotFoundError(ENGINE)
    if base.compute_processes():
        raise RuntimeError("another GPU compute process is active")
    if base.read_meminfo()["MemAvailable"] < 6 * base.GIB:
        raise RuntimeError("host start floor 6 GiB not met")
    if any(g["memory_free_mib"] < 1024 for g in base.gpu_state()):
        raise RuntimeError("GPU start guard 1024 MiB not met")
    out = OUT / mode
    out.mkdir(exist_ok=False)
    baseline = json.loads((PRIOR / "main-results.json").read_text())
    argv = baseline["arguments"].copy()
    argv[0] = str(ENGINE)
    env = {k: v for k, v in os.environ.items() if not k.startswith("STRATA_") and k != "CUDA_VISIBLE_DEVICES"}
    overrides = baseline["environment_policy"]["STRATA_overrides"].copy()
    if mode.endswith("profile"):
        overrides["STRATA_PREFILL_TIMING"] = "1"
    if mode.startswith("mmq-"):
        overrides["STRATA_MMQ_KQ"] = "1"
    env.update(overrides)
    result = {
        "status": "starting", "argv": argv, "environment_overrides": overrides,
        "engine_sha256": base.sha256(ENGINE),
        "cache_sha256": base.sha256(ENGINE.parent / "CMakeCache.txt"),
        "kquant_library_sha256": base.sha256(ENGINE.parent / "libstrata_kq.so"),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "artifact_identity": base.prior_artifact_identity(),
        "guards": {"host_start_gib": 6, "host_stop_gib": 5, "gpu_stop_mib": 1024, "poll_seconds": 1},
        "started_utc": base.utc_now(), "requests": [],
        "note": "rebuilt unchanged runtime source; not the lost original /tmp binary",
    }
    def save():
        (out / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    save()
    started = time.monotonic()
    process = None
    phase = "startup"
    last_sample = -10.0
    lines = queue.Queue()
    tokenizer = prepare.load_tokenizer()
    with (out / "engine.log").open("w", buffering=1) as log, (out / "monitor.jsonl").open("w", buffering=1) as monitor:
        def sample():
            nonlocal last_sample
            now = time.monotonic()
            if now - last_sample < 1:
                return
            last_sample = now
            memory = base.read_meminfo()
            gpus = base.gpu_state()
            others = []
            for row in base.compute_processes().splitlines():
                if int(row.split(",")[0].strip()) != process.pid:
                    others.append(row)
            record = {
                "utc": base.utc_now(), "elapsed_seconds": now - started, "phase": phase,
                "memory": memory, "gpus": gpus, "process": base.process_state(process.pid),
                "other_compute": others,
            }
            monitor.write(json.dumps(record, sort_keys=True) + "\n")
            if memory["MemAvailable"] < 5 * base.GIB:
                raise RuntimeError("host below 5 GiB reaction threshold")
            if any(g["memory_free_mib"] < 1024 for g in gpus):
                raise RuntimeError("GPU below 1024 MiB stop guard")
            if others:
                raise RuntimeError(f"other compute process appeared: {others}")

        def read_until(predicate, timeout):
            deadline = time.monotonic() + timeout
            captured = []
            while time.monotonic() < deadline:
                sample()
                try:
                    line = lines.get(timeout=0.1)
                except queue.Empty:
                    continue
                if line is None:
                    raise RuntimeError("engine exited unexpectedly")
                log.write(line + "\n")
                captured.append(line)
                if line.startswith("ERR "):
                    raise RuntimeError(line)
                if predicate(line):
                    return captured
            raise TimeoutError(f"{phase} exceeded {timeout}s")

        try:
            process = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            def reader():
                for line in process.stdout:
                    lines.put(line.rstrip("\n"))
                lines.put(None)
            thread = threading.Thread(target=reader, daemon=True)
            thread.start()
            print(f"{base.utc_now()} START mode={mode} pid={process.pid}", flush=True)
            read_until(lambda line: line == "READY 262144 stop", 1200)
            result["startup_seconds"] = time.monotonic() - started
            result["status"] = "running"
            save()
            print(f"{base.utc_now()} READY mode={mode} startup={result['startup_seconds']:.2f}s", flush=True)
            schedule = [("warmup", bundle[-1])] + [
                (f"rep{rep}-{meta['name']}", (meta, ids))
                for rep in range(1, 4) for meta, ids in bundle
            ]
            for label, (meta, ids) in schedule:
                phase = label
                request_start = time.monotonic()
                process.stdin.write("GEN 32 " + ",".join(map(str, ids)) + "\n")
                process.stdin.flush()
                captured = read_until(lambda line: line.startswith("DONE "), 120)
                generated = [int(line.split()[1]) for line in captured if line.startswith("T ")]
                data = validate_done(captured[-1], ids, generated, tokenizer)
                data.update({
                    "name": label, "wall_seconds_to_done": time.monotonic() - request_start,
                    "prompt_ids_sha256": meta["prompt_ids_sha256"],
                    "profile_lines": [line for line in captured if line.startswith("strata prefill timing:")],
                })
                result["requests"].append(data)
                save()
                print(f"{base.utc_now()} PASS {label} input={len(ids)} prefill={data['prefill_ms']}ms", flush=True)
            result["status"] = "passed"
        except BaseException as exc:
            result["status"] = "failed"
            result["error"] = f"{type(exc).__name__}: {exc}"
            save()
            raise
        finally:
            if process is not None:
                if result["status"] != "passed" and process.poll() is None:
                    base.safe_stop(process, result.get("error", "profile failed"))
                elif process.poll() is None:
                    process.stdin.close()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        base.safe_stop(process, "EOF shutdown timeout")
                while not lines.empty():
                    line = lines.get_nowait()
                    if line is not None:
                        log.write(line + "\n")
                result["engine_return_code"] = process.returncode
                if process.returncode != 0:
                    result["status"] = "failed"
            result["finished_utc"] = base.utc_now()
            result["elapsed_seconds"] = time.monotonic() - started
            save()
    if result["status"] != "passed":
        raise RuntimeError("short profile did not pass")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("control", "profile", "mmq-control", "mmq-profile"), required=True)
    args = parser.parse_args()
    run(args.mode)
