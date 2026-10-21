#!/usr/bin/env python3
"""Run the guarded K35 short adaptive-expert A/B experiment."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
import queue


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
PRIOR = OUT.parent / "2026-10-16-cmp170hx-rtx3090-mmq-640"
SHORT_PROFILE = OUT.parent / "2026-10-16-cmp170hx-rtx3090-short-profile"
LONG_FIXTURE = OUT.parent / "2026-10-16-cmp170hx-rtx3090-gpu-owned-long-context"
PROMPTS = OUT / "prompts"
GIB = 1 << 30
MIB = 1 << 20
HOST_START = 6 * GIB
HOST_STOP = 5 * GIB
GPU_STOPS = {0: 256 * MIB, 1: 640 * MIB}
ENGINE = Path(json.loads((PRIOR / "mmq/results.json").read_text())["argv"][0])
PROFILE_PATH = Path("/home/nikolay/projects/copilot-worktrees/170hx-3090/sibnick-upgraded-succotash/data/expert-profile.bin")
BASELINE = json.loads((PRIOR / "mmq/results.json").read_text())
K37_SHORT_REF = {
    "manifest_sha256": "de3b5214d60af2709d19baf8deed4f1c89e35006e26023882794437011d2ddd6",
    "engine_sha256": "cc8d0ca2ed3364459227147202da3627454ccb85ea1960614c6e241055ef2214",
    "library_sha256": "f0476d2db4fc631d77b2923f35dacdc7dcbb80462e583356c85d773768a12ef4",
    "cache_sha256": "ccc181b3ca2f948cd44f8806179e105a9b29927e48554b99d3e0f1e2554b921c",
}
LONG_PROMPT_SHA = {
    "32k.txt": "bccff648b9ba7d921cce44468f43377e02b6ed683ed1cd5e63c1b9fa2643085a",
    "32k.ids": "48619c77cba7b5b9aff070dd990675453a327c4bf25a8637bfae195ecfc454a7",
    "32k.json": "1809c2e36d8b16afccc7d9bb172c305034a846ef39c7cfbef52ddf4375e19eb6",
}


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


short_runner = load("k35_short_reference", PRIOR / "run-short.py")
base = short_runner.base
profile = short_runner.profile
prepare = profile.prepare


class Nvml:
    class Memory(ctypes.Structure):
        _fields_ = [("total", ctypes.c_ulonglong),
                    ("free", ctypes.c_ulonglong),
                    ("used", ctypes.c_ulonglong)]

    def __init__(self):
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        self.check(self.lib.nvmlInit_v2())
        self.handles = []
        for index in (0, 1):
            handle = ctypes.c_void_p()
            self.check(self.lib.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle)))
            self.handles.append(handle)

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f"NVML error {code}")

    def memory(self):
        result = []
        for index, handle in enumerate(self.handles):
            memory = self.Memory()
            self.check(self.lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)))
            result.append({
                "physical_index": index,
                "total_bytes": memory.total,
                "free_bytes": memory.free,
                "used_bytes": memory.used,
            })
        return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_argv(adaptive: bool) -> list[str]:
    args = list(BASELINE["argv"])
    args[args.index("--layer-split") + 1] = "35"
    args[args.index("--vram-reserve-later-mib") + 1] = "2049"
    args[args.index("--resident-budget-gib") + 1] = "4"
    args.remove("--gpu-owned-experts")
    args[args.index("--resident-budget-gib"):args.index("--resident-budget-gib") + 2] = []
    insert_at = args.index("--mmap-experts") + 1
    args[insert_at:insert_at] = ["--resident-cpu-experts", "--resident-budget-gib", "4"]
    args.extend([
        "--adapt-every", "4" if adaptive else "0",
        "--adapt-swaps", "96" if adaptive else "0",
        "--adapt-async", "0",
    ])
    args[0] = str(ENGINE)
    if "--expert-cache" in args or "--expert-cache-device1" in args:
        raise RuntimeError("fixed expert-cache sizing is forbidden for this experiment")
    if "--gpu-owned-experts" in args:
        raise RuntimeError("K35 must not use --gpu-owned-experts")
    return args


def environment() -> tuple[dict[str, str], dict[str, str]]:
    overrides = dict(BASELINE["environment_overrides"])
    overrides.pop("STRATA_TRACE", None)
    overrides["STRATA_RESIDENT_PIN"] = "1"
    overrides["STRATA_RESIDENT_HEADROOM_GIB"] = "4"
    overrides["STRATA_PIN_RESERVE_GIB"] = "4"
    overrides.pop("STRATA_PREFILL_TIMING", None)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("STRATA_") and key != "CUDA_VISIBLE_DEVICES"}
    env.update(overrides)
    return env, overrides


def validate_binary_options() -> str:
    completed = subprocess.run(
        [str(ENGINE), "--help"], check=True, capture_output=True, text=True, timeout=15
    )
    help_text = completed.stdout + completed.stderr
    for option in (
        "--layer-split", "--resident-cpu-experts",
        "--expert-profile", "--adapt-async",
    ):
        if option not in help_text:
            raise RuntimeError(f"tested binary help does not advertise {option}")
    source = (ROOT / "src/program/generate.cpp").read_text(encoding="utf-8")
    for option in (
        "--vram-reserve-later-mib", "--resident-budget-gib", "--adapt-every",
        "--adapt-swaps", "--adapt-async",
    ):
        if option not in source:
            raise RuntimeError(f"current parser source does not advertise {option}")
    return help_text


def preflight(nvml: Nvml) -> dict:
    if not ENGINE.is_file():
        raise FileNotFoundError(ENGINE)
    if not PROFILE_PATH.is_file():
        raise FileNotFoundError(PROFILE_PATH)
    if base.compute_processes().strip():
        raise RuntimeError("another GPU compute process is active")
    memory = base.read_meminfo()
    gpu = nvml.memory()
    if memory["MemAvailable"] < HOST_START:
        raise RuntimeError("host startup free memory is below 6 GiB")
    if gpu[0]["free_bytes"] < GPU_STOPS[0] or gpu[1]["free_bytes"] < GPU_STOPS[1]:
        raise RuntimeError("physical GPU free-memory startup threshold not met")
    return {"memory": memory, "gpu_memory_bytes": gpu}


def check_pressure(memory: dict, gpus: list[dict], other_compute: list[str]) -> None:
    if memory["MemAvailable"] < HOST_STOP:
        raise RuntimeError("host below 5 GiB reaction threshold (4 GiB floor plus 1 GiB margin)")
    for gpu in gpus:
        threshold = GPU_STOPS[gpu["physical_index"]]
        if gpu["free_bytes"] < threshold:
            raise RuntimeError(
                f"physical GPU {gpu['physical_index']} below "
                f"{threshold // MIB} MiB monitor guard"
            )
    if other_compute:
        raise RuntimeError(f"another GPU compute process appeared: {other_compute}")


def prepare_prompts() -> tuple[list[tuple[dict, list[int]]], tuple[dict, list[int]]]:
    profile.OUT = OUT
    profile.prepare.EXPECTED = {"ORIGIN": "714", "CENTER": "928", "FINAL": "365"}
    profile.prepare.GENERATION_CAP = 32
    short_bundle = profile.prompts()

    PROMPTS.mkdir(exist_ok=True)
    source = LONG_FIXTURE / "prompts"
    long_meta = {}
    for name, expected_hash in LONG_PROMPT_SHA.items():
        source_path = source / name
        target_path = PROMPTS / name
        if sha256(source_path) != expected_hash:
            raise RuntimeError(f"original 32K fixture hash changed: {source_path}")
        if target_path.exists():
            if sha256(target_path) != expected_hash:
                raise RuntimeError(f"refusing to overwrite changed prompt fixture: {target_path}")
        else:
            shutil.copyfile(source_path, target_path)
        long_meta[name] = {
            "source": str(source_path),
            "sha256": expected_hash,
            "bytes": target_path.stat().st_size,
        }
    metadata = json.loads((PROMPTS / "32k.json").read_text(encoding="utf-8"))
    id_text = (PROMPTS / "32k.ids").read_text(encoding="ascii").strip()
    ids = [int(token) for token in id_text.split(",") if token]
    if len(ids) != 32788 or metadata["actual_input_tokens"] != len(ids):
        raise RuntimeError("existing 32K fixture does not contain exactly 32,788 tokens")
    if metadata["generation_cap"] != 192 or len(ids) + 192 > 262144:
        raise RuntimeError("existing 32K fixture does not match its 192-token cap or context")
    if hashlib.sha256((id_text + "\n").encode("ascii")).hexdigest() != metadata["prompt_ids_sha256"]:
        raise RuntimeError("existing 32K prompt token hash does not match metadata")
    references = {
        "short_prompts": {
            meta["name"]: {
                "prompt_ids_sha256": meta["prompt_ids_sha256"],
                "prompt_text_sha256": meta["prompt_text_sha256"],
                "input_tokens": len(ids_),
                "generation_cap": 32,
                "source_bundle": str(PRIOR / "prompts"),
            }
            for meta, ids_ in short_bundle
        },
        "32k_prompt": {
            "source_metadata": str(source / "32k.json"),
            "metadata": metadata,
            "files": long_meta,
        },
    }
    references_path = OUT / "prompt-references.json"
    content = json.dumps(references, indent=2, sort_keys=True) + "\n"
    if references_path.exists() and references_path.read_text() != content:
        raise RuntimeError(f"refusing to overwrite changed prompt references: {references_path}")
    if not references_path.exists():
        references_path.write_text(content)
    return short_bundle, (metadata, ids)


def startup_gate(lines: list[str], ready_gpu: list[dict], argv: list[str]) -> dict:
    joined = "\n".join(lines)
    requirements = (
        "strata: K-quant prompt kernels from",
        "layer split: CUDA0 runs layers 0-34",
        "layer split: CUDA1 runs layers 35-47",
    )
    for requirement in requirements:
        if requirement not in joined:
            raise RuntimeError(f"startup gate missing exact config report: {requirement}")
    reserve_at = argv.index("--vram-reserve-later-mib") + 1
    if argv[reserve_at] != "2049":
        raise RuntimeError("the RTX later-stage reserve is not the approved 2,049 MiB")

    cache_lines = [
        line for line in lines
        if "layer split: CUDA1 runs layers 35-47, expert cache " in line
    ]
    if len(cache_lines) != 1:
        raise RuntimeError("startup gate could not identify RTX AUTO cache slots and bytes")
    match = re.search(
        r"expert cache (\d+) slots \(([\d.]+) GiB\), (\d+) of its (\d+) profiled pairs",
        cache_lines[0],
    )
    if not match:
        raise RuntimeError(f"unexpected RTX cache report: {cache_lines[0]}")
    slots, cache_gib, kept, profiled = int(match[1]), float(match[2]), int(match[3]), int(match[4])
    if not 5300 <= slots <= 5800 or not 15.5 <= cache_gib <= 17.1 or kept <= 0 or profiled <= 0:
        raise RuntimeError(
            f"RTX AUTO cache outside the approved ~16,707 MiB target: "
            f"{slots} slots, {cache_gib:.2f} GiB, {kept}/{profiled} profiled"
        )
    stage_room_lines = [
        line for line in lines
        if "layer split, CUDA1: " in line and "room for experts " in line
    ]
    if len(stage_room_lines) != 1:
        raise RuntimeError("startup did not report the RTX stage's AUTO cache capacity")
    room_match = re.search(
        r"CUDA1: ([\d.]+) GiB free of [\d.]+, room for experts ([\d.]+) GiB",
        stage_room_lines[0],
    )
    if not room_match or abs(float(room_match[2]) - cache_gib) > 0.02:
        raise RuntimeError("RTX AUTO cache allocation does not match its reported stage capacity")
    primary_cache = [
        line for line in lines
        if "expert cache " in line and " slots, " in line and " GiB of VRAM; policy is" in line
    ]
    profile_lines = [line for line in lines if "ranked pairs, built for " in line]
    if len(primary_cache) != 1 or len(profile_lines) != 1:
        raise RuntimeError("startup did not report complete primary-cache/profile ownership")
    primary_match = re.search(r"expert cache (\d+) slots", primary_cache[0])
    profile_match = re.search(r"profile .*: (\d+) ranked pairs, built for (\d+) slots", profile_lines[0])
    if not primary_match or not profile_match or int(profile_match[1]) != 24576:
        raise RuntimeError("profile/cache owner counts are incomplete")
    expected_cold = int(profile_match[1]) - int(primary_match[1]) - slots

    budget_lines = [line for line in lines if "RAM budget 4.00 GiB:" in line]
    complement = [
        line for line in lines if "mapped pinned cache complement ready:" in line
    ]
    if len(budget_lines) != 1 or len(complement) != 1:
        raise RuntimeError("startup did not confirm a complete budgeted, page-locked RAM complement")
    budget_match = re.search(
        r"RAM budget 4\.00 GiB: (\d+) of the ([\d.]+) GiB of experts the GPU cache does not hold",
        budget_lines[0],
    )
    complement_match = re.search(
        r"resident ([\d.]+) GiB, pinned ([\d.]+) GiB",
        complement[0],
    )
    if not budget_match or not complement_match:
        raise RuntimeError("unrecognized resident-complement accounting")
    cold_pairs = int(budget_match[1])
    unowned_gib = float(budget_match[2])
    resident_gib, pinned_gib = float(complement_match[1]), float(complement_match[2])
    if cold_pairs != expected_cold or not 2.8 <= unowned_gib <= 3.6:
        raise RuntimeError(
            f"cold complement does not cover every unowned profile pair: "
            f"{cold_pairs} pairs / {unowned_gib:.2f} GiB"
        )
    if not 2.8 <= resident_gib <= 3.6 or abs(resident_gib - pinned_gib) > 0.01:
        raise RuntimeError(
            f"resident complement is not fully page-locked near 3.3 GiB: "
            f"resident={resident_gib:.2f}, pinned={pinned_gib:.2f} GiB"
        )
    if abs(unowned_gib - resident_gib) > 0.02:
        raise RuntimeError(
            "the 4 GiB resident budget leaves a file-backed expert remainder"
        )
    warnings = [
        line for line in lines if "WARNING:" in line and any(
            term in line.lower()
            for term in ("resident", "budget", "complement", "page-lock", "mmap fallback")
        )
    ]
    if warnings:
        raise RuntimeError(f"startup emitted a warning before inference: {warnings}")

    rtx = ready_gpu[0]["free_bytes"]  # physical index 0
    cmp = ready_gpu[1]["free_bytes"]  # physical index 1
    if rtx < GPU_STOPS[0] or cmp < GPU_STOPS[1]:
        raise RuntimeError("startup-ready physical free VRAM is below the hard guard")
    return {
        "stage1_cache": {
            "physical_device": "RTX 3090 index 0",
            "slots": slots,
            "reported_gib": cache_gib,
            "resident_pairs": kept,
            "profiled_pairs": profiled,
            "reported_stage_room_gib": float(room_match[2]),
            "configured_later_reserve_mib": int(argv[reserve_at]),
        },
        "cold_complement": {
            "reported_unowned_pairs": cold_pairs,
            "expected_unowned_pairs_from_profile_owners": expected_cold,
            "reported_unowned_gib": unowned_gib,
            "resident_gib": resident_gib,
            "pinned_gib": pinned_gib,
            "fully_page_locked": True,
        },
        "startup_ready_gpu_free_bytes": ready_gpu,
    }


def validate_done(line: str, ids: list[int], generated: list[int], tokenizer,
                  expected: dict[str, str], cap: int) -> dict:
    fields = line.split()
    if len(fields) < 16:
        raise ValueError(f"unexpected DONE: {line}")
    if int(fields[1]) != len(generated) or int(fields[2]) != len(ids):
        raise ValueError("DONE token count mismatch")
    if int(fields[8]) != 0 or int(fields[14]) != len(ids):
        raise ValueError("request did not fully prefill without prefix reuse")
    if fields[5] != "stop" or len(generated) >= cap:
        raise ValueError("answer did not complete below its generation cap")
    answer = tokenizer.decode(
        generated[:-1] if generated and generated[-1] in {248044, 248046} else generated
    )
    wanted = "\n".join(f"{key}: {value}" for key, value in expected.items())
    if answer.strip() != wanted:
        raise ValueError(f"retrieval failed: {answer!r} != {wanted!r}")
    return {
        "done": line,
        "answer": answer,
        "generated_ids": generated,
        "input_tokens": len(ids),
        "output_tokens": len(generated),
        "prefill_ms": float(fields[3]),
        "decode_ms": float(fields[4]),
        "mtp_accepted_offered": [int(fields[6]), int(fields[7])],
        "reused": int(fields[8]),
        "fresh": int(fields[14]),
    }


def result_path(mode: str) -> Path:
    return OUT / mode / "results.json"


def run(mode: str) -> None:
    adaptive = mode == "adaptive"
    if mode not in ("control", "adaptive"):
        raise ValueError(f"unknown arm {mode}")
    if adaptive:
        control = json.loads(result_path("control").read_text())
        if control.get("status") != "passed" or len(control.get("requests", [])) != 10:
            raise RuntimeError("adaptive arm requires a fully passed 10-request control arm")
    nvml = Nvml()
    before = preflight(nvml)
    help_text = validate_binary_options()
    flags = {token for token in make_argv(adaptive) if token.startswith("--")}
    required_flags = {
        "--pack", "--native", "--gpu", "--layer-split", "--trim-stage-weights",
        "--max-context", "--kv", "--mtp", "--spec", "--prefill",
        "--no-prefill-borrow", "--mmap-experts", "--resident-cpu-experts",
        "--resident-budget-gib", "--expert-profile", "--ple-io", "--ple-row-cache",
        "--vram-reserve-mib", "--vram-reserve-later-mib", "--pool-workers", "--serve",
        "--prompt-cache", "--conversation-cache-min-free-mib", "--adapt-every",
        "--adapt-swaps", "--adapt-async",
    }
    if required_flags - flags:
        raise RuntimeError(f"constructed argv is missing flags: {sorted(required_flags - flags)}")
    if "--gpu-owned-experts" in flags or "--expert-cache" in flags:
        raise RuntimeError("constructed argv violates the partial AUTO-cache experiment shape")
    if "--adapt-async" not in help_text:
        raise RuntimeError("tested binary does not advertise --adapt-async")
    short_bundle, long_prompt = prepare_prompts()
    if mode == "control" and result_path("control").parent.exists():
        raise FileExistsError(result_path("control").parent)
    out = result_path(mode).parent
    out.mkdir(parents=False, exist_ok=False)

    argv = make_argv(adaptive)
    env, overrides = environment()
    result = {
        "status": "starting",
        "arm": mode,
        "scheduler": {
            "adapt_every": 4 if adaptive else 0,
            "adapt_swaps": 96 if adaptive else 0,
            "adapt_async": 0,
        },
        "argv": argv,
        "environment_overrides": overrides,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "engine_sha256": base.sha256(ENGINE),
        "kquant_library_sha256": base.sha256(ENGINE.parent / "libstrata_kq.so"),
        "cache_sha256": base.sha256(ENGINE.parent / "CMakeCache.txt"),
        "prior_artifact_identity": base.prior_artifact_identity(),
        "prior_artifact_manifest_sha256": K37_SHORT_REF["manifest_sha256"],
        "guards": {
            "gpu_stop_mib_physical": {"0": 256, "1": 640},
            "gpu_allocation_reserve_mib": 1024,
            "gpu_later_allocation_reserve_mib": 2049,
            "host_start_gib": 6,
            "host_stop_gib": 5,
            "host_floor_gib": 4,
            "reaction_headroom_gib": 1,
            "poll_seconds": 1,
        },
        "resource_preflight": before,
        "started_utc": base.utc_now(),
        "requests": [],
        "startup_gate": None,
    }
    if result["engine_sha256"] != K37_SHORT_REF["engine_sha256"]:
        raise RuntimeError("tested engine binary hash changed")
    if result["kquant_library_sha256"] != K37_SHORT_REF["library_sha256"]:
        raise RuntimeError("tested MMQ library hash changed")
    if result["cache_sha256"] != K37_SHORT_REF["cache_sha256"]:
        raise RuntimeError("tested CMake cache hash changed")
    if result["prior_artifact_identity"]["manifest_sha256"] != K37_SHORT_REF["manifest_sha256"]:
        raise RuntimeError("preserved artifact manifest hash changed")
    if "STRATA_TRACE" in overrides or "STRATA_PREFILL_TIMING" in overrides:
        raise RuntimeError("profiling/timing instrumentation must remain disabled")

    def save():
        result_path(mode).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    save()
    started = time.monotonic()
    phase = "startup"
    last_sample = -10.0
    lines: queue.Queue[str | None] = queue.Queue()
    process = None
    reader_thread = None
    tokenizer = prepare.load_tokenizer()
    with (out / "engine.log").open("w", buffering=1) as log, \
            (out / "monitor.jsonl").open("w", buffering=1) as monitor:
        def sample(force: bool = False) -> None:
            nonlocal last_sample
            now = time.monotonic()
            if not force and now - last_sample < 1.0:
                return
            last_sample = now
            memory = base.read_meminfo()
            gpus = nvml.memory()
            other_compute = [
                row for row in base.compute_processes().splitlines()
                if int(row.split(",")[0].strip()) != process.pid
            ]
            monitor.write(json.dumps({
                "utc": base.utc_now(),
                "elapsed_seconds": now - started,
                "phase": phase,
                "memory": memory,
                "gpu_memory_bytes": gpus,
                "gpu_state": base.gpu_state(),
                "process": base.process_state(process.pid),
                "other_compute": other_compute,
            }, sort_keys=True) + "\n")
            check_pressure(memory, gpus, other_compute)

        def read_until(predicate, timeout: float) -> list[str]:
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
            process = subprocess.Popen(
                argv, cwd=ROOT, env=env, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )

            def reader():
                for line in process.stdout:
                    lines.put(line.rstrip("\n"))
                lines.put(None)

            reader_thread = threading.Thread(target=reader, name="k35-engine-reader", daemon=True)
            reader_thread.start()
            print(f"{base.utc_now()} START K35 {mode} pid={process.pid}", flush=True)
            startup = read_until(lambda line: line == "READY 262144 stop", 1200)
            result["startup_seconds"] = time.monotonic() - started
            result["startup_gate"] = startup_gate(startup, nvml.memory(), argv)
            result["startup_lines"] = startup
            result["status"] = "running"
            result["ready_memory"] = {
                "host": base.read_meminfo(),
                "gpu_memory_bytes": nvml.memory(),
            }
            save()
            print(
                f"{base.utc_now()} READY K35 {mode} "
                f"RTX-cache={result['startup_gate']['stage1_cache']['reported_gib']:.2f}GiB "
                f"pinned-complement={result['startup_gate']['cold_complement']['resident_gib']:.2f}GiB",
                flush=True,
            )

            schedule = [("warmup", short_bundle[-1], 32, {"ORIGIN": "714", "CENTER": "928", "FINAL": "365"})]
            schedule.extend(
                (f"rep{rep}-{metadata['name']}", (metadata, ids), 32,
                 {"ORIGIN": "714", "CENTER": "928", "FINAL": "365"})
                for rep in range(1, 4) for metadata, ids in short_bundle
            )
            if adaptive:
                prepare.EXPECTED = {
                    "ORIGIN": "IRIS-4Q7-LANTERN",
                    "CENTER": "CEDAR-8M2-QUARRY",
                    "FINAL": "ORBIT-6V9-KETTLE",
                }
                schedule.append(("sanity-32k", long_prompt, 192, prepare.EXPECTED.copy()))

            for label, (metadata, ids), cap, expected in schedule:
                phase = label
                request_start = time.monotonic()
                first_token = []
                command = "GEN " + str(cap) + " " + ",".join(map(str, ids)) + "\n"
                write_error = []

                def write_request():
                    try:
                        process.stdin.write(command)
                        process.stdin.flush()
                    except BaseException as exc:
                        write_error.append(exc)

                writer_thread = threading.Thread(
                    target=write_request, name=f"k35-request-{label}", daemon=True
                )
                writer_thread.start()
                captured = read_until(lambda line: line.startswith("DONE "), 1200 if cap > 32 else 180)
                writer_thread.join(timeout=15)
                if writer_thread.is_alive():
                    raise TimeoutError(f"{label}: request pipe write did not finish")
                if write_error:
                    raise RuntimeError(f"{label}: request write failed: {write_error[0]}")
                for line in captured:
                    if line.startswith("T ") and not first_token:
                        first_token.append(time.monotonic() - request_start)
                generated = [int(line.split()[1]) for line in captured if line.startswith("T ")]
                request = validate_done(
                    captured[-1], ids, generated, tokenizer, expected, cap
                )
                counter_lines = read_until(
                    lambda line: "file tier I/O this request:" in line, 120
                )
                if label not in ("warmup",) and not counter_lines[-1].endswith(
                    "for 0.0 MB of expert reads"
                ):
                    raise RuntimeError(f"{label}: file-tier expert read delta was not zero")
                diagnostics = captured + counter_lines
                request.update({
                    "name": label,
                    "wall_seconds_to_done": time.monotonic() - request_start,
                    "ttft_seconds_observed": first_token[0] if first_token else None,
                    "prompt_ids_sha256": metadata["prompt_ids_sha256"],
                    "generation_cap": cap,
                    "counter_lines": counter_lines,
                    "adaptive_diagnostics": [
                        line for line in diagnostics
                        if any(token in line.lower() for token in (
                            "adaptive tier", "expert tiers:", "cpu experts",
                            "decode expert cache hit rate", "resident ram:",
                            "accepted per round", "verify window",
                            "file tier i/o this request:",
                        ))
                    ],
                    "memory_after": nvml.memory(),
                })
                result["requests"].append(request)
                sample(force=True)
                save()
                print(
                    f"{base.utc_now()} PASS K35 {mode} {label} "
                    f"input={request['input_tokens']} prefill={request['prefill_ms']:.1f}ms "
                    f"wall={request['wall_seconds_to_done']:.3f}s",
                    flush=True,
                )

            result["status"] = "passed"
        except BaseException as exc:
            result["status"] = "failed"
            result["error"] = f"{type(exc).__name__}: {exc}"
            save()
            raise
        finally:
            if process is not None:
                if result["status"] != "passed" and process.poll() is None:
                    base.safe_stop(process, result.get("error", "K35 run failed"))
                elif process.poll() is None:
                    process.stdin.close()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        base.safe_stop(process, "K35 EOF shutdown timeout")
                if reader_thread is not None:
                    reader_thread.join(timeout=5)
                while not lines.empty():
                    line = lines.get_nowait()
                    if line is not None:
                        log.write(line + "\n")
                result["engine_return_code"] = process.returncode
            result["finished_utc"] = base.utc_now()
            result["elapsed_seconds"] = time.monotonic() - started
            result["after_memory"] = {
                "host": base.read_meminfo(),
                "gpu_memory_bytes": nvml.memory(),
            }
            result["compute_processes_after"] = base.compute_processes()
            save()
    if result["status"] != "passed":
        raise RuntimeError(f"K35 {mode} arm did not pass")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("control", "adaptive"), required=True)
    run(parser.parse_args().mode)
