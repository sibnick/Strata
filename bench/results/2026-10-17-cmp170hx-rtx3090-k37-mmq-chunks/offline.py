#!/usr/bin/env python3
"""Link only the existing sizing program against the unchanged validated libraries."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
BUILD = Path("/home/nikolay/projects/copilot-worktrees/170hx-3090/"
             "sibnick-animated-robot/build/cuda-batched")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    reference = json.loads((OUT.parent / "2026-10-17-cmp170hx-rtx3090-k37-mmq-batched-long/on/results.json").read_text())
    expected = json.loads((OUT.parent / "2026-10-17-cmp170hx-rtx3090-k37-mmq-batched-gather/build-identity.json").read_text())
    for relative, digest in expected["candidate_hashes"].items():
        path = BUILD / Path(relative).name if relative.startswith("build/") else ROOT / relative
        if sha(path) != digest:
            raise RuntimeError(f"validated source/build mismatch: {path}")
    subprocess.run(["git", "merge-base", "--is-ancestor", "97b83332", "HEAD"], check=True)
    subprocess.run(["git", "merge-base", "--is-ancestor", "2fcdbd59", "HEAD"], check=True)
    if subprocess.check_output(["git", "diff", "97b83332", "--", "src", "include", "CMakeLists.txt"]):
        raise RuntimeError("runtime differs from validated batching source")
    scratch = ROOT / "build/chunk-sizing"
    scratch.mkdir(parents=True, exist_ok=True)
    dso = scratch / "libstrata_kq.so"
    if not dso.exists():
        dso.symlink_to(BUILD / "libstrata_kq.so")
    obj, exe = scratch / "ledger.o", scratch / "ledger"
    compile_cmd = ["/usr/bin/c++", "-std=c++20", "-O2", "-DSTRATA_GPU_CUDA",
                   "-I" + str(ROOT / "include"),
                   "-I/usr/local/cuda-13.4/include", "-c",
                   str(ROOT / "src/program/split_memory_ledger.cpp"), "-o", str(obj)]
    link_cmd = shlex.split((BUILD / "CMakeFiles/strata.dir/link.txt").read_text())
    link_cmd[link_cmd.index("CMakeFiles/strata.dir/src/program/generate.cpp.o")] = str(obj)
    link_cmd[link_cmd.index("-o") + 1] = str(exe)
    commands = [compile_cmd, link_cmd]
    with (OUT / "sizing-build.log").open("w") as log:
        for cmd in commands:
            subprocess.run(cmd, cwd=BUILD, stdout=log, stderr=subprocess.STDOUT, check=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("STRATA_")
           and k not in ("CUDA_VISIBLE_DEVICES", "LD_PRELOAD")}
    env.update(reference["environment_overrides"])
    argv = reference["argv"]
    rows = {}
    for chunk in (256, 384, 512):
        cmd = [str(exe), "--pack", argv[argv.index("--pack") + 1],
               "--native", argv[argv.index("--native") + 1],
               "--mtp", argv[argv.index("--mtp") + 1], "--context", "262144", "--chunk", str(chunk)]
        commands.append(cmd)
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
        if "K-quant prompt kernels from" not in p.stderr or "FP16 prompt path" in p.stderr:
            raise RuntimeError("sizing must use the actual K-quant DSO, not FP16 fallback")
        (OUT / f"ledger-{chunk}.csv").write_text(p.stdout)
        (OUT / f"ledger-{chunk}.log").write_text(p.stderr)
        rows[str(chunk)] = {
            stage: {line.split(",")[2]: int(line.split(",")[3]) for line in p.stdout.splitlines()
                    if line.startswith(f"37,{stage},")} for stage in (0, 1)}
    data = {"commands": commands, "environment_overrides": reference["environment_overrides"],
            "build": str(BUILD), "sizing_executable_sha256": sha(exe), "k37": rows,
            "source_sha256": {p: sha(ROOT / p) for p in (
                "src/program/split_memory_ledger.cpp", "src/prefill/prefill.cpp",
                "src/program/generate.cpp", "src/core/session.cpp")}}
    (OUT / "offline.json").write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
