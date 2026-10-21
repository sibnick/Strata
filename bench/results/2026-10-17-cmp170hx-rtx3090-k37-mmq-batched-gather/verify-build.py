#!/usr/bin/env python3
"""Compare every K-quant SASS function/instruction; omit only cuobjdump's container metadata."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
libraries = {
    "preserved": Path("/home/nikolay/projects/copilot-worktrees/170hx-3090/"
                      "sibnick-fuzzy-spoon/build/cuda-short-profile/libstrata_kq.so"),
    "clean": ROOT / "build/clean-unpatched/libstrata_kq.so",
    "candidate": ROOT / "build/cuda-batched/libstrata_kq.so",
}
result = {}
for name, path in libraries.items():
    dump = subprocess.check_output(
        ["/usr/local/cuda-13.4/bin/cuobjdump", "--dump-sass", str(path)], text=True)
    lines = [s for s in dump.splitlines() if s.strip() and not s.startswith(
        ("code for ", "Fatbin", "arch =", "identifier =", "host =", "compile_size ="))]
    result[name] = {
        "path": str(path), "dso_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sass_normalized_sha256": hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest(),
        "sass_lines": len(lines),
    }
if len({r["sass_normalized_sha256"] for r in result.values()}) != 1:
    raise RuntimeError("compiled K-quant instruction mismatch")
print(json.dumps(result, indent=2))
