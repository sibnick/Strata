#!/usr/bin/env python3
"""CPU/source/resource audit only: no context, kernel launch, rebuild or tile changes."""
import ctypes
import hashlib
import json
from pathlib import Path
import re
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
BUILD = Path(json.loads((OUT / "offline.json").read_text())["build"])
DEP = BUILD.parent / "llama-pristine"
CUDA = DEP / "ggml/src/ggml-cuda"


def main():
    driver = ctypes.CDLL("libcuda.so.1")
    if driver.cuInit(0):
        raise RuntimeError("CUDA driver query failed")
    devices = []
    for index in (0, 1):
        device = ctypes.c_int()
        if driver.cuDeviceGet(ctypes.byref(device), index):
            raise RuntimeError("CUDA device query failed")
        info = {"physical_index": index}
        for key, attr in (("sm_count", 16), ("registers_per_sm", 82),
                          ("shared_bytes_per_sm", 81), ("shared_optin_per_block", 97),
                          ("threads_per_sm", 39), ("cc_major", 75), ("cc_minor", 76)):
            value = ctypes.c_int()
            if driver.cuDeviceGetAttribute(ctypes.byref(value), attr, device):
                raise RuntimeError(f"driver attribute {attr} failed")
            info[key] = value.value
        devices.append(info)
    config_text = (CUDA / "mmq-config-ampere.cuh").read_text()
    configs = []
    for match in re.finditer(
        r"CASE\(GGML_TYPE_(Q[456]_K),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*"
        r"GGML_CUDA_MMQ_SRAM_LAYOUT_(\w+),\s*MMQ_ITER_K,\s*(true|false),\s*false\)", config_text
    ):
        quant, threads, occupancy, i, j, layout, stream = match.groups()
        configs.append({"type": quant, "threads": int(threads), "target_occupancy": int(occupancy),
                        "I": int(i), "J": int(j), "sram_layout": layout, "K": 256,
                        "stream_k": stream == "true", "fallback": False})
    raw = (OUT / "kq-resources.txt").read_text()
    resources, arch = [], None
    lines = raw.splitlines()
    for n, line in enumerate(lines):
        if line.startswith("arch = "):
            arch = line.split()[-1]
        match = re.search(r"Function _Z9mul_mat_qIL9ggml_type(12|13|14)ELi(\d+)ELb0E", line)
        if match:
            fields = {k: int(v) for k, v in re.findall(r"(REG|STACK|SHARED|LOCAL):(\d+)", lines[n + 1])}
            resources.append({"arch": arch, "type": {"12": "Q4_K", "13": "Q5_K", "14": "Q6_K"}[match[1]],
                              "J": int(match[2]), **fields})
    table = []
    for config in configs:
        # Both relevant layouts have 76 int32 words per I row in this pinned source.
        j = config["J"]
        dynamic_shared = config["I"] * 76 * 4 + j * 4 + ((j * 144 + 1023) // 1024) * 1024
        for device in devices:
            arch = f"sm_{device['cc_major']}{device['cc_minor']}"
            resource = next(r for r in resources if (r["arch"], r["type"], r["J"]) ==
                            (arch, config["type"], j))
            # Register allocation granularity and other limits can reduce this upper bound.
            upper_bound = min(device["threads_per_sm"] // config["threads"],
                              device["registers_per_sm"] // (resource["REG"] * config["threads"]),
                              device["shared_bytes_per_sm"] // dynamic_shared)
            table.append({**config, **resource, "physical_gpu": device["physical_index"],
                          "dynamic_shared_bytes": dynamic_shared,
                          "fits_shared_per_block": dynamic_shared <= device["shared_optin_per_block"],
                          "resident_cta_upper_bound": upper_bound})
    selection = {}
    for rows in range(1, 513):
        best_j, best_tiles = 0, 1 << 30
        for c in configs:
            if c["type"] != "Q4_K":
                continue
            tiles = (rows + c["J"] - 1) // c["J"]
            if tiles < best_tiles:
                best_j, best_tiles = c["J"], tiles
            if best_tiles == 1:
                break
        selection[rows] = best_j
    files = [CUDA / "mmq.cuh", CUDA / "mmq-config-ampere.cuh",
             DEP / "ggml/src/ggml-common.h", ROOT / "src/prefill/moe_mmq.cu",
             ROOT / "src/prefill/mmq_kq_lib.cu", ROOT / "src/prefill/prefill.cpp"]
    result = {
        "method": "read-only driver attributes (no context), pinned source, unchanged DSO cuobjdump resource report",
        "dependency_commit": subprocess.check_output(["git", "-C", str(DEP), "rev-parse", "HEAD"], text=True).strip(),
        "dependency_diff": subprocess.check_output(["git", "-C", str(DEP), "diff"], text=True),
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "devices": devices, "configs_and_resources": table,
        "selected_J_for_group_max_rows_1_through_512": selection,
        "observed_group_row_histogram": None,
        "limits": "Selection is exact conditional on max_rows, not observed launch frequencies. "
                  "Resource CTA numbers are upper bounds, not achieved occupancy. STACK is not proof of spills. "
                  "No bottleneck or concrete tile speedup is established; do not change tiles from this evidence.",
    }
    (OUT / "tile-audit.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
