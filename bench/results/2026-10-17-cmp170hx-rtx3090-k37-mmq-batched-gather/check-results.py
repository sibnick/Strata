#!/usr/bin/env python3
"""Fail closed on incomplete arms, correctness, identity, safety or mechanism evidence."""
import importlib.util
import json
from pathlib import Path
import re

OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("gather_checks", OUT / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)
expected = {"preserved": 4, "off-large": 4, "on-large": 4, "on-short": 7,
            "off-short": 7, "counts-off": 2, "counts-on": 2}
arms = {name: json.loads((OUT / name / "results.json").read_text()) for name in expected}
fixtures = {meta["prompt_ids_sha256"]: (ids, ref) for meta, ids, ref in
            run.bundle(["small", "medium", "large"])}
candidate = arms["off-large"]
base_env = dict(candidate["environment_overrides"])
base_env.pop("STRATA_PREFILL_BATCH_GATHER")
for name, arm in arms.items():
    assert arm["status"] == "passed" and len(arm["requests"]) == expected[name], name
    assert arm["engine_return_code"] == 0 and not arm["compute_processes_after"], name
    env = dict(arm["environment_overrides"])
    env.pop("STRATA_PREFILL_BATCH_GATHER")
    env.pop("STRATA_PREFILL_GATHER_COUNTS", None)
    assert env == base_env, name
    assert "STRATA_TRACE" not in env and "STRATA_PREFILL_TIMING" not in env
    if name != "preserved":
        for key in ("argv", "engine_sha256", "kquant_library_sha256", "cache_sha256"):
            assert arm[key] == candidate[key], (name, key)
    for request in arm["requests"]:
        ids, ref = fixtures[request["prompt_ids_sha256"]]
        run.validate_done(request["done"], ids, request["generated_ids"], ref)
        if request["name"] != "warmup":
            assert request["counter_lines"][-1].endswith("for 0.0 MB of expert reads"), name
        assert bool(request["gather_counts"]) == name.startswith("counts-"), name
    for line in (OUT / name / "monitor.jsonl").read_text().splitlines():
        sample = json.loads(line)
        run.check_pressure(sample["memory"], sample["gpu_memory_bytes"], sample["other_compute"])
        assert sample["process"].get("VmSwap", 0) == 0, name
    startup = (OUT / name / "engine.log").read_text().splitlines()
    run.startup_gate(startup[:startup.index("READY 262144 stop") + 1], arm["argv"], Path(arm["argv"][0]))
count_re = re.compile(r"device=(\d+).* launches=(\d+) experts=(\d+) bytes=(\d+) fallbacks=(\d+)")


def counts(request):
    totals = {}
    for line in request["gather_counts"]:
        match = count_re.search(line)
        assert match, line
        device, *values = map(int, match.groups())
        dest = totals.setdefault(device, [0] * 4)
        for i, value in enumerate(values):
            dest[i] += value
    assert set(totals) == {0, 1}, totals
    return totals


off, on = [counts(arms[name]["requests"][1]) for name in ("counts-off", "counts-on")]
mechanism = {}
for device in (0, 1):
    assert off[device][0] == off[device][1], off
    assert off[device][1:] == on[device][1:], (off, on)
    assert on[device][0] < off[device][0] and off[device][3] == on[device][3] == 0
    mechanism[device] = {"off_launches": off[device][0], "on_launches": on[device][0],
                         "experts": off[device][1], "bytes": off[device][2],
                         "fallbacks": 0, "launch_reduction_percent": 100 * (1 - on[device][0] / off[device][0])}
print(json.dumps({"status": "passed", "arms": list(expected),
                  "requests": sum(expected.values()), "mechanism": mechanism}, indent=2))
