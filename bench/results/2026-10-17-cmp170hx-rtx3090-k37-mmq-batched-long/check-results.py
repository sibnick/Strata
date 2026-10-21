#!/usr/bin/env python3
"""Independent read-only verification of the authorized five-request schedule."""
import importlib.util
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("long_result_check", OUT / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)
run.verify_identity()
bundle = run.fixtures()
tokenizer = run.long.profile.prepare.load_tokenizer()
arms = {a: json.loads((OUT / a / "results.json").read_text()) for a in ("off", "on")}
run.off_gate(arms["off"], arms["on"]["argv"], arms["on"]["environment_overrides"])
for arm, result in arms.items():
    assert result["status"] == "passed" and result["engine_return_code"] == 0, arm
    assert not result["compute_processes_after"], arm
    assert [r["name"] for r in result["requests"]] == \
        ["warmup", "32k"] + (["near-max"] if arm == "on" else []), arm
    assert result["guards"] == {
        "gpu_stop_mib": 640, "allocation_reserves_mib": [1024, 1024],
        "host_start_gib": 6, "host_stop_gib": 5, "host_floor_gib": 4,
        "reaction_headroom_gib": 1, "poll_seconds": 1}
    for key in ("engine_sha256", "kquant_library_sha256", "cache_sha256"):
        assert result[key] == arms["off"][key], (arm, key)
    for request in result["requests"]:
        meta, ids, reference = bundle[request["name"]]
        run.validate(request["done"], ids, request["generated_ids"], tokenizer, meta, reference)
        if request["name"] != "warmup":
            assert request["counter_lines"][-1].endswith("for 0.0 MB of expert reads")
        assert request["ttft_seconds_observed"] > 0
        assert request["pcie"]["available"]
        assert request["timeout_seconds"] >= 1800
    samples = [json.loads(s) for s in (OUT / arm / "monitor.jsonl").read_text().splitlines()]
    for sample in samples:
        run.short.check_pressure(sample["memory"], sample["gpu_memory_bytes"], sample["other_compute"])
        assert sample["process"].get("VmSwap", 0) == 0
    lines = (OUT / arm / "engine.log").read_text().splitlines()
    run.short.startup_gate(lines[:lines.index("READY 262144 stop") + 1], result["argv"], run.ENGINE)
    assert not any(s.startswith(("strata prefill timing:", "strata gather counts:", "strata trace:")) for s in lines)
    assert any("batched gather enabled" in s for s in lines) == (arm == "on")
assert arms["on"]["batch_opt_in_confirmed"] and arms["on"]["near_limit_gate"]["on_32k_exact_parity"]
assert arms["on"]["near_limit_gate"]["off_results_sha256"] == run.base.sha256(OUT / "off/results.json")
print(json.dumps({"status": "passed", "requests": 5,
                  "controlled_32k_exact_output_parity": True, "near_limit_matches_historical_ids": True,
                  "full_fresh_input_zero_reuse": True, "both_640_guards_and_host_5gib_passed": True,
                  "unchanged_candidate_identity": True, "both_clean_exit_zero": True}, indent=2))
