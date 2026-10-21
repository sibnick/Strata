#!/usr/bin/env python3
import importlib.util
import json
from pathlib import Path
import unittest

OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("stage_chunks", OUT / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


class StageChunkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = run.long.profile.prepare.load_tokenizer()

    def test_gpu_reserve_policy_keeps_host_and_compute_guards(self):
        gpus = [{"free_bytes": 0}, {"free_bytes": 0}]
        run.check_pressure({"MemAvailable": 5 * run.short.GIB}, gpus, [])
        with self.assertRaisesRegex(RuntimeError, "host"):
            run.check_pressure({"MemAvailable": 5 * run.short.GIB - 1}, gpus, [])
        with self.assertRaisesRegex(RuntimeError, "unrelated"):
            run.check_pressure({"MemAvailable": 5 * run.short.GIB}, gpus, ["123"])
        with self.assertRaisesRegex(RuntimeError, "invalid GPU"):
            run.check_pressure({"MemAvailable": 5 * run.short.GIB}, [{"free_bytes": -1}, gpus[1]], [])

    def test_exact_offline_sizing_and_fit(self):
        data = json.loads((OUT / "offline.json").read_text())
        self.assertEqual(data["per_stage_chunks"], [2048, 512])
        self.assertEqual(data["cuda_reserve_mib"], 0)
        self.assertIsNone(data["external_gpu_free_abort_mib"])
        for boundary in ("36", "37"):
            self.assertEqual(
                [data["sizing"][boundary][stage]["owned_prefill_allocation_estimate"] for stage in ("0", "1")],
                [1_430_257_664, 507_510_784],
            )
        k36_rtx = data["sizing"]["36"]["1"]
        self.assertTrue(k36_rtx["fits_current_idle_free_memory"])
        self.assertTrue(k36_rtx["passes_mandatory_prompt_init_admission"])
        self.assertEqual(k36_rtx["admission_margin_after_mandatory_headroom_bytes"], 220_824_340)
        self.assertTrue(data["sizing"]["36"]["0"]["passes_mandatory_prompt_init_admission"])
        self.assertTrue(data["sizing"]["37"]["0"]["passes_mandatory_prompt_init_admission"])
        self.assertTrue(data["sizing"]["37"]["1"]["passes_mandatory_prompt_init_admission"])

    def test_request_shapes_and_historical_ids(self):
        for name, token_count in (("short", 1977), ("32k", 32788)):
            meta, ids, ref = run.fixture(name)
            self.assertEqual(len(ids), token_count)
            self.assertEqual(meta["generation_cap"], 32 if name == "short" else 192)
            positions = list(range(512, token_count - 1, 512)) + [token_count - 1]
            lines = [f"PP {position}" for position in positions]
            lines.extend(f"T {token}" for token in ref["generated_ids"])
            lines.append(ref["done"])
            result = run.validate(lines, meta, ids, ref, self.tokenizer)
            self.assertEqual(sum(result["actual_chunk_shape"]), token_count - 1)
            self.assertEqual(result["stage_chunks"], [2048, 512])
            self.assertEqual(result["fresh"], token_count)
            self.assertEqual(result["reused"], 0)
            with self.assertRaisesRegex(RuntimeError, "actual prompt chunks"):
                run.validate(
                    [f"PP {position}" for position in list(range(1024, token_count - 1, 1024)) + [token_count - 1]]
                    + lines[len(positions):],
                    meta, ids, ref, self.tokenizer,
                )

    def test_stage_one_consumes_larger_handoff_in_subchunks(self):
        expected = "strata prefill: layer handoff CUDA1: 1976 rows consumed in 4 sub-chunks (512-token stage chunks)"
        self.assertEqual(run.handoff_gate([expected], 1977), expected)
        with self.assertRaisesRegex(RuntimeError, "larger stage-0 handoff"):
            run.handoff_gate(["strata prefill: layer handoff CUDA1: 1976 rows consumed in 2 sub-chunks"], 1977)

    def test_startup_gate_checks_stage_scratch_without_changing_baseline_identity(self):
        lines = (run.prior.OUT / "on/engine.log").read_text().split("READY 262144 stop")[0].splitlines()
        previous_dso = str(Path(run.REFERENCE["argv"][0]).parent / "libstrata_kq.so")
        current_dso = str(run.ENGINE.parent / "libstrata_kq.so")
        lines = [line.replace(previous_dso, current_dso) for line in lines]
        replacements = {
            "CUDA0 owns layers [0,37), 18944 experts, 59192115200 resident bytes":
                "CUDA0 owns layers [0,36), 18432 experts, 57619251200 resident bytes",
            "CUDA1 owns layers [37,48), 5632 experts, 17825792000 resident bytes":
                "CUDA1 owns layers [36,48), 6144 experts, 19398656000 resident bytes",
            "layer split: CUDA0 runs layers 0-36": "layer split: CUDA0 runs layers 0-35",
            "layer split: CUDA1 runs layers 37-47": "layer split: CUDA1 runs layers 36-47",
            "expert_slots_primary=18944": "expert_slots_primary=18432",
        }
        lines = [next((line.replace(old, new) for old, new in replacements.items() if old in line), line)
                 for line in lines]
        sizing = json.loads((OUT / "offline.json").read_text())["sizing"]["36"]
        lines.extend([
            "strata generate: per-stage prefill chunks: CUDA0=2048 CUDA1=512",
            f"strata generate: --gpu-owned-experts exact prefill pricing: the prompt path's own buffers for a "
            f"2048-token chunk: {sizing['0']['owned_prefill_allocation_estimate'] // run.short.MIB} MiB",
            f"strata generate: --prefill-stage-chunks: CUDA1 chunk 512 exact scratch "
            f"{sizing['1']['owned_prefill_allocation_estimate']} bytes; conservative reserve 1200 MiB",
        ])
        argv = list(run.REFERENCE["argv"])
        argv[0] = str(run.ENGINE)
        argv[argv.index("--layer-split") + 1] = "36"
        argv[argv.index("--prefill") + 1] = "2048"
        index = argv.index("--prefill") + 2
        argv[index:index] = ["--prefill-stage-chunks", "2048,512"]
        for flag in ("--vram-reserve-mib", "--vram-reserve-later-mib"):
            argv[argv.index(flag) + 1] = "0"
        run.startup_gate(lines, argv)
        with self.assertRaisesRegex(RuntimeError, "startup admission missing"):
            run.startup_gate(
                [line.replace("507510784", "507510785") for line in lines], argv
            )


if __name__ == "__main__":
    unittest.main()
