import importlib.util
from pathlib import Path
import unittest

from analyze import k36_delta

spec = importlib.util.spec_from_file_location("run_short", Path(__file__).with_name("run-short.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class GuardTests(unittest.TestCase):
    def test_exact_640_mib_allowed(self):
        runner.check_pressure({"MemAvailable": 5 << 30}, [{"free_bytes": 640 << 20}], [])

    def test_one_byte_below_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "640 MiB"):
            runner.check_pressure({"MemAvailable": 5 << 30}, [{"free_bytes": (640 << 20) - 1}], [])

    def test_host_reaction_margin_preserved(self):
        with self.assertRaisesRegex(RuntimeError, "5 GiB"):
            runner.check_pressure({"MemAvailable": (5 << 30) - 1}, [{"free_bytes": 1 << 30}], [])

    def test_other_compute_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "other compute"):
            runner.check_pressure({"MemAvailable": 6 << 30}, [{"free_bytes": 1 << 30}], ["123"])

    def test_original_prompt_hashes(self):
        runner.profile.OUT = runner.OUT
        import json
        for metadata, ids in runner.profile.prompts():
            old = json.loads((runner.PRIOR / "prompts" / (metadata["name"] + ".json")).read_text())
            self.assertEqual(metadata["prompt_ids_sha256"], old["prompt_ids_sha256"])
            self.assertLessEqual(len(ids), 2048)
            self.assertEqual(metadata["generation_cap"], 32)

    def test_k36_ledger_deltas(self):
        delta = k36_delta()
        self.assertEqual(delta["1"]["expert_payload"]["delta_bytes"], 1500 << 20)
        self.assertEqual(delta["1"]["native_projection_allocation_estimate"]["delta_bytes"], 66 << 20)
        self.assertEqual(delta["1"]["session_arena"]["delta_bytes"], 3268608)
        self.assertEqual(delta["1"]["owned_prefill_allocation_estimate"]["delta_bytes"], 0)
        self.assertEqual(delta["1"]["required_free_before_model_load"]["delta_bytes"], 1598 << 20)
        self.assertEqual(delta["0"]["required_free_before_model_load"]["delta_bytes"], -(1594 << 20))


if __name__ == "__main__":
    unittest.main()
