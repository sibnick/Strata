import unittest

import importlib.util
from pathlib import Path

from summarize import stage_times

spec = importlib.util.spec_from_file_location("profile_short", Path(__file__).with_name("profile-short.py"))
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


class Tokenizer:
    def decode(self, ids):
        return "ORIGIN: 714\nCENTER: 928\nFINAL: 365"


class ShortProfileTests(unittest.TestCase):
    def setUp(self):
        profile.prepare.EXPECTED = {"ORIGIN": "714", "CENTER": "928", "FINAL": "365"}

    def test_valid_fresh_answer(self):
        line = "DONE 2 3 20.0 4.0 stop 0 0 0 0 0 0 0 0.0 3 0"
        result = profile.validate_done(line, [1, 2, 3], [1, 248046], Tokenizer())
        self.assertEqual(result["fresh"], 3)
        self.assertEqual(result["prefill_tokens_per_second"], 150)

    def test_reuse_rejected(self):
        line = "DONE 2 3 20.0 4.0 stop 0 0 1 0 0 0 0 0.0 2 0"
        with self.assertRaisesRegex(ValueError, "without reuse"):
            profile.validate_done(line, [1, 2, 3], [1, 248046], Tokenizer())

    def test_incomplete_answer_rejected(self):
        line = "DONE 2 3 20.0 4.0 length 0 0 0 0 0 0 0 0.0 3 0"
        with self.assertRaisesRegex(ValueError, "did not complete"):
            profile.validate_done(line, [1, 2, 3], [1, 248046], Tokenizer())

    def test_prompt_caps(self):
        bundle = profile.prompts()
        self.assertEqual(len(bundle), 3)
        for metadata, ids in bundle:
            self.assertLessEqual(len(ids), 2048)
            self.assertGreater(len(ids), 256)
            self.assertEqual(metadata["generation_cap"], 32)

    def test_stage_attribution_and_overlap(self):
        lines = [
            "strata prefill timing: 256 tokens, GPU timeline 10 ms, wall 12 ms, host staging 0 ms: dequant 8 (80.0%) gemm gate/up 2 (20.0%)",
            "strata prefill timing: 245 tokens, GPU timeline 9 ms, wall 11 ms, host staging 0 ms: dequant 7 (77.8%) gemm gate/up 2 (22.2%)",
            "strata prefill timing: 501 tokens, GPU timeline 50 ms, wall 55 ms, host staging 0 ms: dequant 40 (80.0%) gemm gate/up 10 (20.0%)",
        ]
        stages = stage_times(lines)
        self.assertEqual(stages["CMP_prefix"]["tokens"], 501)
        self.assertEqual(stages["RTX_suffix"]["sum_gpu_timeline_ms"], 19)
        self.assertEqual(stages["RTX_suffix"]["phase_ms"]["dequant"], 15)
        with self.assertRaisesRegex(ValueError, "token mismatch"):
            stage_times(lines[1:])


if __name__ == "__main__":
    unittest.main()
