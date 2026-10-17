#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_170hx_3090", ROOT / "tools/run_170hx_3090.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class RunnerRecipeTests(unittest.TestCase):
    def test_recipe_keeps_validated_k36_placement_and_chunks(self):
        args = runner.build_config()["args"]
        for flag, expected in (
            ("--gpu", "1,0"),
            ("--layer-split", "36"),
            ("--max-context", "262144"),
            ("--kv", "int8"),
            ("--prefill", "2048"),
            ("--prefill-stage-chunks", "2048,512"),
            ("--mtp-draft-vocab", str(runner.DRAFT_VOCAB)),
            ("--vram-reserve-mib", "0"),
            ("--vram-reserve-later-mib", "0"),
        ):
            self.assertEqual(args[args.index(flag) + 1], expected)

    def test_api_is_loopback_only_and_gpu_owned(self):
        config = runner.build_config()
        args = config["args"]
        self.assertEqual((config["host"], config["port"]), ("127.0.0.1", 8080))
        self.assertIn("--serve", args)
        self.assertIn("--gpu-owned-experts", args)
        self.assertIn("--no-prefill-borrow", args)
        self.assertNotIn("--api-key", args)

    def test_cyrillic_subset_is_not_an_ngram_switch(self):
        self.assertTrue(runner.DRAFT_VOCAB.is_file())
        self.assertIn("--mtp-draft-vocab", runner.build_config()["args"])


if __name__ == "__main__":
    unittest.main()
