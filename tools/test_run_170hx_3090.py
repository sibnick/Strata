#!/usr/bin/env python3
import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_170hx_3090", ROOT / "tools/run_170hx_3090.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class RunnerRecipeTests(unittest.TestCase):
    def test_ampere_decode_settings_reach_server_config(self):
        expected = {
            "STRATA_KQ_SPLIT": "1",
            "STRATA_SIDE_MIXER": "2",
            "STRATA_ATTN_DAILY": "1",
            "STRATA_GR_UP75": "1",
            "STRATA_QSA_ROWS": "1",
            "STRATA_B6_MMVQ_ROWS": "1",
        }
        env = runner.build_config()["env"]
        for key, value in expected.items():
            self.assertEqual(env[key], value)
        self.assertNotIn("STRATA_SPLIT_OVERLAP", env)

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
        self.assertEqual(config["log"], str(runner.ENGINE_LOG))
        self.assertIn("--serve", args)
        self.assertIn("--gpu-owned-experts", args)
        self.assertIn("--no-prefill-borrow", args)
        self.assertNotIn("--api-key", args)

    def test_conversation_checkpoints_use_engine_default(self):
        self.assertNotIn("--prompt-cache", runner.build_config()["args"])

    def test_conversation_parking_keeps_custom_runner_settings(self):
        args = runner.build_config()["args"]
        self.assertEqual(args[args.index("--conversation-cache-mib") + 1], "16384")
        self.assertEqual(args[args.index("--conversation-cache-slots") + 1], "32")

    def test_cyrillic_subset_is_not_an_ngram_switch(self):
        self.assertTrue(runner.DRAFT_VOCAB.is_file())
        self.assertIn("--mtp-draft-vocab", runner.build_config()["args"])

    def test_readiness_requires_a_valid_loaded_health_response(self):
        response = MagicMock()
        response.read.return_value = json.dumps({
            "status": "ok", "service": "strata", "loaded": True,
        }).encode()
        response.__enter__.return_value = response
        with patch.object(runner.urllib.request, "urlopen", return_value=response) as urlopen:
            self.assertTrue(runner.api_health()["loaded"])
        urlopen.assert_called_once_with("http://127.0.0.1:8080/health", timeout=1)

        response.read.return_value = json.dumps({
            "status": "ok", "service": "strata", "loaded": False,
        }).encode()
        with patch.object(runner.urllib.request, "urlopen", return_value=response):
            self.assertFalse(runner.api_health()["loaded"])

        with patch.object(runner.urllib.request, "urlopen", side_effect=OSError("not listening")):
            self.assertIsNone(runner.api_health())

    def test_stale_engine_help_is_rejected_before_startup(self):
        with self.assertRaisesRegex(RuntimeError, "Run ./run-170hx-3090.sh build"):
            runner.validate_engine_flags("--gpu-owned-experts")


if __name__ == "__main__":
    unittest.main()
