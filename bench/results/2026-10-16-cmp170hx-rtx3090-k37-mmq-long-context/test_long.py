import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock

spec = importlib.util.spec_from_file_location("runner", Path(__file__).with_name("run-long.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class LongTests(unittest.TestCase):
    def test_original_near_limit_fixture(self):
        meta, ids = runner.base.load_bundle("near-max")
        self.assertEqual(len(ids), 261682)
        self.assertEqual(meta["generation_cap"], 192)
        self.assertEqual(meta["prompt_ids_sha256"], "67a876781b4fde7323b1b340a5e55700acb3f6741dc19d953d32229c42cf55ed")
        self.assertLess(len(ids) + 192, 262144)

    def test_timeout_uses_preceding_rate(self):
        self.assertEqual(runner.request_timeout(261682, 300), 4 * 261682 / 300 + 300)
        self.assertEqual(runner.request_timeout(1977, 700), 1800)

    def test_both_guards_640(self):
        self.assertEqual(runner.GPU_STOP_MIB, {0: 640, 1: 640})
        for index in (0, 1):
            with self.assertRaises(RuntimeError):
                runner.check_pressure({"MemAvailable": 5 << 30}, [
                    {"physical_index": index, "free_bytes": (640 << 20) - 1}
                ], [])

    def test_long_answer_validation(self):
        tokenizer = Mock()
        tokenizer.decode.return_value = "ORIGIN: A\nCENTER: B\nFINAL: C"
        meta = {"generation_cap": 192, "expected": {"ORIGIN": "A", "CENTER": "B", "FINAL": "C"}}
        line = "DONE 2 3 20 4 stop 1 2 0 0 0 0 0 0 3 0"
        self.assertEqual(runner.validate_done(line, [1, 2, 3], [1, 248046], tokenizer, meta)["fresh"], 3)
        with self.assertRaises(ValueError):
            runner.validate_done(line.replace("stop", "length"), [1, 2, 3], [1, 248046], tokenizer, meta)
        with self.assertRaises(ValueError):
            runner.validate_done(line.replace("2 0 0", "2 1 0"), [1, 2, 3], [1, 248046], tokenizer, meta)


if __name__ == "__main__":
    unittest.main()
