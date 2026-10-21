"""Bounded checks for the optional CUDA split-memory diagnostic; no model loading."""
import subprocess
import unittest
from pathlib import Path


BINARY = Path(__file__).resolve().parents[1] / "build" / "strata-split-memory"


class SplitMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not BINARY.is_file():
            raise unittest.SkipTest("build the strata-split-memory target first")

    def run_ledger(self, *args):
        return subprocess.run([str(BINARY), *args], capture_output=True, text=True, timeout=15)

    def test_session_contract(self):
        result = self.run_ledger("--self-test")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("self-test passed", result.stdout)

    def test_help(self):
        result = self.run_ledger("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("not a speed prediction or a fit guarantee", result.stdout)

    def test_missing_arguments(self):
        result = self.run_ledger()
        self.assertEqual(result.returncode, 1)
        self.assertIn("--pack, --native, --mtp", result.stderr)

    def test_invalid_dimensions(self):
        for option in ("--context", "--chunk"):
            for value in ("0", "-1", "262145", "foo", "256x"):
                with self.subTest(option=option, value=value):
                    result = self.run_ledger(option, value)
                    self.assertEqual(result.returncode, 1)
                    self.assertTrue(result.stderr.startswith("strata-split-memory:"), result.stderr)


if __name__ == "__main__":
    unittest.main()
