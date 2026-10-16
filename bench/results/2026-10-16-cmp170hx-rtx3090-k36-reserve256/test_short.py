import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location("runner", Path(__file__).with_name("run-short.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class Guards(unittest.TestCase):
    def check(self, rtx, cmp, host=5 << 30):
        runner.check_pressure({"MemAvailable": host}, [
            {"physical_index": 0, "free_bytes": rtx},
            {"physical_index": 1, "free_bytes": cmp},
        ], [])

    def test_boundaries(self):
        self.check(256 << 20, 640 << 20)

    def test_rtx_below(self):
        with self.assertRaisesRegex(RuntimeError, "GPU 0 below 256"):
            self.check((256 << 20) - 1, 640 << 20)

    def test_cmp_unchanged(self):
        with self.assertRaisesRegex(RuntimeError, "GPU 1 below 640"):
            self.check(256 << 20, (640 << 20) - 1)

    def test_host_unchanged(self):
        with self.assertRaisesRegex(RuntimeError, "5 GiB"):
            self.check(256 << 20, 640 << 20, (5 << 30) - 1)

    def test_exit_race_explicit(self):
        process = Mock(pid=123)
        process.poll.return_value = 1
        with patch.object(runner.base, "process_state", side_effect=PermissionError):
            self.assertEqual(runner.process_state(process), {"exited_return_code": 1})

    def test_live_permission_failure_not_hidden(self):
        process = Mock(pid=123)
        process.poll.return_value = None
        with patch.object(runner.base, "process_state", side_effect=PermissionError):
            with self.assertRaises(PermissionError):
                runner.process_state(process)


if __name__ == "__main__":
    unittest.main()
