import importlib.util
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load("run_k35", HERE / "run-k35.py")


class GuardTests(unittest.TestCase):
    def test_exact_memory_stops_are_allowed(self):
        runner.check_pressure(
            {"MemAvailable": 5 << 30},
            [
                {"physical_index": 0, "free_bytes": 256 << 20},
                {"physical_index": 1, "free_bytes": 640 << 20},
            ],
            [],
        )

    def test_one_byte_below_either_gpu_stop_is_rejected(self):
        for index, limit in ((0, 256 << 20), (1, 640 << 20)):
            gpu = [
                {"physical_index": 0, "free_bytes": 256 << 20},
                {"physical_index": 1, "free_bytes": 640 << 20},
            ]
            gpu[index]["free_bytes"] = limit - 1
            with self.subTest(index=index), self.assertRaisesRegex(
                RuntimeError, f"physical GPU {index}"
            ):
                runner.check_pressure({"MemAvailable": 5 << 30}, gpu, [])

    def test_host_reaction_margin_is_preserved(self):
        with self.assertRaisesRegex(RuntimeError, "5 GiB"):
            runner.check_pressure(
                {"MemAvailable": (5 << 30) - 1},
                [
                    {"physical_index": 0, "free_bytes": 1 << 30},
                    {"physical_index": 1, "free_bytes": 1 << 30},
                ],
                [],
            )

    def test_other_compute_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "another GPU"):
            runner.check_pressure(
                {"MemAvailable": 6 << 30},
                [
                    {"physical_index": 0, "free_bytes": 1 << 30},
                    {"physical_index": 1, "free_bytes": 1 << 30},
                ],
                ["123, unrelated, 1 MiB"],
            )

    def test_arms_differ_only_by_the_explicit_adaptive_schedule(self):
        control = runner.make_argv(False)
        adaptive = runner.make_argv(True)
        self.assertEqual(
            load("analyze_k35", HERE / "analyze.py").normalize_adaptive_argv(control),
            load("analyze_k35_again", HERE / "analyze.py").normalize_adaptive_argv(adaptive),
        )
        self.assertNotIn("--gpu-owned-experts", control)
        self.assertNotIn("--expert-cache", control)
        self.assertEqual(control[control.index("--layer-split") + 1], "35")
        self.assertEqual(control[control.index("--vram-reserve-later-mib") + 1], "2049")
        self.assertEqual(adaptive[adaptive.index("--adapt-every") + 1], "4")
        self.assertEqual(adaptive[adaptive.index("--adapt-swaps") + 1], "96")
        self.assertEqual(adaptive[adaptive.index("--adapt-async") + 1], "0")

    def test_environment_keeps_pinning_without_timing_instrumentation(self):
        _, overrides = runner.environment()
        self.assertEqual(overrides["STRATA_RESIDENT_PIN"], "1")
        self.assertEqual(overrides["STRATA_RESIDENT_HEADROOM_GIB"], "4")
        self.assertEqual(overrides["STRATA_PIN_RESERVE_GIB"], "4")
        self.assertNotIn("STRATA_TRACE", overrides)
        self.assertNotIn("STRATA_PREFILL_TIMING", overrides)

    def test_startup_gate_accepts_only_expected_k35_resource_shape(self):
        lines = [
            "strata: K-quant prompt kernels from libstrata_kq.so (STRATA_MMQ_KQ=1)",
            "strata generate: layer split: CUDA0 runs layers 0-34",
            "strata generate: layer split, CUDA1: 19.01 GiB free of 23.56, room for experts 16.59 GiB",
            "strata generate: profile profile.bin: 24576 ranked pairs, built for 24576 slots",
            "strata generate: expert cache 17920 slots, 52.20 GiB of VRAM; policy is",
            "strata generate: layer split: CUDA1 runs layers 35-47, expert cache 5659 slots (16.59 GiB), 5659 of its 6656 profiled pairs; slot 0 verified",
            "FileExpertSource: RAM budget 4.00 GiB: 997 of the 2.94 GiB of experts the GPU cache does not hold, by profile rank; the rest are read from the files",
            "FileExpertSource: mapped pinned cache complement ready: resident 2.94 GiB, pinned 2.94 GiB in 20 s",
        ]
        gpu = [
            {"physical_index": 0, "free_bytes": 512 << 20},
            {"physical_index": 1, "free_bytes": 1024 << 20},
        ]
        result = runner.startup_gate(lines, gpu, runner.make_argv(False))
        self.assertTrue(result["cold_complement"]["fully_page_locked"])
        self.assertEqual(result["stage1_cache"]["slots"], 5659)

    def test_startup_gate_rejects_file_backed_complement(self):
        lines = [
            "strata: K-quant prompt kernels from libstrata_kq.so (STRATA_MMQ_KQ=1)",
            "strata generate: layer split: CUDA0 runs layers 0-34",
            "strata generate: layer split, CUDA1: 19.01 GiB free of 23.56, room for experts 16.59 GiB",
            "strata generate: profile profile.bin: 24576 ranked pairs, built for 24576 slots",
            "strata generate: expert cache 17920 slots, 52.20 GiB of VRAM; policy is",
            "strata generate: layer split: CUDA1 runs layers 35-47, expert cache 5659 slots (16.59 GiB), 5659 of its 6656 profiled pairs; slot 0 verified",
            "FileExpertSource: RAM budget 4.00 GiB: 997 of the 4.20 GiB of experts the GPU cache does not hold, by profile rank; the rest are read from the files",
            "FileExpertSource: mapped pinned cache complement ready: resident 4.00 GiB, pinned 4.00 GiB in 20 s",
        ]
        argv = runner.make_argv(False)
        gpu = [
            {"physical_index": 0, "free_bytes": 512 << 20},
            {"physical_index": 1, "free_bytes": 1024 << 20},
        ]
        with self.assertRaisesRegex(RuntimeError, "cold complement"):
            runner.startup_gate(lines, gpu, argv)


if __name__ == "__main__":
    unittest.main()
