import unittest

from analyze import prefill_rows, summarize_gpus, summary


class AnalysisTests(unittest.TestCase):
    def test_phase_edges_and_stage_selection(self):
        rows = [
            {"stage": stage, "request_elapsed_seconds": elapsed}
            for stage, elapsed in (
                ("startup", None), ("x", 1.9), ("x", 2), ("x", 8),
                ("x", 8.1), ("other", 4),
            )
        ]
        self.assertEqual(prefill_rows(rows, {"name": "x", "ttft_seconds": 10}), rows[2:4])

    def test_physical_gpu_index_not_array_order(self):
        rows = [{"gpus": [
            {"index": index, "name": f"GPU{index}", "utilization_gpu_percent": value, "power_w": 100}
            for index, value in ((1, 99), (0, 16))
        ]}]
        result = summarize_gpus(rows)
        self.assertEqual(result["0"]["utilization_percent"]["mean"], 16)
        self.assertEqual(result["1"]["utilization_percent"]["mean"], 99)
        self.assertEqual(result["0"]["samples_at_or_below_20_percent"], 1)
        self.assertEqual(result["1"]["samples_at_or_above_90_percent"], 1)

    def test_empty_and_nonempty_statistics(self):
        self.assertEqual(summary([]), {"sample_count": 0})
        self.assertEqual(summary([0, 20, 100]), {
            "sample_count": 3, "mean": 40, "median": 20, "minimum": 0, "maximum": 100,
        })


if __name__ == "__main__":
    unittest.main()
