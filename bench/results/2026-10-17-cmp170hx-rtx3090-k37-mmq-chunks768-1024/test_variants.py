import importlib.util
import json
from pathlib import Path
import unittest

OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("larger_chunks", OUT / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


class VariantTests(unittest.TestCase):
    def test_unchanged_binary_and_source(self):
        run.identity()

    def test_only_gpu_threshold_removed(self):
        gpu = [{"free_bytes": 0}, {"free_bytes": 1}]
        run.check_pressure({"MemAvailable": 5 * run.short.GIB}, gpu, [])
        with self.assertRaisesRegex(RuntimeError, "host"):
            run.check_pressure({"MemAvailable": 5 * run.short.GIB - 1}, gpu, [])
        with self.assertRaisesRegex(RuntimeError, "unrelated"):
            run.check_pressure({"MemAvailable": 5 * run.short.GIB}, gpu, ["123"])
        with self.assertRaisesRegex(RuntimeError, "invalid GPU"):
            run.check_pressure({"MemAvailable": 5 * run.short.GIB}, [], [])

    def test_fixtures_shapes_exact_outputs(self):
        tokenizer = run.long.profile.prepare.load_tokenizer()
        for name, count, cap in (("short", 1977, 32), ("32k", 32788, 192)):
            meta, ids, ref = run.fixture(name)
            self.assertEqual((len(ids), meta["generation_cap"]), (count, cap))
            for chunk in (768, 1024):
                positions = list(range(chunk, count - 1, chunk)) + [count - 1]
                lines = [f"PP {n}" for n in positions]
                lines += [f"T {token}" for token in ref["generated_ids"]] + [ref["done"]]
                result = run.validate(lines, meta, ids, ref, chunk, tokenizer)
                self.assertEqual(sum(result["actual_chunk_shape"]), count - 1)
                self.assertEqual(result["execution_strategy_per_chunk"][0],
                                 "stream_all resident-filtered" if chunk == 1024 else "routed MMQ")
                with self.assertRaisesRegex(RuntimeError, "actual prompt chunks"):
                    run.validate(lines, meta, ids, ref, 512, tokenizer)
                fields = ref["done"].split()
                fields[8] = "1"
                with self.assertRaises((RuntimeError, ValueError)):
                    run.validate(lines[:-1] + [" ".join(fields)], meta, ids, ref, chunk, tokenizer)
                with self.assertRaisesRegex(RuntimeError, "IDs"):
                    run.validate(lines, meta, ids, dict(ref, generated_ids=[0]), chunk, tokenizer)

    def test_exact_sizing_and_adjustment(self):
        data = json.loads((OUT / "offline.json").read_text())
        for stage in ("0", "1"):
            for chunk, expected in (("768", 572522496), ("1024", 1025507328)):
                row = data["k37"][chunk][stage]
                self.assertEqual(row["owned_prefill_allocation_estimate"], expected)
                self.assertEqual(row["required_free_before_model_load"] -
                                 row["required_without_discretionary_cuda_allowance"], 1 << 30)
        self.assertIsNone(data["authorization"]["external_gpu_free_abort_threshold"])

    def test_zero_reserve_startup_gate(self):
        original = (run.prior.OUT / "on/engine.log").read_text().split("READY 262144 stop")[0].splitlines()
        for chunk, mib, correction in ((768, 546, 0), (1024, 978, 0)):
            # Preserve every original admission identity line; synthesize only the source-derived price lines.
            lines = []
            for line in original:
                if "own buffers for a 256-token chunk:" in line:
                    line = line.replace("256-token", f"{chunk}-token").replace("414 MiB", f"{mib} MiB")
                if "CUDA1 adds 155189248 bytes" in line:
                    line = line.replace("155189248 bytes", f"{correction} bytes")
                lines.append(line)
            argv = list(run.REFERENCE["argv"])
            argv[argv.index("--prefill") + 1] = str(chunk)
            for flag in ("--vram-reserve-mib", "--vram-reserve-later-mib"):
                argv[argv.index(flag) + 1] = "0"
            run.startup_gate(lines, argv, chunk)
            with self.assertRaisesRegex(RuntimeError, "clamped"):
                run.startup_gate(lines + ["strata serve: trying a 512-token chunk"], argv, chunk)
            argv[argv.index("--vram-reserve-mib") + 1] = "1024"
            with self.assertRaisesRegex(RuntimeError, "zero reserve"):
                run.startup_gate(lines, argv, chunk)


if __name__ == "__main__":
    unittest.main()
