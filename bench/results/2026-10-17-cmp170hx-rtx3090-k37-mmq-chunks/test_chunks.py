import importlib.util
import json
from pathlib import Path
import unittest

OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("chunks", OUT / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


class ChunkTests(unittest.TestCase):
    def test_identity(self):
        run.identity()

    def test_bracket_is_baseline_only(self):
        with self.assertRaisesRegex(ValueError, "baseline256"):
            run.run(384, bracket=True)

    def test_fixtures_and_exact_outputs(self):
        tokenizer = run.long.profile.prepare.load_tokenizer()
        for name, count, cap in (("short", 1977, 32), ("32k", 32788, 192)):
            meta, ids, ref = run.fixture(name)
            self.assertEqual((len(ids), meta["generation_cap"]), (count, cap))
            lines = [f"PP {n} {count} 0 0" for n in list(range(256, count - 1, 256)) + [count - 1]]
            lines += [f"T {token}" for token in ref["generated_ids"]] + [ref["done"]]
            run.validate(lines, meta, ids, ref, 256, tokenizer)
            with self.assertRaisesRegex(RuntimeError, "actual prompt chunks"):
                run.validate(lines, meta, ids, ref, 384, tokenizer)
            bad = dict(ref, generated_ids=[0])
            with self.assertRaisesRegex(RuntimeError, "IDs"):
                run.validate(lines, meta, ids, bad, 256, tokenizer)

    def test_shapes(self):
        for chunk in (256, 384, 512):
            pos = list(range(chunk, 1976, chunk)) + [1976]
            shape = run.shapes([f"PP {n}" for n in pos], 1977, chunk)
            self.assertEqual(sum(shape), 1976)
            self.assertEqual(shape[0], chunk)
            self.assertLessEqual(shape[-1], chunk)
        with self.assertRaises(RuntimeError):
            run.shapes(["PP 256", "PP 1976"], 1977, 512)

    def test_pricing(self):
        sizing = json.loads((OUT / "offline.json").read_text())["k37"]
        for stage in ("0", "1"):
            self.assertEqual([sizing[str(c)][stage]["owned_prefill_allocation_estimate"]
                              for c in (256, 384, 512)], [434110464, 461373440, 507510784])
            self.assertEqual(len({sizing[str(c)][stage]["session_arena"] for c in (256, 384, 512)}), 1)
            self.assertEqual(len({sizing[str(c)][stage]["expert_payload"] for c in (256, 384, 512)}), 1)

    def test_startup(self):
        lines = (run.prior.OUT / "on/engine.log").read_text().split("READY 262144 stop")[0].splitlines()
        run.startup_gate(lines, run.REFERENCE["argv"], 256)
        with self.assertRaisesRegex(RuntimeError, "configured chunk"):
            run.startup_gate(lines, run.REFERENCE["argv"], 384)
        with self.assertRaisesRegex(RuntimeError, "clamped"):
            run.startup_gate(lines + ["strata serve: trying a 512-token chunk"], run.REFERENCE["argv"], 256)

    def test_guards(self):
        gpus = [{"free_bytes": 640 * run.short.MIB}] * 2
        run.short.check_pressure({"MemAvailable": 5 * run.short.GIB}, gpus, [])
        for index in (0, 1):
            bad = list(gpus)
            bad[index] = {"free_bytes": 640 * run.short.MIB - 1}
            with self.assertRaisesRegex(RuntimeError, "GPU"):
                run.short.check_pressure({"MemAvailable": 5 * run.short.GIB}, bad, [])
        with self.assertRaisesRegex(RuntimeError, "host"):
            run.short.check_pressure({"MemAvailable": 5 * run.short.GIB - 1}, gpus, [])
        with self.assertRaisesRegex(RuntimeError, "unrelated"):
            run.short.check_pressure({"MemAvailable": 5 * run.short.GIB}, gpus, ["123"])

    def test_tile_audit(self):
        audit = json.loads((OUT / "tile-audit.json").read_text())
        self.assertEqual(audit["dependency_diff"], "")
        self.assertEqual(audit["dependency_commit"], "3cf03257f219afbe7334045ff7c6a06ac68c627d")
        rows = audit["configs_and_resources"]
        self.assertEqual(len(rows), 66)
        self.assertTrue(all(r["fits_shared_per_block"] and r["I"] == 128 and
                            r["threads"] == 256 and r["stream_k"] and not r["fallback"] for r in rows))
        table = audit["selected_J_for_group_max_rows_1_through_512"]
        for n, j in ((1, 8), (9, 16), (17, 24), (25, 32), (49, 64), (113, 128), (129, 80)):
            self.assertEqual(table[str(n)], j)


if __name__ == "__main__":
    unittest.main()
