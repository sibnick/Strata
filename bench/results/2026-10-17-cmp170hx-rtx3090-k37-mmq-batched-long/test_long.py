import copy
import importlib.util
from pathlib import Path
import unittest

OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("batched_long", OUT / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


class LongTests(unittest.TestCase):
    def test_immutable_original_fixtures(self):
        bundle = run.fixtures()
        self.assertEqual([len(bundle[n][1]) for n in ("warmup", "32k", "near-max")],
                         [1977, 32788, 261682])
        self.assertEqual(len(bundle["near-max"][1]) + 192, 261874)
        self.assertEqual(bundle["near-max"][0]["expected"], {
            "ORIGIN": "IRIS-4Q7-LANTERN", "CENTER": "CEDAR-8M2-QUARRY", "FINAL": "ORBIT-6V9-KETTLE"})

    def test_exact_output_and_freshness(self):
        tokenizer = run.long.profile.prepare.load_tokenizer()
        for meta, ids, ref in run.fixtures().values():
            run.validate(ref["done"], ids, ref["generated_ids"], tokenizer, meta, ref)
            fields = ref["done"].split()
            fields[8] = "1"
            with self.assertRaisesRegex(ValueError, "reuse"):
                run.validate(" ".join(fields), ids, ref["generated_ids"], tokenizer, meta, ref)
            bad_ref = dict(ref, generated_ids=[0] * len(ref["generated_ids"]))
            with self.assertRaisesRegex(RuntimeError, "IDs"):
                run.validate(ref["done"], ids, ref["generated_ids"], tokenizer, meta, bad_ref)

    def test_unchanged_build(self):
        run.verify_identity()

    def test_pressure_and_timeout(self):
        good = [{"free_bytes": 640 * run.short.MIB}] * 2
        run.short.check_pressure({"MemAvailable": 5 * run.short.GIB}, good, [])
        for i in (0, 1):
            bad = list(good)
            bad[i] = {"free_bytes": 640 * run.short.MIB - 1}
            with self.assertRaisesRegex(RuntimeError, "GPU"):
                run.short.check_pressure({"MemAvailable": 5 * run.short.GIB}, bad, [])
        with self.assertRaisesRegex(RuntimeError, "host"):
            run.short.check_pressure({"MemAvailable": 5 * run.short.GIB - 1}, good, [])
        with self.assertRaisesRegex(RuntimeError, "unrelated"):
            run.short.check_pressure({"MemAvailable": 5 * run.short.GIB}, good, ["123"])
        self.assertEqual(run.long.request_timeout(261682, 766.854), 1800)
        self.assertGreater(run.long.request_timeout(261682, 100), 1800)

    def test_off_admission_gate(self):
        env = {"STRATA_MMQ_KQ": "1", "STRATA_PREFILL_BATCH_GATHER": "1"}
        off = {"status": "passed", "engine_return_code": 0, "compute_processes_after": "",
               "argv": ["engine"], "environment_overrides": dict(env, STRATA_PREFILL_BATCH_GATHER="0"),
               "requests": [{"name": n, "reused": 0, "fresh": t, "input_tokens": t}
                            for n, t in (("warmup", 1977), ("32k", 32788))]}
        run.off_gate(off, ["engine"], env)
        for key, value in (("status", "failed"), ("engine_return_code", -15),
                           ("compute_processes_after", "123"), ("argv", ["wrong"]),
                           ("requests", [])):
            bad = copy.deepcopy(off)
            bad[key] = value
            with self.assertRaises(RuntimeError):
                run.off_gate(bad, ["engine"], env)


if __name__ == "__main__":
    unittest.main()
