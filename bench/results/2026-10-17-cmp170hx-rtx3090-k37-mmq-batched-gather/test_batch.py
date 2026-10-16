import importlib.util
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("gather_run", HERE / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


class BatchTests(unittest.TestCase):
    def test_original_fixtures_and_outputs(self):
        self.assertEqual([len(p[1]) for p in run.bundle(["small", "medium", "large"])], [502, 994, 1977])
        for _, ids, reference in run.bundle(["small", "medium", "large"]):
            valid = run.validate_done(reference["done"], ids, reference["generated_ids"], reference)
            self.assertEqual(valid["fresh"], len(ids))
            fields = reference["done"].split()
            fields[8] = "1"
            with self.assertRaisesRegex(RuntimeError, "fresh"):
                run.validate_done(" ".join(fields), ids, reference["generated_ids"], reference)
            with self.assertRaisesRegex(RuntimeError, "differ"):
                run.validate_done(reference["done"], ids, [0] * len(reference["generated_ids"]), reference)

    def test_hard_guards(self):
        gpu = [{"free_bytes": 640 * run.MIB}] * 2
        run.check_pressure({"MemAvailable": 5 * run.GIB}, gpu, [])
        for device in range(2):
            bad = list(gpu)
            bad[device] = {"free_bytes": 640 * run.MIB - 1}
            with self.assertRaisesRegex(RuntimeError, "GPU"):
                run.check_pressure({"MemAvailable": 5 * run.GIB}, bad, [])
        with self.assertRaisesRegex(RuntimeError, "host"):
            run.check_pressure({"MemAvailable": 5 * run.GIB - 1}, gpu, [])
        with self.assertRaisesRegex(RuntimeError, "unrelated"):
            run.check_pressure({"MemAvailable": 5 * run.GIB}, gpu, ["123"])

    def test_startup_rejects_ownership_and_mapping_mismatch(self):
        engine = Path(run.REFERENCE["argv"][0])
        lines = (run.PRIOR / "mmq/engine.log").read_text().splitlines()
        lines = lines[:lines.index("READY 262144 stop") + 1]
        run.startup_gate(lines, run.REFERENCE["argv"], engine)
        for field in ("CUDA1 owns layers [37,48)", "all 24576 experts", "GPU 0: NVIDIA CMP"):
            with self.assertRaisesRegex(RuntimeError, "missing"):
                run.startup_gate([s for s in lines if field not in s], run.REFERENCE["argv"], engine)

    def test_mixed_ring_and_partial_flush_model(self):
        # Model both stage-ahead schedules. A pending grouped copy MUST be released before a slot can be refilled.
        for ahead in (False, True):
            for mask in (0, 1, 3, 7, 0xffff, 0xaaaa, 0x5555, 0x8101):
                order = [bool(mask & (1 << (q % 16))) for q in range(129)]
                live, released, staging = {}, set(), {}
                next_slot = staged = pending = first = 0
                gathered = []
                open_group = []
                for j in range(len(order)):
                    while staged < len(order) and (pending < 8 if ahead else staged <= j + 7):
                        if order[staged]:
                            slot = next_slot
                            next_slot = (next_slot + 1) % 8
                            self.assertTrue(slot not in live or live[slot] in released)
                            live[slot] = staged
                            staging[staged] = slot
                            pending += 1
                        staged += 1
                    q = j % 16
                    self.assertEqual(first + len(open_group), q)
                    open_group.append(j)
                    last = q == 15 or j == len(order) - 1
                    if last or j in staging:
                        gathered.extend(open_group)
                        released.update(open_group)
                        open_group = []
                        first = 0 if last else q + 1
                    if j in staging:
                        self.assertIn(j, released)
                        pending -= 1
                self.assertEqual(gathered, list(range(len(order))))
                self.assertFalse(open_group)


if __name__ == "__main__":
    unittest.main()
