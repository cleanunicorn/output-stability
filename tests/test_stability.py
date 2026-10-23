import copy
import json
from pathlib import Path
import tempfile
import unittest

import stability_lab as lab


class StabilityTests(unittest.TestCase):
    def record(self, variant, text, **kw):
        return {"variant": variant, "prompt": "p", "text": text,
                "text_sha256": lab.digest(text.encode()), "prompt_sha256": "same", **kw}

    def test_repeatability_is_distinct_from_matching_baseline(self):
        records = [self.record("baseline", "A") for _ in range(3)]
        records += [self.record("different", "B") for _ in range(3)]
        records += [self.record("unstable", text) for text in ["A", "B", "A"]]
        rows = {r["variant"]: r for r in lab.summarize(records)}
        self.assertTrue(rows["different"]["repeatable_observed"])
        self.assertEqual(rows["different"]["matches_baseline"], 0)
        self.assertFalse(rows["unstable"]["repeatable_observed"])
        self.assertEqual(rows["unstable"]["unique_outputs"], 2)

    def test_errors_cannot_pass_stability(self):
        records = [self.record("baseline", "A"), self.record("baseline", "A"),
                   {"variant": "baseline", "prompt": "p", "error": "crash"}]
        row = lab.summarize(records)[0]
        self.assertEqual(row["errors"], 1)
        self.assertFalse(row["repeatable_observed"])

    def test_exact_whitespace_and_truncation(self):
        records = [self.record("baseline", "A\n", finish_reason="length"), self.record("baseline", "A")]
        row = lab.summarize(records)[0]
        self.assertEqual(row["unique_outputs"], 2)
        self.assertEqual(row["truncated"], 1)

    def test_missing_first_baseline_is_not_silently_replaced(self):
        records = [{"variant": "baseline", "prompt": "p", "error": "failed"},
                   self.record("baseline", "A"), self.record("v", "A")]
        self.assertIsNone(lab.summarize(records)[1]["matches_baseline"])

    def test_cross_run_compare_rejects_changed_prompts_errors_and_variation(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.json", Path(tmp) / "b.json"
            base = {"records": [self.record("baseline", "A"), self.record("baseline", "A")]}
            lab.write_json(a, base)
            lab.write_json(b, base)
            self.assertTrue(lab.compare(a, b)["all_identical_observed"])
            changed = copy.deepcopy(base)
            changed["records"][0]["prompt_sha256"] = "changed"
            lab.write_json(b, changed)
            self.assertFalse(lab.compare(a, b)["all_identical_observed"])
            lab.write_json(b, {"records": base["records"][:1]})
            self.assertFalse(lab.compare(a, b)["all_identical_observed"])
            lab.write_json(b, {"records": []})
            self.assertFalse(lab.compare(a, b)["all_identical_observed"])
            changed = copy.deepcopy(base)
            changed["records"][0]["error"] = "failed"
            lab.write_json(b, changed)
            self.assertFalse(lab.compare(a, b)["all_identical_observed"])
            changed = copy.deepcopy(base)
            changed["records"][0] = self.record("baseline", "B")
            lab.write_json(b, changed)
            self.assertFalse(lab.compare(a, b)["all_identical_observed"])

    def test_fresh_and_warm_subprocess_reporting(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {"backend": "demo", "repeats": 3,
                      "prompts": [{"name": "p", "text": "test", "sha256": lab.digest(b"test")}],
                      "variants": [{"name": "different", "generate": {"temperature": 0.6}},
                                   {"name": "unstable", "load": {"demo_unstable": True}}]}
            for mode in ("fresh", "warm"):
                config["process_mode"] = mode
                out = Path(tmp) / mode
                report = lab.run(config, out)
                rows = {r["variant"]: r for r in report["summary"]}
                self.assertTrue(rows["baseline"]["repeatable_observed"])
                self.assertTrue(rows["different"]["repeatable_observed"])
                self.assertFalse(rows["unstable"]["repeatable_observed"])
                self.assertEqual(len(report["records"]), 9)
                self.assertTrue((out / "report.md").exists())
                self.assertEqual(len(list((out / "artifacts").glob("*.diff"))), 6)

    def test_worker_crash_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {"backend": "demo", "repeats": 2, "python": "/nonexistent/python",
                      "prompts": [{"name": "p", "text": "test", "sha256": lab.digest(b"test")}]}
            report = lab.run(config, Path(tmp) / "run")
            self.assertEqual(report["summary"][0]["errors"], 2)

    def test_config_preserves_prompt_bytes_and_does_not_mutate_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "prompt.txt").write_bytes(b"hello\r\n")
            config = {"backend": "demo", "repeats": 2, "generate": {"seed": 42, "temperature": 0},
                      "prompts": [{"name": "p", "file": "prompt.txt"}],
                      "variants": [{"name": "temp", "generate": {"temperature": 0.6}}]}
            lab.write_json(path / "config.json", config)
            loaded = lab.read_config(path / "config.json")
            self.assertEqual(loaded["prompts"][0]["text"], "hello\r\n")
            effective = lab.effective(loaded, loaded["variants"][0])
            self.assertEqual(effective["generate"]["seed"], 42)
            self.assertEqual(loaded["generate"]["temperature"], 0)


if __name__ == "__main__":
    unittest.main()
