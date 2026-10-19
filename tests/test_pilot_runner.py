import json
import tempfile
import unittest
from pathlib import Path

from sc4001.pilot import (
    atomic_json, case_path, check_manifest, export_results,
    case_seed, load_examples, plan,
)


class PilotRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.examples = [
            {"id": "gsm8k-train-000001", "question": "A\u2028B", "answer": "#### 1"},
            {"id": "gsm8k-train-000002", "question": "What is 2?", "answer": "#### 2"},
        ]
        self.fp = "f" * 64

    def test_sampling_seeds_stable_per_case(self):
        sample_id = self.examples[0]["id"]
        seed_fast = case_seed(42, sample_id, "fast")
        seed_slow = case_seed(42, sample_id, "slow")
        self.assertEqual(seed_fast, case_seed(42, sample_id, "fast"))
        self.assertNotEqual(seed_fast, seed_slow)
        self.assertNotEqual(seed_fast, case_seed(42, self.examples[1]["id"], "fast"))
        with self.assertRaises(ValueError):
            case_seed(42, sample_id, "invalid")

    def test_two_shard_partition(self):
        examples = self.examples + [
            {"id": "gsm8k-train-000003", "question": "3?", "answer": "#### 3"},
        ]
        first = [e for i, e in enumerate(examples) if i % 2 == 0]
        second = [e for i, e in enumerate(examples) if i % 2 == 1]
        self.assertEqual(len(first) + len(second), len(examples))
        self.assertFalse({e["id"] for e in first} & {e["id"] for e in second})
        self.assertEqual({e["id"] for e in first + second}, {e["id"] for e in examples})

    def test_unicode_line(self):
        filename = self.base / "data.jsonl"
        filename.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in self.examples))
        self.assertEqual(load_examples(filename), self.examples)

    def test_dry_run_no_side_effects(self):
        out = self.base / "results"
        selection, pending, skipped = plan(self.examples, out, self.fp)
        self.assertEqual(len(selection), 2)
        self.assertEqual(len(pending), 4)
        self.assertEqual(skipped, 0)
        self.assertFalse(out.exists())

    def test_resume_skips_completed(self):
        out = self.base / "results"
        path = case_path(out, self.examples[0]["id"], "fast")
        atomic_json(path, {"id": self.examples[0]["id"], "mode": "fast", "status": "ok",
                           "fingerprint": self.fp, "response": "Final answer: 1", "output_tokens": 5})
        _, pending, skipped = plan(self.examples, out, self.fp)
        self.assertEqual((skipped, len(pending)), (1, 3))

    def test_mismatched_checkpoint_fails(self):
        out = self.base / "results"
        path = case_path(out, self.examples[0]["id"], "fast")
        atomic_json(path, {"id": self.examples[0]["id"], "mode": "fast", "status": "ok",
                           "fingerprint": "wrong", "response": "Final answer: 1", "output_tokens": 5})
        with self.assertRaises(ValueError):
            plan(self.examples, out, self.fp)

    def test_manifest_rejects_mixed_runs(self):
        out = self.base / "results"
        check_manifest(out, {"fingerprint": self.fp, "settings": {}}, create=True)
        check_manifest(out, {"fingerprint": self.fp, "settings": {}}, create=False)
        with self.assertRaises(ValueError):
            check_manifest(out, {"fingerprint": "different"}, create=False)

    def test_export_refuses_partial(self):
        out = self.base / "results"
        atomic_json(case_path(out, self.examples[0]["id"], "fast"), {
            "id": self.examples[0]["id"], "mode": "fast", "status": "ok",
            "fingerprint": self.fp, "response": "Final answer: 1", "output_tokens": 3,
        })
        with self.assertRaises(ValueError):
            export_results(self.examples, out, self.fp, complete_only=True)
        self.assertFalse((out / "predictions.jsonl").exists())

    def test_unsafe_path_rejected(self):
        with self.assertRaises(ValueError):
            case_path(self.base, "../unsafe", "fast")
        with self.assertRaises(ValueError):
            case_path(self.base, self.examples[0]["id"], "invalid")


if __name__ == "__main__":
    unittest.main()
