"""CPU-only tests for small real-smoke result summarization.

All predictions below are synthetic fixtures, not actual Qwen runs.
"""
import tempfile
import unittest
from pathlib import Path

from sc4001.pilot import atomic_json, case_path
from scripts.evaluate_smoke import build_smoke_report


class SmokeReportTests(unittest.TestCase):
    def test_paired_smoke_metrics(self):
        examples = [
            {"id": "gsm8k-train-000001", "question": "What is 4?", "answer": "#### 4"},
            {"id": "gsm8k-train-000002", "question": "What is 7?", "answer": "#### 7"},
        ]
        outputs = [
            ("gsm8k-train-000001", "fast",  "Final answer: 4", 10),
            ("gsm8k-train-000001", "slow",  "<think>4</think> Final answer: 4", 30),
            ("gsm8k-train-000002", "fast",  "Final answer: 8", 10),
            ("gsm8k-train-000002", "slow",  "<think>7</think> Final answer: 7", 30),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for sample_id, mode, response, tokens in outputs:
                atomic_json(case_path(root, sample_id, mode), {
                    "id": sample_id, "mode": mode, "response": response,
                    "output_tokens": tokens, "latency_sec": tokens / 10,
                    "status": "ok", "fingerprint": "fixture", "truncated": False,
                })
            report = build_smoke_report(examples, root, "fixture")
        self.assertEqual(report["metrics"]["fast_accuracy"], 0.5)
        self.assertEqual(report["metrics"]["slow_accuracy"], 1.0)
        self.assertEqual(report["metrics"]["mean_output_tokens"]["fast"], 10)
        self.assertEqual(report["metrics"]["mean_output_tokens"]["slow"], 30)
        self.assertEqual(report["metrics"]["paired_counts"]["slow_only"], 1)
        self.assertEqual(len(report["per_case"]), 2)

    def test_incomplete_pair_is_refused(self):
        examples = [{"id": "gsm8k-train-000001", "question": "1?", "answer": "#### 1"}]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Missing fast output"):
                build_smoke_report(examples, Path(directory), "fixture")


if __name__ == "__main__":
    unittest.main()
