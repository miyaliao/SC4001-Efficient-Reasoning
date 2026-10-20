"""Unit tests for mixed benchmark; no CUDA or model import needed."""
import json
import tempfile
import unittest
from pathlib import Path

from sc4001.mixed_protocol import (
    MIXED_INSTRUCTION, case_path, case_seed, completed, export_paired,
    load_mixed, selected_questions,
)
from sc4001.pilot import atomic_json
from scripts.evaluate_mixed import boxed_final, after_thinking


class MixedProtocolTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"id": "gsmplus-000001", "benchmark": "gsm_plus", "stratum": "digit expansion",
             "question": "How many?", "answer": "12"},
            {"id": "math500-0001", "benchmark": "math_500", "stratum": "3",
             "question": "Find x", "answer": r"\frac{1}{2}"},
            {"id": "gsmplus-000002", "benchmark": "gsm_plus", "stratum": "adding operation",
             "question": "How much?", "answer": "30"},
            {"id": "math500-0002", "benchmark": "math_500", "stratum": "2",
             "question": "Coordinate?", "answer": "(1,2)"}
        ]

    def test_prompt_contract(self):
        self.assertIn(r"\boxed{YOUR_ANSWER}", MIXED_INSTRUCTION)

    def test_nested_box_and_thinking(self):
        self.assertEqual(boxed_final(
            r"<think>Final answer: \boxed{999}</think>Because: math. Final answer: \boxed{\frac{1}{2}}<|im_end|>"),
            (r"\boxed{\frac{1}{2}}", "ok"))
        self.assertEqual(boxed_final(r"Final answer: \boxed{(3,\frac{\pi}{2})}"),
                         (r"\boxed{(3,\frac{\pi}{2})}", "ok"))
        self.assertEqual(boxed_final("<think>Final answer: 3"), (None, "missing_final"))
        self.assertEqual(boxed_final(r"Final answer: \boxed{5"), (None, "unclosed_box"))
        self.assertEqual(boxed_final(r"Final answer: \boxed{5} trailing"), (None, "trailing_content"))
        self.assertEqual(boxed_final(r"Final answer: \boxed{5} Final answer: \boxed{6}"),
                         (None, "missing_or_multiple_final_labels"))
        self.assertEqual(boxed_final("We get 4 + 5 = 9."), (None, "missing_or_multiple_final_labels"))

    def test_smoke_selection_and_shards(self):
        focus = selected_questions(self.rows, 1)
        self.assertEqual([r["id"] for r in focus],
                         ["gsmplus-000001", "math500-0001"])
        first = {r["id"] for i, r in enumerate(focus) if i % 2 == 0}
        second = {r["id"] for i, r in enumerate(focus) if i % 2 == 1}
        self.assertFalse(first & second)
        self.assertEqual(first | second, {r["id"] for r in focus})

    def test_seeds_and_export_fail_closed(self):
        a = case_seed(42, "gsmplus-000001", "fast")
        self.assertEqual(a, case_seed(42, "gsmplus-000001", "fast"))
        self.assertNotEqual(a, case_seed(42, "gsmplus-000001", "slow"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "Missing completed"):
                export_paired(self.rows, root, "abc")
            self.assertFalse((root/"predictions.jsonl").exists())
            atomic_json(case_path(root, self.rows[0]["id"], "fast"), {
                "id": self.rows[0]["id"], "mode": "fast", "fingerprint": "abc",
                "status": "ok", "response": r"Final answer: \boxed{12}",
                "output_tokens": 10,
            })
            self.assertIsNotNone(completed(case_path(root, self.rows[0]["id"], "fast"),
                                           self.rows[0]["id"], "fast", "abc"))

    def test_loader_refuses_unreviewed_visual(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"mixed.jsonl"
            rows = [dict(self.rows[0], visual_review_required=True)]
            path.write_text("".join(json.dumps(r)+"\n" for r in rows))
            with self.assertRaisesRegex(ValueError, "visual"):
                load_mixed(path)
