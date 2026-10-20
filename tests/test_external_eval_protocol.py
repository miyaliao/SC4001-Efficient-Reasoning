import unittest
from scripts.evaluate_math_verify import final_text
from scripts.prepare_mixed_eval import build, visual_status


class ExternalEvaluationTests(unittest.TestCase):
    def test_corrections_lookup_uses_id_and_mode(self):
        import json
        import tempfile
        from pathlib import Path
        from scripts.audit_math_verify_disagreements import read_corrections
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.json"
            path.write_text(json.dumps({"corrections": [
                {"id": "sample1", "mode": "fast", "correct": True},
                {"id": "sample1", "mode": "slow", "correct": False},
            ]}))
            corrections = read_corrections(path)
            self.assertTrue(corrections[("sample1", "fast")]["correct"])
            self.assertFalse(corrections[("sample1", "slow")]["correct"])

    def test_final_text(self):
        self.assertEqual(final_text("<think>Final answer: 999</think>Final answer: 6<|im_end|>"),
                         "Final answer: 6")
        self.assertEqual(final_text("<think>Final answer: 4"), "")
        self.assertIn("Final answer: 36", final_text(r"\text{Final answer: } 36"))

    def test_visual_filter_distinguishes_question_and_solution(self):
        self.assertEqual(visual_status("Find the area. [asy] draw((0,0)--(1,1)); [/asy]")[0], True)
        self.assertEqual(visual_status(r"Find the area. \begin{tikzpicture} ...")[0], True)
        self.assertEqual(visual_status("Use the diagram below to find x.")[1], True)
        self.assertEqual(visual_status("A rectangle has length 4 and width 3. Find its area."),
                         (False, False, ""))

    def test_only_question_image_is_excluded(self):
        gsm = [{"question": "q", "answer": "1", "seed_question": "seed",
                "perturbation_type": "num"}]
        math = [
            {"problem": "Find the area of a 4 by 3 rectangle.", "answer": "12",
             "level": 1, "solution": "[asy] draw(...) [/asy]"},
            {"problem": "Find x from [asy] draw(...) [/asy]", "answer": "3",
             "level": 1},
        ]
        selected = build(gsm, math, 1, 42)
        self.assertEqual(len(selected), 2)
        self.assertEqual([x for x in selected if x["benchmark"] == "math_500"][0]["answer"], "12")

    def test_sample_without_repeated_seed(self):
        gsm = [{"question": f"q{i}", "answer": str(i),
                "perturbation_type": "num", "seed_question": f"seed{i}"}
               for i in range(5)]
        gsm.append({"question":"q extra","answer":"1","perturbation_type":"arith",
                    "seed_question":"seed0"})
        math = [{"problem": f"p{i}", "answer":str(i), "level":i%2+1,
                 "type":"Algebra"} for i in range(5)]
        records = build(gsm, math, 2, 42)
        self.assertEqual(len(records), 4)
        selected = [x["seed_question"] for x in records if x["benchmark"]=="gsm_plus"]
        self.assertEqual(len(selected), len(set(selected)))
