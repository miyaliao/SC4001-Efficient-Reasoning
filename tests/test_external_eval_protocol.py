import unittest
from scripts.evaluate_math_verify import final_text
from scripts.prepare_mixed_eval import build


class ExternalEvaluationTests(unittest.TestCase):
    def test_final_text(self):
        self.assertEqual(final_text("<think>Final answer: 999</think>Final answer: 6<|im_end|>"),
                         "Final answer: 6")
        self.assertEqual(final_text("<think>Final answer: 4"), "")
        self.assertIn("Final answer: 36", final_text(r"\text{Final answer: } 36"))

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
