import unittest

from sc4001.answers import gold_answer, predicted_answer
from sc4001.metrics import evaluate_pairs


class EvaluationTests(unittest.TestCase):
    def test_gold(self):
        self.assertEqual(gold_answer("Compute.\n#### 1,234.5"), 1234.5)

    def test_boxed(self):
        self.assertEqual(predicted_answer(r"The answer is \boxed{42}"), 42)

    def test_ambiguous(self):
        self.assertIsNone(predicted_answer("First 12 then 13 finally 18"))

    def test_final(self):
        self.assertEqual(predicted_answer("Final answer: -3.5"), -3.5)

    def test_paired_counts(self):
        samples = [
            {"id": "a", "answer": "steps\n#### 4"},
            {"id": "b", "answer": "steps\n#### 7"},
        ]
        predictions = [
            {"id": "a", "mode": "fast", "response": "Answer: 4", "output_tokens": 10},
            {"id": "a", "mode": "slow", "response": "Answer: 4", "output_tokens": 30},
            {"id": "b", "mode": "fast", "response": "Answer: 6", "output_tokens": 10},
            {"id": "b", "mode": "slow", "response": "Answer: 7", "output_tokens": 30},
        ]
        result = evaluate_pairs(samples, predictions)
        self.assertEqual(result["fast_accuracy"], 0.5)
        self.assertEqual(result["slow_accuracy"], 1.0)
        self.assertEqual(result["paired_counts"]["slow_only"], 1)
        self.assertEqual(result["mean_output_tokens"]["slow"], 30)

    def test_missing_raises(self):
        with self.assertRaises(ValueError):
            evaluate_pairs(
                [{"id": "a", "answer": "#### 1"}],
                [{"id": "a", "mode": "fast", "response": "Answer: 1"}],
            )

    def test_duplicate_raises(self):
        with self.assertRaises(ValueError):
            evaluate_pairs(
                [{"id": "a", "answer": "#### 1"}],
                [
                    {"id": "a", "mode": "fast", "response": "Answer: 1"},
                    {"id": "a", "mode": "fast", "response": "Answer: 1"},
                    {"id": "a", "mode": "slow", "response": "Answer: 1"},
                ],
            )


if __name__ == "__main__":
    unittest.main()
