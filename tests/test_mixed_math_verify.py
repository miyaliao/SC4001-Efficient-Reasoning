"""Small Math-Verify integration checks on real installed dependency (CPU)."""
import importlib.util
import unittest

from scripts.evaluate_mixed import boxed_final, math_verify_objects

@unittest.skipUnless(importlib.util.find_spec("math_verify"), "math-verify not installed")
class MixedMathVerifyTests(unittest.TestCase):
    def test_numeric_boxed_answer(self):
        answer, status = boxed_final(r"Reasoning: 2+2=4. Final answer: \boxed{4}")
        self.assertEqual(status, "ok")
        reference, prediction, correct = math_verify_objects("4", answer)
        self.assertTrue(reference)
        self.assertTrue(prediction)
        self.assertTrue(correct)

    def test_equivalent_fraction(self):
        answer, status = boxed_final(r"Final answer: \boxed{\frac{2}{4}}")
        self.assertEqual(status, "ok")
        _, prediction, correct = math_verify_objects(r"\frac{1}{2}", answer)
        self.assertTrue(prediction)
        self.assertTrue(correct)

    def test_wrong_numeric_answer(self):
        answer, _ = boxed_final(r"Final answer: \boxed{5}")
        _, _, correct = math_verify_objects("6", answer)
        self.assertFalse(correct)
