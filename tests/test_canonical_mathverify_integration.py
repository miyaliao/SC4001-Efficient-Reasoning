"""CPU Math-Verify integration tests for fixed canonical boxed scoring."""
import importlib.util
import unittest

from scripts.evaluate_mixed_canonical import math_grade
from sc4001.boxed_scoring import canonical_box


@unittest.skipUnless(importlib.util.find_spec("math_verify"), "math-verify missing")
class CanonicalMathVerifyTests(unittest.TestCase):
    def test_real_colonless_heading_wrong_answer_stays_wrong(self):
        response = (
            "<think>Reasoning done.</think>\n"
            "### Final Answer\n"
            "$$\n"
            + r"\boxed{12}" + "\n$$<|im_end|>"
        )
        extracted, status, wrapper = canonical_box(response)
        self.assertEqual((status, wrapper), ("ok", "display_dollar"))
        self.assertEqual(extracted, r"\boxed{12}")
        correct, parsed, _, _ = math_grade(
            "18", extracted, "How much does six pounds cost?"
        )
        self.assertTrue(parsed)
        self.assertFalse(correct)

    def test_plain_numeric(self):
        correct, parsed, _, special = math_grade("5200", r"\boxed{5200}", "How many?")
        self.assertTrue(correct)
        self.assertTrue(parsed)
        self.assertFalse(special)

    def test_numeric_wrong(self):
        correct, parsed, _, _ = math_grade("5200", r"\boxed{5000}", "How many?")
        self.assertTrue(parsed)
        self.assertFalse(correct)

    def test_fraction(self):
        correct, parsed, _, _ = math_grade(r"\frac{1}{2}", r"\boxed{\frac{2}{4}}",
                                           "Evaluate the expression")
        self.assertTrue(parsed)
        self.assertTrue(correct)

    def test_percent_as_percentage_points(self):
        correct, parsed, _, normalized = math_grade("36", r"\boxed{36\%}",
                                                   "What percentage of puppies have spots?")
        self.assertTrue(parsed)
        self.assertTrue(normalized)
        self.assertTrue(correct)

    def test_percent_not_count(self):
        correct, parsed, _, normalized = math_grade("36", r"\boxed{36\%}",
                                                   "How many puppies have spots?")
        self.assertFalse(normalized)
        self.assertFalse(correct)


if __name__ == "__main__":
    unittest.main()
