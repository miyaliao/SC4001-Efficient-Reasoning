"""CPU Math-Verify integration tests for fixed canonical boxed scoring."""
import importlib.util
import unittest

from scripts.evaluate_mixed_canonical import math_grade


@unittest.skipUnless(importlib.util.find_spec("math_verify"), "math-verify missing")
class CanonicalMathVerifyTests(unittest.TestCase):
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
