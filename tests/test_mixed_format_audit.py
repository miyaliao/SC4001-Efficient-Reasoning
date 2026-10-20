"""CPU tests: classification does not relax final-answer scoring."""
import unittest

from scripts.audit_mixed_smoke import classify


class MixedFormatAuditTests(unittest.TestCase):
    def test_strict_boxed(self):
        self.assertEqual(classify(r"Final answer: \boxed{\frac{1}{2}}", False), "strict_format_ok")

    def test_noncompliant_outputs_are_diagnostic_only(self):
        self.assertEqual(classify("The answer is 6.", False), "no_boxed_expression")
        self.assertEqual(classify(r"\boxed{6}", False), "box_without_final_answer_label")
        self.assertEqual(
            classify(r"Final answer: $\boxed{6}$", False),
            "boxed_expression_wrapped_in_math_delimiters",
        )
        self.assertEqual(
            classify(r"Final answer: \boxed{3}", True),
            "strict_format_ok",  # format and truncation are distinct metrics
        )
        self.assertEqual(
            classify("<think>It might be 3", True),
            "truncated_before_valid_final",
        )

    def test_multiple_answers_flagged(self):
        self.assertEqual(
            classify(r"Final answer: \boxed{2} Final answer: \boxed{3}", False),
            "multiple_boxes_or_final_labels",
        )


if __name__ == "__main__":
    unittest.main()
