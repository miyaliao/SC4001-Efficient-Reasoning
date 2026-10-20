"""Frozen canonical Boxed format: standard display wrappers are permitted."""
import unittest

from sc4001.boxed_scoring import canonical_box, numeric_percent_points


class CanonicalBoxedTests(unittest.TestCase):
    def test_wrappers_match_contract(self):
        examples = [
            (r"Final answer: \boxed{5200}", "none"),
            (r"Final answer: $\boxed{5200}$", "inline_dollar"),
            ("### Final Answer:\n$$\n"+r"\boxed{5200}"+"\n$$", "display_dollar"),
            ("### Final Answer\n$$\n"+r"\boxed{5200}"+"\n$$", "display_dollar"),
            ("### Final Answer\n\n$$\n"+r"\boxed{5200}"+"\n$$", "display_dollar"),
            ("**Final Answer**\n$$\n"+r"\boxed{5200}"+"\n$$", "display_dollar"),
            ("**Final answer:** "+r"\boxed{5200}", "none"),
            ("**Final answer**: "+r"\boxed{5200}", "none"),
            ("Final answer: "+r"\(\boxed{5200}\)", "inline_paren"),
            ("Final answer: "+r"\[\boxed{5200}\]", "display_bracket"),
        ]
        for response, wrap in examples:
            with self.subTest(response=response):
                box, status, kind = canonical_box(response+"<|im_end|>")
                self.assertEqual(box, r"\boxed{5200}")
                self.assertEqual((status, kind), ("ok", wrap))

    def test_symbolic_nested_latex(self):
        response = ("<think>Final answer: "+r"\boxed{999}"+"</think>\n"
                    "### Final Answer:\n$$\n"
                    +r"\boxed{\begin{pmatrix} -\frac{1}{3} \\ \frac{2}{3} \\ \frac{5}{3} \end{pmatrix}}"+"\n$$")
        box, status, kind = canonical_box(response)
        self.assertTrue(box.startswith(r"\boxed{\begin{pmatrix}"))
        self.assertTrue(box.endswith("}}"))
        self.assertEqual(status, "ok")
        self.assertEqual(kind, "display_dollar")

    def test_no_rescue_of_missing_final_box(self):
        cases = [
            "The answer is 24.",
            r"\boxed{24}",
            "<think>Final answer: "+r"\boxed{24}",
            "Final answer: 24",
            r"Final answer: \boxed{24} and something else",
            r"Final answer: \boxed{24} Final answer: \boxed{25}",
            r"Final answer: \boxed{24",
            "Final answer: $"+r"\boxed{24}",
            "### Final Answer\n$$\n12\n$$",
            "### Final Answer\n$$\n"+r"\boxed{12}"+"\n$ extra",
            "### Final Answer\n$$\n"+r"\boxed{12}"+"\n$$\nFinal answer: "+r"\boxed{13}",
            "Final answering: "+r"\boxed{12}",
        ]
        for case in cases:
            with self.subTest(case=case):
                self.assertIsNone(canonical_box(case)[0])

    def test_percentage_points_only_for_percent_question(self):
        box = r"\boxed{36\%}"
        self.assertEqual(numeric_percent_points("36", box, "What percentage are spotted?"),
                         (r"\boxed{36}", True))
        self.assertEqual(numeric_percent_points("36", box, "How many puppies are spotted?"),
                         (box, False))
        self.assertEqual(numeric_percent_points(r"\frac{1}{2}", box, "What percentage?"),
                         (box, False))


if __name__ == "__main__":
    unittest.main()
