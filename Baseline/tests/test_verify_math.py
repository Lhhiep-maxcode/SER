import unittest

from Baseline.verify_math import (
    _extract_last_boxed,
    answers_equivalent,
    extract_answer,
    extract_gsm8k_answer,
    verify_math_answer,
)


class MathVerifierTests(unittest.TestCase):
    def test_extracts_boxed_answer(self):
        self.assertEqual(extract_answer("Reasoning... final is \\boxed{42}."), "42")

    def test_extracts_boxed_answer_with_spaces(self):
        self.assertEqual(extract_answer("Final: \\boxed   { 42 }"), "42")

    def test_extracts_last_boxed_answer_in_text_order(self):
        self.assertEqual(extract_answer("First \\fbox{1}, final \\boxed{2}."), "2")
        self.assertEqual(extract_answer("First \\boxed{1}, final \\fbox{2}."), "2")

    def test_extracts_plain_boxed_answer(self):
        self.assertEqual(extract_answer("The answer is boxed{9}."), "9")

    def test_extracts_nested_boxed_answer(self):
        self.assertEqual(extract_answer("Final answer: \\boxed{\\frac{1}{2}}."), "\\frac{1}{2}")

    def test_ignores_malformed_boxed_answer(self):
        self.assertIsNone(_extract_last_boxed("Reasoning \\boxed{not closed"))

    def test_ignores_boxed_inside_words(self):
        self.assertIsNone(_extract_last_boxed("This is unboxed{wrong} text"))

    def test_extracts_answer_tag(self):
        self.assertEqual(extract_answer("<answer> 7 </answer>"), "7")

    def test_extracts_gsm8k_marker(self):
        self.assertEqual(extract_gsm8k_answer("work\n#### 1,234"), "1,234")

    def test_numeric_equivalence(self):
        self.assertTrue(answers_equivalent("\\frac{1}{2}", "0.5"))
        self.assertTrue(answers_equivalent("1,000", "1000"))

    def test_verification_failure(self):
        result = verify_math_answer("The answer is \\boxed{5}", "6")
        self.assertFalse(result.passed)


if __name__ == "__main__":
    unittest.main()
