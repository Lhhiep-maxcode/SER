import unittest

from Baseline.verify_code import (
    extract_python_code,
    infer_entry_point_from_tests,
    verify_code_completion,
)


class CodeVerifierTests(unittest.TestCase):
    def test_extracts_fenced_code(self):
        completion = "```python\ndef add_one(x):\n    return x + 1\n```"
        self.assertEqual(extract_python_code(completion, "add_one"), "def add_one(x):\n    return x + 1")

    def test_passes_valid_solution(self):
        result = verify_code_completion(
            "def add_one(x):\n    return x + 1",
            ["assert add_one(1) == 2", "assert add_one(-1) == 0"],
            entry_point="add_one",
            timeout_seconds=2.0,
        )
        self.assertTrue(result.passed)

    def test_fails_invalid_solution(self):
        result = verify_code_completion(
            "def add_one(x):\n    return x",
            ["assert add_one(1) == 2"],
            entry_point="add_one",
            timeout_seconds=2.0,
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.error_type, "assertion")

    def test_times_out(self):
        result = verify_code_completion(
            "def hang():\n    while True:\n        pass",
            ["assert hang() == 1"],
            entry_point="hang",
            timeout_seconds=0.5,
        )
        self.assertFalse(result.passed)
        self.assertTrue(result.timed_out)

    def test_infers_entry_point(self):
        self.assertEqual(infer_entry_point_from_tests(["assert square(3) == 9"]), "square")


if __name__ == "__main__":
    unittest.main()

