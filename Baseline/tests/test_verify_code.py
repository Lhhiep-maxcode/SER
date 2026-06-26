import unittest
import json

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

    def test_passes_stdin_stdout_solution(self):
        result = verify_code_completion(
            "a, b = map(int, input().split())\nprint(a + b)",
            [json.dumps({"input": "2 5\n", "output": "7\n"})],
            test_type="stdin_stdout",
            timeout_seconds=2.0,
        )
        self.assertTrue(result.passed)

    def test_fails_stdin_stdout_wrong_answer(self):
        result = verify_code_completion(
            "a, b = map(int, input().split())\nprint(a - b)",
            [json.dumps({"input": "2 5\n", "output": "7\n"})],
            test_type="stdin_stdout",
            timeout_seconds=2.0,
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.error_type, "wrong_answer")

    def test_passes_large_stdin_stdout_input(self):
        large_input = ("x" * 1_000_001) + "\n"
        result = verify_code_completion(
            "import sys\ndata = sys.stdin.read().rstrip('\\n')\nprint(len(data))",
            [json.dumps({"input": large_input, "output": "1000001\n"})],
            test_type="stdin_stdout",
            timeout_seconds=2.0,
        )
        self.assertTrue(result.passed)


if __name__ == "__main__":
    unittest.main()
