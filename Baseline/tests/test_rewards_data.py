import unittest

from Baseline.data import (
    build_mixed_dataset_from_records,
    weighted_mix_records,
)
from Baseline.rewards import make_code_reward, make_math_reward


MATH_RECORDS = [
    {
        "id": "m1",
        "source": "unit",
        "task": "math",
        "prompt": [{"role": "user", "content": "2+2?"}],
        "answer": "4",
        "tests": [],
        "entry_point": "",
    }
]

CODE_RECORDS = [
    {
        "id": "c1",
        "source": "unit",
        "task": "code",
        "prompt": [{"role": "user", "content": "write add_one"}],
        "answer": "",
        "tests": ["assert add_one(1) == 2"],
        "entry_point": "add_one",
    }
]


class RewardsAndDataTests(unittest.TestCase):
    def test_weighted_mix_is_deterministic(self):
        first = weighted_mix_records(
            {"math": MATH_RECORDS, "code": CODE_RECORDS},
            {"math": 0.5, "code": 0.5},
            max_samples=6,
            seed=123,
        )
        second = weighted_mix_records(
            {"math": MATH_RECORDS, "code": CODE_RECORDS},
            {"math": 0.5, "code": 0.5},
            max_samples=6,
            seed=123,
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)
        self.assertTrue({record["task"] for record in first}.issubset({"math", "code"}))

    def test_dataset_schema(self):
        dataset = build_mixed_dataset_from_records(
            MATH_RECORDS,
            CODE_RECORDS,
            task_weights={"math": 0.5, "code": 0.5},
            max_samples=4,
            seed=5,
        )
        self.assertEqual(len(dataset), 4)
        for column in ("prompt", "task", "answer", "tests", "entry_point"):
            self.assertIn(column, dataset.column_names)

    def test_math_reward_routes_by_task(self):
        reward = make_math_reward()
        self.assertEqual(reward(["\\boxed{4}"], task=["math"], answer=["4"]), [1.0])
        self.assertEqual(reward(["def f(): pass"], task=["code"], answer=[""]), [None])

    def test_code_reward_routes_by_task_and_requires_permission(self):
        blocked_reward = make_code_reward(allow_code_execution=False)
        self.assertEqual(blocked_reward(["\\boxed{4}"], task=["math"], tests=[[]]), [None])
        with self.assertRaises(RuntimeError):
            blocked_reward(
                ["def add_one(x):\n    return x + 1"],
                task=["code"],
                tests=[["assert add_one(1) == 2"]],
                entry_point=["add_one"],
            )

    def test_code_reward_executes_when_allowed(self):
        reward = make_code_reward(allow_code_execution=True, timeout_seconds=2.0)
        self.assertEqual(
            reward(
                ["def add_one(x):\n    return x + 1"],
                task=["code"],
                tests=[["assert add_one(1) == 2"]],
                entry_point=["add_one"],
            ),
            [1.0],
        )


if __name__ == "__main__":
    unittest.main()

