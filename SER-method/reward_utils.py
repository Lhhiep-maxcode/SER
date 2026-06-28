"""Reward routing for processed math/code RLVR rows."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verify_code import verify_code_completion  # noqa: E402
from verify_math import verify_math_answer  # noqa: E402


@dataclass
class RewardStats:
    math_calls: int = 0
    math_correct: int = 0
    code_calls: int = 0
    code_correct: int = 0
    code_timeouts: int = 0
    code_errors: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        logs = {
            "math_calls": float(self.math_calls),
            "code_calls": float(self.code_calls),
            "code_timeouts": float(self.code_timeouts),
        }
        if self.math_calls:
            logs["math_accuracy"] = self.math_correct / self.math_calls
        if self.code_calls:
            logs["code_accuracy"] = self.code_correct / self.code_calls
        for key, value in sorted(self.code_errors.items()):
            logs[f"code_errors/{key}"] = float(value)
        return logs


def compute_reward(
    completion: str,
    row: dict[str, Any],
    *,
    allow_code_execution: bool,
    code_timeout_seconds: float,
    stats: RewardStats,
) -> float:
    task = row.get("task")
    if task == "math":
        result = verify_math_answer(completion, str(row.get("answer") or ""))
        stats.math_calls += 1
        stats.math_correct += int(result.passed)
        return 1.0 if result.passed else 0.0

    if task == "code":
        if not allow_code_execution:
            raise RuntimeError(
                "Code reward execution is disabled. Re-run with --allow_code_execution "
                "in a trusted environment."
            )
        result = verify_code_completion(
            completion,
            row.get("tests") or [],
            entry_point=row.get("entry_point") or None,
            test_type=row.get("test_type") or "assert",
            timeout_seconds=code_timeout_seconds,
        )
        stats.code_calls += 1
        stats.code_correct += int(result.passed)
        stats.code_timeouts += int(result.timed_out)
        if result.error_type != "none":
            stats.code_errors[result.error_type] = stats.code_errors.get(result.error_type, 0) + 1
        return 1.0 if result.passed else 0.0

    raise ValueError(f"Unknown task for reward: {task!r}")

