"""Task-routed reward functions for TRL GRPOTrainer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

try:
    from .verify_code import verify_code_completion
    from .verify_math import verify_math_answer
except ImportError:  # pragma: no cover - used when running as a script.
    from verify_code import verify_code_completion
    from verify_math import verify_math_answer


RewardFunction = Callable[..., list[Optional[float]]]


@dataclass
class RewardStats:
    math_calls: int = 0
    math_correct: int = 0
    code_calls: int = 0
    code_correct: int = 0
    code_timeouts: int = 0
    code_errors: dict[str, int] = field(default_factory=dict)

    def log_dict(self, prefix: str = "verifier") -> dict[str, float]:
        logs: dict[str, float] = {
            f"{prefix}/math_calls": float(self.math_calls),
            f"{prefix}/code_calls": float(self.code_calls),
            f"{prefix}/code_timeouts": float(self.code_timeouts),
        }
        if self.math_calls:
            logs[f"{prefix}/math_accuracy"] = self.math_correct / self.math_calls
        if self.code_calls:
            logs[f"{prefix}/code_accuracy"] = self.code_correct / self.code_calls
        for name, count in sorted(self.code_errors.items()):
            logs[f"{prefix}/code_errors/{name}"] = float(count)
        return logs


def build_reward_functions(
    *,
    allow_code_execution: bool,
    code_timeout_seconds: float = 5.0,
    stats: Optional[RewardStats] = None,
) -> tuple[list[RewardFunction], RewardStats]:
    reward_stats = stats or RewardStats()
    code_reward = make_code_reward(
        allow_code_execution=allow_code_execution,
        timeout_seconds=code_timeout_seconds,
        stats=reward_stats,
    )
    math_reward = make_math_reward(reward_stats)
    return [math_reward, code_reward], reward_stats


def make_math_reward(stats: Optional[RewardStats] = None) -> RewardFunction:
    reward_stats = stats or RewardStats()

    def math_reward(completions, task=None, answer=None, **kwargs) -> list[Optional[float]]:
        texts = [completion_to_text(item) for item in completions]
        tasks = _as_list(task, len(texts), default="math")
        answers = _as_list(answer or kwargs.get("answers"), len(texts), default="")

        rewards: list[Optional[float]] = []
        for text, sample_task, expected in zip(texts, tasks, answers):
            if sample_task != "math":
                rewards.append(None)
                continue
            result = verify_math_answer(text, str(expected))
            reward_stats.math_calls += 1
            reward_stats.math_correct += int(result.passed)
            rewards.append(1.0 if result.passed else 0.0)
        return rewards

    math_reward.__name__ = "math_reward"
    return math_reward


def make_code_reward(
    *,
    allow_code_execution: bool,
    timeout_seconds: float = 5.0,
    stats: Optional[RewardStats] = None,
) -> RewardFunction:
    reward_stats = stats or RewardStats()

    def code_reward(completions, task=None, tests=None, entry_point=None, **kwargs) -> list[Optional[float]]:
        texts = [completion_to_text(item) for item in completions]
        tasks = _as_list(task, len(texts), default="code")
        test_cases = _as_list(tests or kwargs.get("test_list"), len(texts), default=[])
        entry_points = _as_list(entry_point, len(texts), default="")

        rewards: list[Optional[float]] = []
        for text, sample_task, sample_tests, sample_entry_point in zip(
            texts, tasks, test_cases, entry_points
        ):
            if sample_task != "code":
                rewards.append(None)
                continue
            if not allow_code_execution:
                raise RuntimeError(
                    "Code reward execution is disabled. Re-run with --allow_code_execution "
                    "or set allow_code_execution: true in the config."
                )
            result = verify_code_completion(
                text,
                sample_tests,
                entry_point=sample_entry_point or None,
                timeout_seconds=timeout_seconds,
            )
            reward_stats.code_calls += 1
            reward_stats.code_correct += int(result.passed)
            if result.timed_out:
                reward_stats.code_timeouts += 1
            if result.error_type != "none":
                reward_stats.code_errors[result.error_type] = (
                    reward_stats.code_errors.get(result.error_type, 0) + 1
                )
            rewards.append(1.0 if result.passed else 0.0)
        return rewards

    code_reward.__name__ = "code_reward"
    return code_reward


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        if completion and isinstance(completion[-1], dict):
            for message in reversed(completion):
                if message.get("role") == "assistant":
                    return str(message.get("content", ""))
            return str(completion[-1].get("content", ""))
        return "\n".join(str(item) for item in completion)
    if isinstance(completion, dict):
        return str(completion.get("content", completion))
    return str(completion)


def _as_list(value: Any, length: int, *, default: Any) -> list[Any]:
    if value is None:
        return [default for _ in range(length)]
    if isinstance(value, list):
        if len(value) == length:
            return value
        if len(value) == 1:
            return value * length
    return [value for _ in range(length)]

