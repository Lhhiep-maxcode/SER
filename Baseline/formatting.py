"""Shared prompt formatting for baseline datasets."""

from __future__ import annotations

from typing import Iterable


MATH_SYSTEM_PROMPT = (
    "You are a careful mathematical reasoner. Solve the problem step by step, "
    "then put the final answer in \\boxed{...}."
)

CODE_SYSTEM_PROMPT = (
    "You are a precise Python programmer. Return only valid Python code that "
    "solves the requested task."
)


def math_prompt(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": MATH_SYSTEM_PROMPT},
        {"role": "user", "content": str(question).strip()},
    ]


def code_prompt(
    problem: str,
    tests: Iterable[str],
    *,
    entry_point: str = "",
    test_type: str = "assert",
    starter_code: str = "",
) -> list[dict[str, str]]:
    if test_type == "stdin_stdout":
        starter = f"\n\nStarter code:\n{starter_code}" if starter_code else ""
        user = (
            f"{str(problem).strip()}{starter}\n\n"
            "Return a complete Python 3 program that reads from standard input "
            "and writes the final answer to standard output. Do not include Markdown fences."
        )
    else:
        test_text = "\n".join(str(test) for test in tests)
        signature = f"\nFunction name: {entry_point}" if entry_point else ""
        starter = f"\nStarter code:\n{starter_code}" if starter_code else ""
        user = (
            f"{str(problem).strip()}{signature}{starter}\n\n"
            f"Reference tests:\n{test_text}\n\n"
            "Return a complete Python solution. Do not include Markdown fences."
        )

    return [
        {"role": "system", "content": CODE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]

