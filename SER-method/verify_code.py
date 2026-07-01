"""Deterministic Python code verification for function and stdin/stdout tasks."""

from __future__ import annotations

import re
import json
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


_FENCE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_MAX_PIPE_STDIN_CHARS = 1_000_000


@dataclass(frozen=True)
class CodeVerificationResult:
    passed: bool
    timed_out: bool
    error_type: str
    stdout: str = ""
    stderr: str = ""


def extract_python_code(completion: str, entry_point: Optional[str] = None) -> str:
    """Extract executable Python from a model completion."""

    text = str(completion or "").strip()
    fenced = _FENCE_RE.findall(text)
    if fenced:
        text = fenced[-1].strip()

    if entry_point:
        match = re.search(rf"(^|\n)\s*def\s+{re.escape(entry_point)}\s*\(", text)
        if match:
            text = text[match.start() :].strip()

    return textwrap.dedent(text).strip()


def verify_code_completion(
    completion: str,
    tests: str | Iterable[str] | Iterable[Mapping[str, Any]],
    *,
    entry_point: Optional[str] = None,
    test_type: str = "assert",
    timeout_seconds: float = 5.0,
) -> CodeVerificationResult:
    candidate = extract_python_code(completion, entry_point=entry_point)

    if not candidate:
        return CodeVerificationResult(False, False, "empty_completion")

    if test_type == "stdin_stdout":
        return _verify_stdin_stdout(candidate, tests, timeout_seconds=timeout_seconds)

    test_code = _normalize_tests(tests)
    if not test_code.strip():
        return CodeVerificationResult(False, False, "empty_tests")

    script = _build_test_script(candidate, test_code)
    with tempfile.TemporaryDirectory(prefix="grpo_code_verify_") as tmpdir:
        script_path = Path(tmpdir) / "candidate_test.py"
        script_path.write_text(script, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            return CodeVerificationResult(
                False,
                True,
                "timeout",
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
            )

    if proc.returncode == 0:
        return CodeVerificationResult(True, False, "none", proc.stdout, proc.stderr)

    stderr = proc.stderr or ""
    if "AssertionError" in stderr:
        error_type = "assertion"
    elif "SyntaxError" in stderr:
        error_type = "syntax"
    else:
        error_type = "runtime"
    return CodeVerificationResult(False, False, error_type, proc.stdout, proc.stderr)


def infer_entry_point_from_tests(tests: str | Iterable[str]) -> Optional[str]:
    test_code = _normalize_tests(tests)
    match = re.search(r"assert\s+([A-Za-z_]\w*)\s*\(", test_code)
    return match.group(1) if match else None


def _verify_stdin_stdout(
    candidate: str,
    tests: str | Iterable[str] | Iterable[Mapping[str, Any]],
    *,
    timeout_seconds: float,
) -> CodeVerificationResult:
    test_cases = _normalize_io_tests(tests)
    if not test_cases:
        return CodeVerificationResult(False, False, "empty_tests")

    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    with tempfile.TemporaryDirectory(prefix="grpo_code_verify_") as tmpdir:
        script_path = Path(tmpdir) / "candidate.py"
        script_path.write_text(candidate, encoding="utf-8")
        for index, test_case in enumerate(test_cases):
            try:
                proc = _run_stdin_stdout_case(
                    script_path,
                    test_case["input"],
                    cwd=tmpdir,
                    timeout_seconds=timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                return CodeVerificationResult(
                    False,
                    True,
                    "timeout",
                    stdout=exc.stdout or "",
                    stderr=exc.stderr or "",
                )

            stdout_parts.append(proc.stdout or "")
            stderr_parts.append(proc.stderr or "")
            if proc.returncode != 0:
                error_type = "syntax" if "SyntaxError" in (proc.stderr or "") else "runtime"
                return CodeVerificationResult(
                    False,
                    False,
                    error_type,
                    stdout=proc.stdout or "",
                    stderr=proc.stderr or "",
                )
            if not _outputs_match(proc.stdout or "", test_case["output"]):
                return CodeVerificationResult(
                    False,
                    False,
                    "wrong_answer",
                    stdout=f"test_case={index}\n{proc.stdout or ''}",
                    stderr=proc.stderr or "",
                )

    return CodeVerificationResult(
        True,
        False,
        "none",
        stdout="\n".join(stdout_parts),
        stderr="\n".join(stderr_parts),
    )


def _normalize_tests(tests: str | Iterable[str]) -> str:
    if isinstance(tests, str):
        return textwrap.dedent(tests).strip()
    return "\n".join(str(item).strip() for item in tests if str(item).strip())


def _normalize_io_tests(tests: str | Iterable[str] | Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    if isinstance(tests, str):
        tests = [tests]

    normalized: list[dict[str, str]] = []
    for item in tests:
        if isinstance(item, Mapping):
            test_input = str(item.get("input", ""))
            expected_output = str(item.get("output", ""))
            normalized.append({"input": test_input, "output": expected_output})
        elif isinstance(item, str):
            try:
                parsed = json.loads(item)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, Mapping):
                test_input = str(parsed.get("input", ""))
                expected_output = str(parsed.get("output", ""))
                normalized.append({"input": test_input, "output": expected_output})
    return normalized


def _outputs_match(actual: str, expected: str) -> bool:
    actual_clean = actual.strip()
    expected_clean = expected.strip()
    if actual_clean == expected_clean:
        return True
    return actual_clean.split() == expected_clean.split()


def _run_stdin_stdout_case(
    script_path: Path,
    test_input: str,
    *,
    cwd: str,
    timeout_seconds: float,
) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(script_path)]
    if len(test_input) <= _MAX_PIPE_STDIN_CHARS:
        return subprocess.run(
            command,
            cwd=cwd,
            input=test_input,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )

    input_path = Path(cwd) / "_grpo_stdin.txt"
    input_path.write_text(test_input, encoding="utf-8")
    with input_path.open("r", encoding="utf-8") as stdin_handle:
        return subprocess.run(
            command,
            cwd=cwd,
            stdin=stdin_handle,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )


def _build_test_script(candidate: str, test_code: str) -> str:
    return "\n".join(
        [
            "import math",
            "import re",
            "import sys",
            "from collections import *",
            "from functools import *",
            "from itertools import *",
            "from typing import *",
            "",
            "def _grpo_blocked_exit(*args, **kwargs):",
            "    raise RuntimeError('sys.exit is disabled during code verification')",
            "sys.exit = _grpo_blocked_exit",
            "",
            candidate,
            "",
            test_code,
            "",
        ]
    )
