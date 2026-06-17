"""Deterministic Python code verification for MBPP and HumanEval-style tasks."""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


_FENCE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


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
    tests: str | Iterable[str],
    *,
    entry_point: Optional[str] = None,
    timeout_seconds: float = 5.0,
) -> CodeVerificationResult:
    candidate = extract_python_code(completion, entry_point=entry_point)
    test_code = _normalize_tests(tests)

    if not candidate:
        return CodeVerificationResult(False, False, "empty_completion")
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


def _normalize_tests(tests: str | Iterable[str]) -> str:
    if isinstance(tests, str):
        return textwrap.dedent(tests).strip()
    return "\n".join(str(item).strip() for item in tests if str(item).strip())


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
