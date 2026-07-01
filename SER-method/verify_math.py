"""Deterministic math answer extraction and equivalence checks."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional

try:
    import sympy as sp
except Exception:  # pragma: no cover - sympy is listed as a dependency.
    sp = None


_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_BOXED_START_RE = re.compile(r"(?<![A-Za-z\\])(?:\\boxed|\\fbox|boxed)\s*\{")
_FINAL_NUMBER_RE = re.compile(
    r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/[+-]?\d+(?:\.\d+)?)?"
)


@dataclass(frozen=True)
class MathVerificationResult:
    predicted: str
    expected: str
    passed: bool


def extract_gsm8k_answer(answer: str) -> str:
    """Extract the canonical GSM8K answer after the final #### marker."""

    text = str(answer).strip()
    if "####" in text:
        return text.rsplit("####", 1)[-1].strip()
    return extract_answer(text) or text


def extract_answer(text: str) -> Optional[str]:
    """Extract a final answer from common RLVR/math response formats."""

    if text is None:
        return None

    raw = str(text).strip()
    if not raw:
        return None

    boxed = _extract_last_boxed(raw)
    if boxed:
        return boxed.strip()

    answer_tag = _ANSWER_TAG_RE.findall(raw)
    if answer_tag:
        return answer_tag[-1].strip()

    numbers = _FINAL_NUMBER_RE.findall(raw.replace("\n", " "))
    if numbers:
        return numbers[-1].strip()

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    return lines[-1] if lines else raw


def verify_math_answer(completion: str, expected_answer: str) -> MathVerificationResult:
    predicted = extract_answer(completion) or ""
    expected = extract_answer(expected_answer) or str(expected_answer)
    return MathVerificationResult(
        predicted=predicted,
        expected=expected,
        passed=answers_equivalent(predicted, expected),
    )


def answers_equivalent(predicted: str, expected: str, *, tolerance: float = 1e-6) -> bool:
    pred_norm = normalize_answer(predicted)
    exp_norm = normalize_answer(expected)

    if not pred_norm or not exp_norm:
        return pred_norm == exp_norm
    if pred_norm == exp_norm:
        return True

    pred_num = _to_float(pred_norm)
    exp_num = _to_float(exp_norm)
    if pred_num is not None and exp_num is not None:
        return math.isclose(pred_num, exp_num, rel_tol=tolerance, abs_tol=tolerance)

    pred_expr = _to_sympy(pred_norm)
    exp_expr = _to_sympy(exp_norm)
    if pred_expr is not None and exp_expr is not None:
        try:
            diff = sp.simplify(pred_expr - exp_expr)
            if diff == 0:
                return True
            return bool(abs(float(diff.evalf())) <= tolerance)
        except Exception:
            return False

    return False


def normalize_answer(answer: str) -> str:
    text = str(answer).strip()
    text = text.replace("\u2212", "-")
    text = text.replace("$", "")
    text = text.replace(",", "")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\!", "")
    text = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(".。,:; ")
    return text.lower()


def _extract_last_boxed(text: str) -> Optional[str]:
    matches: list[str] = []
    for match in _BOXED_START_RE.finditer(text):
        value, _ = _read_balanced_braces(text, match.end() - 1)
        if value is not None:
            matches.append(value)
    return matches[-1] if matches else None


def _read_balanced_braces(text: str, open_brace_idx: int) -> tuple[Optional[str], int]:
    depth = 0
    content_start = open_brace_idx + 1
    for idx in range(open_brace_idx, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[content_start:idx], idx + 1
    return None, open_brace_idx + 1


def _to_float(text: str) -> Optional[float]:
    value = text.strip()
    value = _latex_frac_to_plain(value)
    value = value.replace("^", "**")
    if value.endswith("%"):
        value = value[:-1].strip()
    try:
        return float(sp.N(sp.sympify(value))) if sp is not None else float(value)
    except Exception:
        return None


def _to_sympy(text: str):
    if sp is None:
        return None
    value = _latex_frac_to_plain(text.strip())
    value = value.replace("^", "**")
    if value.endswith("%"):
        value = value[:-1].strip()
    try:
        return sp.sympify(value)
    except Exception:
        return None


def _latex_frac_to_plain(text: str) -> str:
    frac_re = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
    previous = None
    current = text
    while previous != current:
        previous = current
        current = frac_re.sub(r"(\1)/(\2)", current)
    return current
