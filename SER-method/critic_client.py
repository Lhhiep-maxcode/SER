"""OpenAI-compatible trajectory critic client for SER."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any

try:
    from openai import AsyncOpenAI, OpenAIError
except ImportError:  # pragma: no cover - handled when enabled critic is built
    AsyncOpenAI = None
    OpenAIError = Exception


_BOXED_SCORE_RE = re.compile(r"(?:\\boxed|boxed)\s*\{\s*([^{}]+?)\s*\}")
_NUMBER_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)")
_FRACTION_RE = re.compile(r"([-+]?(?:\d*\.\d+|\d+))\s*/\s*([-+]?(?:\d*\.\d+|\d+))")


@dataclass(frozen=True)
class CriticResult:
    score: float
    raw_text: str
    error: str = ""


class CriticClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        max_prompt_chars: int = 6000,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        enabled: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.max_prompt_chars = max_prompt_chars
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.enabled = enabled
        self.client = self._build_client() if enabled else None

    def score(
        self,
        requests: list[dict[str, str]],
        *,
        max_concurrency: int | None = None,
    ) -> list[CriticResult]:
        return asyncio.run(self.score_async(requests, max_concurrency=max_concurrency))

    async def score_one(
        self,
        *,
        task: str,
        prompt_text: str,
        partial_completion: str,
        env_name: str,
    ) -> CriticResult:
        if not self.enabled:
            return CriticResult(0.5, "", "disabled")
        if self.client is None:
            return CriticResult(0.5, "", "openai_client_unavailable")

        last_error = ""
        for attempt in range(self.max_retries + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=self._messages(
                        task=task,
                        prompt_text=prompt_text,
                        partial_completion=partial_completion,
                        env_name=env_name,
                    ),
                    temperature=self.temperature,
                    max_tokens=self.max_new_tokens,
                )
                text = response.choices[0].message.content or ""
                return CriticResult(clamp_score(parse_score(text)), text)
            except (OpenAIError, TimeoutError, AttributeError, IndexError, KeyError, json.JSONDecodeError, ValueError) as exc:
                last_error = str(exc)
                if attempt < self.max_retries:
                    await asyncio.sleep(0.5 * (attempt + 1))
        return CriticResult(0.5, "", last_error or "critic_error")

    async def score_async(
        self,
        requests: list[dict[str, str]],
        *,
        max_concurrency: int | None = None,
    ) -> list[CriticResult]:
        if not requests:
            return []
        if not self.enabled:
            return [CriticResult(0.5, "", "disabled") for _ in requests]
        concurrency = max(1, min(int(max_concurrency or 1), len(requests)))
        semaphore = asyncio.Semaphore(concurrency)

        async def run_one(request: dict[str, str]) -> CriticResult:
            async with semaphore:
                try:
                    return await self.score_one(**request)
                except Exception as exc:  # pragma: no cover - defensive async boundary
                    return CriticResult(0.5, "", str(exc))

        return list(await asyncio.gather(*(run_one(request) for request in requests)))

    def _build_client(self):
        if AsyncOpenAI is None:
            raise RuntimeError("The openai package is required for CriticClient. Install it with `pip install openai`.")
        return AsyncOpenAI(
            base_url=openai_base_url(self.base_url),
            api_key=self.api_key or "EMPTY",
            timeout=self.timeout_seconds,
            max_retries=0,
        )

    def _messages(self, *, task: str, prompt_text: str, partial_completion: str, env_name: str):
        prompt_text = truncate_middle(prompt_text, self.max_prompt_chars // 2)
        partial_completion = truncate_middle(partial_completion, self.max_prompt_chars // 2)
        return [
            {
                "role": "system",
                "content": (
                    "You are a strict RLVR trajectory critic. Estimate the probability that "
                    "the partially generated assistant trajectory will eventually pass the "
                    "task verifier if generation is allowed to continue. Do not solve the task. "
                    "Return only the score as a number between 0 and 1 in this format: "
                    "\\boxed{...a_number_between_0_and_1...}. "
                    "0.0 means definitely fail if the partial assistant trajectory contains errors. "
                    "1.0 means you are sure that the trajectory will definitely pass."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Environment: {env_name}\n"
                    f"Task type: {task}\n\n"
                    "Prompt:\n"
                    f"{prompt_text}\n\n"
                    "Partial assistant trajectory:\n"
                    f"{partial_completion}\n\n"
                    "Return the probability that the final completed trajectory will pass."
                ),
            },
        ]


def parse_score(text: str) -> float:
    raw = str(text or "").strip()

    boxed_matches = _BOXED_SCORE_RE.findall(raw)
    if boxed_matches:
        return parse_numeric_score(boxed_matches[-1])

    try:
        data = json.loads(raw)
        for key in ("success_probability", "score", "probability", "p"):
            if key in data:
                return parse_numeric_score(str(data[key]))
    except json.JSONDecodeError:
        pass

    return parse_numeric_score(raw)


def openai_base_url(base_url: str) -> str:
    value = str(base_url or "").rstrip("/")
    if value.endswith("/v1"):
        return value
    return f"{value}/v1"


def run_async(coro_factory):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro_factory())
    raise RuntimeError("CriticClient synchronous methods cannot be called from a running event loop. Use the async methods instead.")


def parse_numeric_score(raw: str) -> float:
    value_text = str(raw or "").strip()

    fraction = _FRACTION_RE.search(value_text)
    if fraction:
        numerator = float(fraction.group(1))
        denominator = float(fraction.group(2))
        if denominator == 0:
            raise ValueError(f"Invalid zero-denominator critic score: {raw!r}")
        return numerator / denominator

    match = _NUMBER_RE.search(value_text)
    if not match:
        raise ValueError(f"No critic score found in response: {raw!r}")
    value = float(match.group(0))
    if "%" in value_text or (value > 1.0 and value <= 100.0):
        value /= 100.0
    return value


def clamp_score(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def truncate_middle(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = max(1, limit // 2)
    omitted = len(text) - (half * 2)
    return f"{text[:half]}\n... <truncated {omitted} chars> ...\n{text[-half:]}"
