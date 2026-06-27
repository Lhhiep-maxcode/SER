"""OpenAI-compatible trajectory critic client for SER."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


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
        temperature: float = 0.0,
        enabled: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.max_prompt_chars = max_prompt_chars
        self.temperature = temperature
        self.enabled = enabled

    def score(
        self,
        *,
        task: str,
        prompt_text: str,
        partial_completion: str,
        env_name: str,
    ) -> CriticResult:
        if not self.enabled:
            return CriticResult(0.5, "", "disabled")

        payload = {
            "model": self.model,
            "messages": self._messages(
                task=task,
                prompt_text=prompt_text,
                partial_completion=partial_completion,
                env_name=env_name,
            ),
            "temperature": self.temperature,
            "max_tokens": 64,
        }
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )

        last_error = ""
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    data = json.loads(response.read().decode("utf-8"))
                text = data["choices"][0]["message"]["content"]
                return CriticResult(clamp_score(parse_score(text)), text)
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, ValueError) as exc:
                last_error = str(exc)
                if attempt < self.max_retries:
                    time.sleep(0.5 * (attempt + 1))
        return CriticResult(0.5, "", last_error or "critic_error")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

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
                    "Return only JSON: {\"success_probability\": number_between_0_and_1}."
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
    try:
        data = json.loads(raw)
        for key in ("success_probability", "score", "probability", "p"):
            if key in data:
                return float(data[key])
    except json.JSONDecodeError:
        pass

    match = re.search(r"[-+]?(?:\d*\.\d+|\d+)", raw)
    if not match:
        raise ValueError(f"No critic score found in response: {raw!r}")
    value = float(match.group(0))
    if value > 1.0 and value <= 100.0:
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
