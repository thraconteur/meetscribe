"""Thin, robust wrapper around OpenAI-compatible chat + transcription endpoints.

Handles: rate limits (honours Retry-After), transient network/5xx errors with
exponential back-off, providers that do not support JSON mode, and models that
wrap JSON in prose or code fences.
"""
from __future__ import annotations

import json
import threading
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import ModelEndpoint
from .schema import PipelineError

REASONING_MODEL_PATTERN = re.compile(r"(gpt-oss|^o\d|/o\d|gpt-5|deepseek-r1|qwen3)", re.I)


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0

    def add(self, other: "Usage") -> None:
        self.calls += other.calls
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.seconds += other.seconds


@dataclass
class UsageTracker:
    by_role: dict[str, Usage] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def record(self, role: str, prompt: int, completion: int, seconds: float) -> None:
        with self._lock:
            u = self.by_role.setdefault(role, Usage())
            u.add(Usage(1, prompt, completion, seconds))

    def as_dict(self) -> dict:
        return {k: vars(v) for k, v in self.by_role.items()}


def make_client(endpoint: ModelEndpoint, timeout: float):
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise PipelineError("The 'openai' package is not installed. Run: pip install -r requirements.txt") from exc
    if not endpoint.api_key:
        raise PipelineError(
            f"No API key configured for the {endpoint.role or 'model'} stage ({endpoint.model}). "
            "Set GROQ_API_KEY in your .env file (free key: https://console.groq.com/keys) or the "
            f"{endpoint.role.upper()}_API_KEY variable.",
            stage=endpoint.role,
        )
    return OpenAI(api_key=endpoint.api_key, base_url=endpoint.base_url, timeout=timeout, max_retries=0)


def _retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    for key in ("retry-after", "x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        value = headers.get(key) if hasattr(headers, "get") else None
        if not value:
            continue
        m = re.match(r"^\s*(?:(\d+)m)?\s*([\d.]+)?s?\s*$", str(value))
        try:
            if m and (m.group(1) or m.group(2)):
                return float(m.group(1) or 0) * 60 + float(m.group(2) or 0)
            return float(value)
        except ValueError:
            continue
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", str(exc))
    if m:
        return float(m.group(1) or 0) * 60 + float(m.group(2))
    return None


def call_with_retries(fn: Callable[[], Any], endpoint: ModelEndpoint, max_retries: int, log: Callable[[str], None] | None = None):
    """Run ``fn`` retrying transient failures. Converts permanent failures into PipelineError."""
    import openai

    attempt = 0
    while True:
        try:
            return fn()
        except openai.AuthenticationError as exc:
            raise PipelineError(
                f"The API key for {endpoint.label} was rejected (401). Check the key in your .env file.",
                endpoint.role,
            ) from exc
        except openai.NotFoundError as exc:
            raise PipelineError(
                f"Model '{endpoint.model}' was not found at {endpoint.base_url or 'OpenAI'}. "
                "Check the model name in your .env file.",
                endpoint.role,
            ) from exc
        except openai.PermissionDeniedError as exc:
            raise PipelineError(f"Access to {endpoint.label} was denied: {exc}", endpoint.role) from exc
        except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError) as exc:
            attempt += 1
            if attempt > max_retries:
                kind = "rate limit" if isinstance(exc, openai.RateLimitError) else "connection problem"
                raise PipelineError(
                    f"{endpoint.label} kept failing ({kind}) after {max_retries} retries: {str(exc)[:240]}",
                    endpoint.role,
                ) from exc
            wait = _retry_after_seconds(exc)
            if wait is None:
                wait = min(60.0, 2 ** attempt + random.random())
            wait = min(max(wait, 1.0), 90.0)
            if log:
                log(f"{endpoint.model}: {type(exc).__name__}, retrying in {wait:.0f}s (attempt {attempt}/{max_retries})")
            time.sleep(wait)
        except openai.APIStatusError as exc:
            status = getattr(exc, "status_code", None)
            if status == 413:
                raise RequestTooLarge(str(exc)) from exc
            if status and status >= 500 and attempt < max_retries:
                attempt += 1
                time.sleep(min(60.0, 2 ** attempt))
                continue
            raise


class RequestTooLarge(Exception):
    """The provider refused the request because of its size (e.g. Groq per-minute token caps)."""


def extract_json(text: str) -> Any:
    """Parse JSON from a model reply that may contain fences, prose or trailing commas."""
    if text is None:
        raise ValueError("empty reply")
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = min([i for i in (text.find("{"), text.find("[")) if i != -1], default=-1)
    if start == -1:
        raise ValueError("no JSON object found")
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    end = text.rfind(closer)
    candidate = text[start : end + 1]
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)  # trailing commas
    return json.loads(candidate)


class ChatModel:
    """A chat model that returns parsed JSON objects."""

    def __init__(self, endpoint: ModelEndpoint, timeout: float = 180.0, max_retries: int = 5,
                 tracker: UsageTracker | None = None, log: Callable[[str], None] | None = None,
                 reasoning_effort: str | None = None):
        self.endpoint = endpoint
        self.client = make_client(endpoint, timeout)
        self.max_retries = max_retries
        self.tracker = tracker or UsageTracker()
        self.log = log
        self.reasoning_effort = reasoning_effort if REASONING_MODEL_PATTERN.search(endpoint.model) else None
        self._json_mode = True

    def json(self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 8000) -> Any:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        last_error: Exception | None = None
        for parse_attempt in range(3):
            reply, finish = self._complete_fitting(messages, temperature, max_tokens)
            try:
                return extract_json(reply)
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                if finish == "length":
                    # Out of tokens (often a reasoning model thinking too long): think less, allow more output.
                    if self.reasoning_effort in ("high", "medium"):
                        self.reasoning_effort = "low" if self.reasoning_effort == "medium" else "medium"
                    max_tokens = int(max_tokens * 1.5)
                if self.log:
                    self.log(f"{self.endpoint.model}: reply was not valid JSON, asking again")
                if not (reply or "").strip():
                    continue
                messages = messages[:2] + [
                    {"role": "assistant", "content": (reply or "")[:4000]},
                    {"role": "user", "content": "That was not valid JSON. Reply again with ONLY the JSON object, "
                                                "no prose and no code fences."},
                ]
        raise PipelineError(
            f"{self.endpoint.label} did not return valid JSON after 3 attempts ({last_error}).", self.endpoint.role
        )

    def _complete_fitting(self, messages: list[dict], temperature: float, max_tokens: int) -> tuple[str, str]:
        """Call the model; if the provider says the request exceeds its per-minute token cap
        (Groq counts max_tokens toward it), shrink max_tokens to fit and try again."""
        for _ in range(3):
            try:
                return self._complete(messages, temperature, max_tokens)
            except RequestTooLarge as exc:
                m = re.search(r"limit\s*(\d+)\D+requested\s*(\d+)", str(exc), re.I)
                if not m:
                    raise
                limit, requested = int(m.group(1)), int(m.group(2))
                prompt_tokens = requested - max_tokens
                new_max = limit - prompt_tokens - 300
                if new_max < 1500 or new_max >= max_tokens:
                    raise
                if self.log:
                    self.log(f"{self.endpoint.model}: lowering max output tokens to {new_max} to fit provider limits")
                max_tokens = new_max
        raise RequestTooLarge("request still too large after shrinking output budget")

    def _complete(self, messages: list[dict], temperature: float, max_tokens: int) -> tuple[str, str]:
        import openai

        def request():
            kwargs: dict[str, Any] = dict(model=self.endpoint.model, messages=messages, max_tokens=max_tokens)
            if self.reasoning_effort:
                kwargs["reasoning_effort"] = self.reasoning_effort
            else:
                kwargs["temperature"] = temperature
            if self._json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            t0 = time.time()
            try:
                resp = self.client.chat.completions.create(**kwargs)
            except openai.BadRequestError as exc:
                msg = str(exc).lower()
                if self._json_mode and ("response_format" in msg or "json" in msg):
                    # Provider/model without JSON mode, or Groq's json_validate_failed — fall back to prompting.
                    if "json_validate_failed" in msg:
                        body = getattr(exc, "body", None) or {}
                        if isinstance(body, dict):
                            generated = body.get("failed_generation") or (body.get("error") or {}).get("failed_generation")
                            if generated:
                                return generated, "json_validate_failed"
                    self._json_mode = False
                    kwargs.pop("response_format", None)
                    resp = self.client.chat.completions.create(**kwargs)
                elif "reasoning_effort" in msg and "reasoning_effort" in kwargs:
                    self.reasoning_effort = None
                    kwargs.pop("reasoning_effort")
                    kwargs["temperature"] = temperature
                    resp = self.client.chat.completions.create(**kwargs)
                else:
                    raise PipelineError(f"{self.endpoint.label} rejected the request: {str(exc)[:300]}",
                                        self.endpoint.role) from exc
            usage = getattr(resp, "usage", None)
            self.tracker.record(
                self.endpoint.role,
                getattr(usage, "prompt_tokens", 0) or 0,
                getattr(usage, "completion_tokens", 0) or 0,
                time.time() - t0,
            )
            choice = resp.choices[0]
            finish = getattr(choice, "finish_reason", None) or ""
            if finish == "length" and self.log:
                self.log(f"{self.endpoint.model}: reply hit the token limit and may be truncated")
            return choice.message.content or "", finish

        return call_with_retries(request, self.endpoint, self.max_retries, self.log)
