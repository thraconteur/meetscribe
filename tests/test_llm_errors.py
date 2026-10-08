"""Provider error handling, using the real openai exception classes."""
from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import openai
import pytest

from meetscribe import llm
from meetscribe.config import ModelEndpoint
from meetscribe.schema import PipelineError

EP = ModelEndpoint("openai/gpt-oss-120b", "https://api.groq.com/openai/v1", "k", "document")


def _err(cls, status, body, headers=None):
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    resp = httpx.Response(status, request=req, headers=headers or {}, json=body)
    return cls(json.dumps(body), response=resp, body=body)


def _ok(content="{\"ok\": true}", finish="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish)],
                           usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))


class ScriptedCompletions:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def model(monkeypatch):
    def build(script):
        comp = ScriptedCompletions(script)
        client = SimpleNamespace(chat=SimpleNamespace(completions=comp))
        monkeypatch.setattr(llm, "make_client", lambda e, t: client)
        monkeypatch.setattr(llm.time, "sleep", lambda s: None)
        m = llm.ChatModel(EP, reasoning_effort="medium")
        return m, comp
    return build


def test_rate_limit_is_retried(model):
    m, comp = model([_err(openai.RateLimitError, 429, {"error": {"message": "slow down"}}, {"retry-after": "2"}),
                     _ok()])
    assert m.json("sys", "user") == {"ok": True}
    assert len(comp.calls) == 2
    assert comp.calls[0]["reasoning_effort"] == "medium" and "temperature" not in comp.calls[0]


def test_request_too_large_shrinks_output_budget(model):
    msg = {"error": {"message": "Request too large for model `openai/gpt-oss-120b` on tokens per minute (TPM): "
                                "Limit 8000, Requested 15000, please reduce your message size"}}
    m, comp = model([_err(openai.APIStatusError, 413, msg), _ok()])
    assert m.json("sys", "user", max_tokens=12000) == {"ok": True}
    assert comp.calls[1]["max_tokens"] == 8000 - (15000 - 12000) - 300


def test_json_validate_failed_uses_failed_generation(model):
    body = {"error": {"message": "Failed to generate JSON", "code": "json_validate_failed",
                      "failed_generation": "{\"recovered\": 1,}"}}
    m, _ = model([_err(openai.BadRequestError, 400, body)])
    assert m.json("sys", "user") == {"recovered": 1}


def test_truncated_reasoning_retries_with_less_effort(model):
    m, comp = model([_ok(content="", finish="length"), _ok()])
    assert m.json("sys", "user", max_tokens=1000) == {"ok": True}
    assert comp.calls[1]["reasoning_effort"] == "low" and comp.calls[1]["max_tokens"] == 1500


def test_auth_error_is_readable(model):
    m, _ = model([_err(openai.AuthenticationError, 401, {"error": {"message": "bad key"}})])
    with pytest.raises(PipelineError, match="rejected"):
        m.json("sys", "user")


def test_model_without_json_mode_falls_back(model):
    body = {"error": {"message": "response_format is not supported for this model"}}
    m, comp = model([_err(openai.BadRequestError, 400, body), _ok("Here you go: {\"a\": 2}")])
    assert m.json("sys", "user") == {"a": 2}
    assert "response_format" not in comp.calls[1]
