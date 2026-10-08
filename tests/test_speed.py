"""Speed-related behaviour: model split, parallel chunking must not change results, thread-safe accounting."""
from __future__ import annotations

import threading
from dataclasses import replace

import pytest

from meetscribe import llm, refine, stt
from meetscribe.config import ModelEndpoint, Settings, load_settings
from meetscribe.pipeline import Pipeline
from meetscribe.schema import PipelineError, Segment, Transcript

from .fakes import FakeClient, RAW_SEGMENTS
from .test_pipeline import _tone, settings, fake_api  # noqa: F401  (fixtures)


def _model(client, monkeypatch, name="openai/gpt-oss-20b", effort="low"):
    monkeypatch.setattr(llm, "make_client", lambda endpoint, timeout: client)
    return llm.ChatModel(ModelEndpoint(name, "https://fake.local/v1", "k", "refine"), reasoning_effort=effort)


def _long_transcript(repeat: int = 6) -> Transcript:
    segs, t = [], 0.0
    for r in range(repeat):
        for a, b, text in RAW_SEGMENTS:
            segs.append(Segment(len(segs) + 1, t, t + (b - a), text))
            t += b - a
    return Transcript(segs)


def test_default_models_are_two_different_models(monkeypatch):
    for k in ("REFINE_MODEL", "DOCUMENT_MODEL", "REFINE_REASONING_EFFORT", "LLM_CONCURRENCY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "x")
    s = load_settings()
    assert s.refine.model != s.document.model
    assert s.refine_reasoning_effort == "low" and s.document_map_reasoning_effort == "low"
    assert s.llm_concurrency == 1


def test_env_overrides_for_speed_knobs(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setenv("LLM_CONCURRENCY", "4")
    monkeypatch.setenv("REFINE_REASONING_EFFORT", "none")
    monkeypatch.setenv("REFINE_MODEL", "some/other-model")
    s = load_settings()
    assert s.llm_concurrency == 4 and s.refine_reasoning_effort is None and s.refine.model == "some/other-model"
    monkeypatch.setenv("LLM_CONCURRENCY", "0")
    assert load_settings().llm_concurrency == 1          # never below 1


def test_parallel_refine_matches_sequential(monkeypatch):
    tr = _long_transcript()
    seq = refine.refine(tr, _model(FakeClient(), monkeypatch), [], 30, concurrency=1)
    par = refine.refine(tr, _model(FakeClient(), monkeypatch), [], 30, concurrency=4)
    key = lambda cs: [(c.segment_id, c.before, c.after, c.applied) for c in cs]  # noqa: E731
    assert len(seq[1]) > 10                               # several chunks really were exercised
    assert key(par[1]) == key(seq[1])                     # same corrections, same order
    assert [s.text for s in par[0].segments] == [s.text for s in seq[0].segments]


def test_parallel_refine_propagates_errors(monkeypatch):
    client = FakeClient()
    real = client.chat.completions.create
    n = {"i": 0}

    def flaky(**kw):
        if "proof-reader" in kw["messages"][0]["content"]:
            n["i"] += 1
            if n["i"] == 3:
                raise PipelineError("boom", "refine")
        return real(**kw)

    client.chat.completions.create = flaky
    with pytest.raises(PipelineError):
        refine.refine(_long_transcript(), _model(client, monkeypatch), [], 30, concurrency=3)


def test_parallel_map_reduce_matches_sequential(tmp_path, settings, fake_api):  # noqa: F811
    audio = _tone(tmp_path / "meeting.mp3")
    base = replace(settings, document_max_words=20)
    a = Pipeline(base).run(audio)
    b = Pipeline(replace(base, llm_concurrency=3)).run(audio)
    assert a.status == b.status == "done", (a.error, b.error)
    sig = lambda st: ([d.decision for d in st.record.decisions],  # noqa: E731
                      [(t.task, t.owner, t.deadline) for t in st.record.action_items],
                      [o.item for o in st.record.open_items])
    assert sig(a) == sig(b)


def test_usage_tracker_is_thread_safe():
    tracker = llm.UsageTracker()
    threads = [threading.Thread(target=lambda: [tracker.record("refine", 10, 5, 0.1) for _ in range(500)])
               for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    u = tracker.as_dict()["refine"]
    assert u["calls"] == 4000 and u["prompt_tokens"] == 40000
