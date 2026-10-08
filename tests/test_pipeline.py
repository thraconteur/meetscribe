"""Offline tests: real ffmpeg audio handling, fake model APIs."""
from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from meetscribe import llm, stt
from meetscribe.audio import plan_chunks, validate_input
from meetscribe.config import ModelEndpoint, Settings
from meetscribe.pipeline import Pipeline
from meetscribe.schema import PipelineError, UNSPECIFIED
from meetscribe.textutil import negation_count, number_values

from .fakes import FakeClient, FakeTranscriptions


def _tone(path: Path, seconds: float = 8, volume: float = 0.5) -> Path:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    f"sine=frequency=440:duration={seconds}", "-af", f"volume={volume}", str(path)], check=True)
    return path


def _silence(path: Path, seconds: float = 8) -> Path:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "anullsrc=r=16000:cl=mono", "-t", str(seconds), str(path)], check=True)
    return path


@pytest.fixture
def settings(tmp_path):
    ep = lambda m, r: ModelEndpoint(m, "https://fake.local/v1", "test-key", r)  # noqa: E731
    return Settings(stt=ep("whisper-large-v3", "stt"), refine=ep("llama-3.3-70b-versatile", "refine"),
                    document=ep("openai/gpt-oss-120b", "document"), output_dir=tmp_path / "runs")


@pytest.fixture
def fake_api(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(llm, "make_client", lambda endpoint, timeout: client)
    monkeypatch.setattr(stt, "make_client", lambda endpoint, timeout: client)
    return client


# --------------------------------------------------------------------------- end to end
def test_end_to_end(tmp_path, settings, fake_api):
    wav = _tone(tmp_path / "meeting.wav")
    state = Pipeline(settings).run(wav, hints="kubectl, Helm")
    assert state.status == "done", state.error

    # stage 1 — Whisper got the glossary as a prompt
    assert "kubectl" in fake_api.audio.transcriptions.calls[0]["prompt"]

    # stage 2 — safe corrections applied, unsafe ones rejected
    refined = {s.id: s.text for s in state.refined.segments}
    assert "kubectl rollout" in refined[2] and "Kubernetes" in refined[2]
    assert "MySQL" in refined[5] and "Postgres" in refined[5]
    assert "Q3" in refined[6]
    assert "should not touch" in refined[7]                       # negation preserved
    assert "four hundred dollars" in refined[10]                  # number preserved
    assert "I'll update" in refined[4]                            # commitment preserved
    reasons = {c.after: c.rejected_because for c in state.corrections if not c.applied}
    assert "negation" in reasons["should touch"]
    assert "number" in reasons["forty dollars"]
    assert "modality" in reasons["I might update"]
    # raw transcript is untouched
    assert "cube control" in state.raw.segment(2).text

    # stage 3 — grounding
    rec = state.record
    assert [d.decision for d in rec.decisions] == ["Migrate from MySQL to Postgres in Q3"]  # fake decision removed
    tasks = {t.task: t for t in rec.action_items}
    helm = tasks["Update the Helm charts for the staging cluster"]
    assert helm.owner == "Priya" and helm.owner_stated and helm.deadline == "by Friday" and helm.deadline_stated
    assert helm.evidence.start == 13.0 and helm.evidence.segment_ids == [3]   # precise citation
    assert rec.decisions[0].evidence.start == 33.0
    assert tasks["Freeze billing tables"].evidence.start == 40.0
    runbook = tasks["Write the migration runbook"]
    assert runbook.owner == UNSPECIFIED and not runbook.owner_stated     # invented owner removed
    assert runbook.deadline == UNSPECIFIED and not runbook.deadline_stated  # invented deadline removed
    bench = tasks["Benchmark the new API gateway"]
    assert bench.owner == "Rahul" and bench.deadline == UNSPECIFIED
    assert tasks["Freeze billing tables"].owner == UNSPECIFIED           # pronoun is not an owner
    assert "Ghost Person" not in rec.participants
    assert rec.open_items and rec.open_items[0].kind == "proposal"
    assert any(t.evidence.start is not None for t in rec.action_items)

    # outputs exist and human/machine formats agree
    for key in ("raw_transcript", "refined_transcript", "record_md", "record_html", "record_json", "tasks_csv",
                "bundle"):
        assert state.files[key].exists() and state.files[key].stat().st_size > 0
    data = json.loads(state.files["record_json"].read_text(encoding="utf-8"))
    md = state.files["record_md"].read_text(encoding="utf-8")
    assert len(data["decisions"]) == len(rec.decisions)
    assert len(data["action_items"]) == len(rec.action_items)
    for t in data["action_items"]:
        assert t["task"] in md and t["owner"] in md
    for d in data["decisions"]:
        assert d["decision"] in md
    assert data["metadata"]["models"]["refine"] != data["metadata"]["models"]["document"]


def test_map_reduce_path(tmp_path, settings, fake_api):
    wav = _tone(tmp_path / "meeting.mp3")
    st = Pipeline(replace(settings, document_max_words=20)).run(wav)
    assert st.status == "done", st.error
    systems = [c["messages"][0]["content"] for c in fake_api.chat.completions.calls]
    assert any("processing ONE PART" in s for s in systems)
    assert st.record.decisions


# --------------------------------------------------------------------------- bad inputs
@pytest.mark.parametrize("name,maker,expect", [
    ("notes.txt", lambda p: p.write_text("hello"), "Unsupported file type"),
    ("empty.mp3", lambda p: p.write_bytes(b""), "empty"),
    ("broken.mp3", lambda p: p.write_bytes(b"\x00\x01garbage" * 100), "could not be read"),
    ("tiny.wav", lambda p: _tone(p, 0.3), "too short"),
])
def test_rejects_bad_files(tmp_path, settings, fake_api, name, maker, expect):
    path = tmp_path / name
    maker(path)
    st = Pipeline(settings).run(path)
    assert st.status == "error" and expect in st.error
    assert not fake_api.chat.completions.calls  # failed before any paid call


def test_rejects_silence(tmp_path, settings, fake_api):
    st = Pipeline(settings).run(_silence(tmp_path / "quiet.wav"))
    assert st.status == "error" and "silent" in st.error


def test_no_speech_recognised(tmp_path, settings, monkeypatch):
    client = FakeClient(transcriptions=FakeTranscriptions(segments=[(0.0, 5.0, "Thank you for watching!")]))
    monkeypatch.setattr(llm, "make_client", lambda e, t: client)
    monkeypatch.setattr(stt, "make_client", lambda e, t: client)
    # make the hallucination look like non-speech
    orig = client.audio.transcriptions.create

    def create(**kw):
        r = orig(**kw)
        r.segments[0]["no_speech_prob"] = 0.9
        r.segments[0]["avg_logprob"] = -1.2
        return r
    client.audio.transcriptions.create = create
    st = Pipeline(settings).run(_tone(tmp_path / "m.wav"))
    assert st.status == "error" and "No speech" in st.error


def test_missing_api_key(tmp_path, settings):
    s = replace(settings, stt=replace(settings.stt, api_key=None))
    st = Pipeline(s).run(_tone(tmp_path / "m.wav"))
    assert st.status == "error" and "API key" in st.error


# --------------------------------------------------------------------------- units
def test_number_and_negation_helpers():
    assert number_values("twenty five") == number_values("25")
    assert number_values("Q three") == number_values("Q3")
    assert number_values("four hundred dollars") != number_values("forty dollars")
    assert number_values("easy two") == number_values("EC2")
    assert negation_count("we don't ship") == 1 and negation_count("we ship") == 0


def test_chunk_plan_prefers_pauses():
    spans = plan_chunks(1300, 600, [(590.0, 592.0), (1190.0, 1191.0)])
    assert spans[0] == (0.0, 591.0) and spans[1][0] == 591.0 and spans[-1][1] == 1300
    assert plan_chunks(500, 600, []) == [(0.0, 500)]


def test_long_recording_is_chunked_at_pauses(tmp_path, settings, fake_api):
    # 3 tone bursts separated by silences; chunk length 10 s forces splitting.
    wav = tmp_path / "long.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "sine=frequency=300:duration=30", "-af",
                    "volume=enable='between(t,9,11)+between(t,19,21)':volume=0", str(wav)], check=True)
    st = Pipeline(replace(settings, stt_chunk_seconds=10)).run(wav)
    assert st.status == "done", st.error
    assert len(fake_api.audio.transcriptions.calls) == 3
    # second chunk's segments are offset by the chunk start (~10 s)
    starts = [s.start for s in st.raw.segments]
    assert max(starts) > 20


def test_extract_json_tolerates_noise():
    assert llm.extract_json('Sure! ```json\n{"a": [1, 2,],}\n```') == {"a": [1, 2]}
    assert llm.extract_json('prefix {"b": 1} suffix') == {"b": 1}


def test_validate_reports_video_without_audio(tmp_path):
    vid = tmp_path / "screen.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=2",
                    str(vid)], check=True)
    with pytest.raises(PipelineError, match="no audio track|could not be read"):
        validate_input(vid)
