"""Stage 1 — speech-to-text.

Two interchangeable back-ends:
* ``api``   — any OpenAI-compatible /audio/transcriptions endpoint (default: Groq whisper-large-v3)
* ``local`` — faster-whisper running on this machine (CPU or GPU), no API key needed

Both return timestamped segments, which later stages cite as evidence.
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Callable

from . import audio
from .config import Settings
from .llm import RequestTooLarge, UsageTracker, call_with_retries, make_client
from .schema import PipelineError, Segment, Transcript

# Phrases Whisper is known to hallucinate on silence / music / noise.
HALLUCINATION_PATTERNS = re.compile(
    r"^(thank(s| you)( so much)? for (watching|listening)|please (like and )?subscribe|subtitles? by|"
    r"transcri(bed|ption) by|captions? by|amara\.org|www\.|♪+|\[music\]|\(music\)|you)\W*$",
    re.I,
)


def build_prompt(glossary: list[str], previous_text: str = "") -> str:
    """Whisper's ``prompt`` biases spelling of rare words. It is limited to ~224 tokens, so keep it tight."""
    parts = []
    if glossary:
        parts.append("Meeting discussion. Terms used: " + ", ".join(glossary[:40]) + ".")
    if previous_text:
        parts.append(" ".join(previous_text.split()[-40:]))
    prompt = " ".join(parts)
    words = prompt.split()
    return " ".join(words[-150:])


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def filter_segments(raw: list[dict], notes: list[str]) -> list[dict]:
    """Remove the classic Whisper failure modes before anything downstream sees them."""
    kept: list[dict] = []
    for seg in raw:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        nsp = seg.get("no_speech_prob")
        alp = seg.get("avg_logprob")
        cr = seg.get("compression_ratio")
        stamp = f"{seg['start']:.1f}-{seg['end']:.1f}s"
        if nsp is not None and alp is not None and nsp > 0.6 and alp < -1.0:
            notes.append(f"Dropped likely non-speech segment at {stamp}: “{text[:60]}”")
            continue
        if HALLUCINATION_PATTERNS.match(text) and (nsp is None or nsp > 0.2 or (alp is not None and alp < -0.6)):
            notes.append(f"Dropped typical Whisper hallucination at {stamp}: “{text[:60]}”")
            continue
        if cr is not None and cr > 2.6 and alp is not None and alp < -0.4:
            notes.append(f"Dropped repetitive (looping) segment at {stamp}: “{text[:60]}…”")
            continue
        # Collapse 3+ identical consecutive segments (decoder loop).
        if len(kept) >= 2 and kept[-1]["text"].strip().lower() == text.lower() == kept[-2]["text"].strip().lower():
            notes.append(f"Collapsed repeated segment at {stamp}: “{text[:60]}”")
            continue
        kept.append({**seg, "text": text})
    return kept


class ApiTranscriber:
    def __init__(self, settings: Settings, tracker: UsageTracker, log: Callable[[str], None] | None = None):
        self.settings = settings
        self.endpoint = settings.stt
        self.client = make_client(self.endpoint, settings.request_timeout)
        self.tracker = tracker
        self.log = log

    def transcribe_file(self, path: Path, prompt: str) -> list[dict]:
        def request():
            t0 = time.time()
            with open(path, "rb") as fh:
                kwargs = dict(
                    model=self.endpoint.model,
                    file=(path.name, fh.read()),
                    response_format="verbose_json",
                    temperature=0.0,
                    language=self.settings.language,
                )
                if prompt:
                    kwargs["prompt"] = prompt
                if "whisper" in self.endpoint.model:
                    kwargs["timestamp_granularities"] = ["segment"]
                try:
                    resp = self.client.audio.transcriptions.create(**kwargs)
                except Exception as exc:  # noqa: BLE001 — some models refuse verbose_json; fall back to json
                    if "response_format" in str(exc) or "verbose_json" in str(exc):
                        kwargs["response_format"] = "json"
                        kwargs.pop("timestamp_granularities", None)
                        resp = self.client.audio.transcriptions.create(**kwargs)
                    else:
                        raise
            self.tracker.record("stt", 0, 0, time.time() - t0)
            return resp

        try:
            resp = call_with_retries(request, self.endpoint, self.settings.max_retries, self.log)
        except RequestTooLarge as exc:
            raise PipelineError(
                "The audio chunk was larger than the transcription API allows. Lower STT_CHUNK_SECONDS in .env "
                "(e.g. 300) and try again.",
                "stt",
            ) from exc

        segments = _get(resp, "segments") or []
        out = []
        for s in segments:
            out.append({
                "start": float(_get(s, "start", 0.0) or 0.0),
                "end": float(_get(s, "end", 0.0) or 0.0),
                "text": _get(s, "text", "") or "",
                "avg_logprob": _get(s, "avg_logprob"),
                "no_speech_prob": _get(s, "no_speech_prob"),
                "compression_ratio": _get(s, "compression_ratio"),
            })
        if not out:
            text = _get(resp, "text", "") or ""
            if text.strip():
                out.append({"start": 0.0, "end": float(_get(resp, "duration", 0.0) or 0.0), "text": text})
        return out


class LocalTranscriber:
    """faster-whisper back-end (pip install faster-whisper). Model weights download on first use."""

    _cache: dict = {}

    def __init__(self, settings: Settings, tracker: UsageTracker, log: Callable[[str], None] | None = None):
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise PipelineError(
                "STT_BACKEND=local needs faster-whisper. Install it with: pip install faster-whisper", "stt"
            ) from exc
        key = (settings.local_whisper_model, settings.local_whisper_device)
        if key not in self._cache:
            device = settings.local_whisper_device
            if device == "auto":
                try:
                    import ctranslate2

                    device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
                except Exception:  # noqa: BLE001
                    device = "cpu"
            compute = "float16" if device == "cuda" else "int8"
            self._cache[key] = WhisperModel(settings.local_whisper_model, device=device, compute_type=compute)
        self.model = self._cache[key]
        self.settings = settings
        self.tracker = tracker

    def transcribe_file(self, path: Path, prompt: str) -> list[dict]:
        t0 = time.time()
        segments, _info = self.model.transcribe(
            str(path), language=self.settings.language, beam_size=5, temperature=0.0,
            initial_prompt=prompt or None, vad_filter=True, condition_on_previous_text=False,
        )
        out = [
            {"start": s.start, "end": s.end, "text": s.text, "avg_logprob": s.avg_logprob,
             "no_speech_prob": s.no_speech_prob, "compression_ratio": s.compression_ratio}
            for s in segments
        ]
        self.tracker.record("stt", 0, 0, time.time() - t0)
        return out


def transcribe(normalised_audio: Path, duration: float, settings: Settings, workdir: Path, glossary: list[str],
               tracker: UsageTracker, log: Callable[[str], None] | None = None,
               progress: Callable[[float, str], None] | None = None) -> Transcript:
    backend = LocalTranscriber if settings.stt_backend == "local" else ApiTranscriber
    engine = backend(settings, tracker, log)

    spans = audio.plan_chunks(duration, settings.stt_chunk_seconds,
                              audio.silences(normalised_audio) if duration > settings.stt_chunk_seconds else [])
    chunks = audio.cut_chunks(normalised_audio, spans, workdir / "chunks")
    if log and len(chunks) > 1:
        log(f"Long recording: transcribing in {len(chunks)} chunks split at pauses")

    raw: list[dict] = []
    previous = ""
    for i, (chunk_path, offset) in enumerate(chunks):
        if progress:
            progress(i / len(chunks), f"Transcribing part {i + 1}/{len(chunks)}")
        segs = engine.transcribe_file(chunk_path, build_prompt(glossary, previous))
        for s in segs:
            s["start"] += offset
            s["end"] += offset
        raw.extend(segs)
        previous = " ".join(s["text"] for s in segs[-6:])

    notes: list[str] = []
    kept = filter_segments(raw, notes)
    segments = [Segment(i + 1, round(s["start"], 2), round(s["end"], 2), s["text"]) for i, s in enumerate(kept)]
    model_name = (f"faster-whisper {settings.local_whisper_model} (local)" if settings.stt_backend == "local"
                  else settings.stt.label)
    transcript = Transcript(segments=segments, duration=duration, language=settings.language,
                            model=model_name, notes=notes)
    if not transcript.text.strip():
        raise PipelineError(
            "No speech could be recognised in this recording. It may contain only music, noise or silence, or "
            "the speech may not be in English.",
            "stt",
        )
    return transcript
