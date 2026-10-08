"""End-to-end orchestration: audio → transcript → refined transcript → meeting record → files.

``run_iter`` yields a ``RunState`` after every stage so an interface can show
results as soon as they exist; ``run`` simply drains it.
"""
from __future__ import annotations

import re
import shutil
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator

from . import audio, document, refine, render, stt
from .config import Settings, load_settings
from .llm import ChatModel, UsageTracker
from .schema import Correction, MeetingRecord, PipelineError, Transcript

STAGES = ["validate", "transcribe", "refine", "document", "export"]
STAGE_LABELS = {
    "validate": "Checking & preparing audio",
    "transcribe": "Speech-to-text",
    "refine": "Domain-term refinement",
    "document": "Minutes, decisions & tasks",
    "export": "Saving outputs",
}


@dataclass
class RunState:
    stage: str = "validate"
    status: str = "running"  # running | done | error
    message: str = ""
    stage_times: dict[str, float] = field(default_factory=dict)
    log: list[str] = field(default_factory=list)
    run_dir: Path | None = None
    audio_info: audio.AudioInfo | None = None
    raw: Transcript | None = None
    refined: Transcript | None = None
    corrections: list[Correction] = field(default_factory=list)
    profile: dict = field(default_factory=dict)
    record: MeetingRecord | None = None
    files: dict[str, Path] = field(default_factory=dict)
    error: str = ""
    failed_stage: str = ""


def parse_hints(text: str | None) -> list[str]:
    if not text:
        return []
    items = re.split(r"[,\n;]+", text)
    return list(dict.fromkeys(i.strip() for i in items if i.strip()))[:60]


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-")[:40] or "meeting"


class Pipeline:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or load_settings()

    def run_iter(self, audio_path: str | Path, hints: str = "", title_hint: str = "", context: str = "",
                 progress: Callable[[float, str], None] | None = None) -> Iterator[RunState]:
        s = self.settings
        st = RunState()
        tracker = UsageTracker()

        def log(msg: str) -> None:
            st.log.append(f"{datetime.now().strftime('%H:%M:%S')}  {msg}")

        def stage_progress(stage: str) -> Callable[[float, str], None]:
            idx = STAGES.index(stage)

            def cb(frac: float, desc: str) -> None:
                st.message = desc
                if progress:
                    progress((idx + max(0.0, min(1.0, frac))) / len(STAGES), desc)
            return cb

        t_start = time.time()
        try:
            # 1. validate + normalise ------------------------------------------------
            st.stage, st.message = "validate", "Checking the uploaded file"
            stage_progress("validate")(0.0, st.message)
            yield st
            t0 = time.time()
            info = audio.validate_input(audio_path, s.max_upload_mb, s.max_duration_min)
            st.audio_info = info
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            st.run_dir = s.output_dir / f"{stamp}-{_slug(Path(audio_path).stem)}"
            work = st.run_dir / "work"
            normalised = audio.normalise(info.path, work)
            audio.assert_not_silent(normalised)
            st.stage_times["validate"] = time.time() - t0
            log(f"Audio OK: {info.duration / 60:.1f} min, {info.codec}, {info.sample_rate} Hz, {info.channels} ch, "
                f"{info.size_mb:.1f} MB")

            # 2. speech-to-text ------------------------------------------------------
            hint_list = parse_hints(hints)
            st.stage, st.message = "transcribe", "Transcribing speech"
            yield st
            t0 = time.time()
            st.raw = stt.transcribe(normalised, info.duration, s, work, hint_list, tracker, log,
                                    stage_progress("transcribe"))
            st.stage_times["transcribe"] = time.time() - t0
            log(f"Transcribed {st.raw.word_count} words in {len(st.raw.segments)} segments with {st.raw.model}")
            for note in st.raw.notes:
                log(note)
            yield st

            # 3. refinement (LM #1) --------------------------------------------------
            st.stage, st.message = "refine", "Correcting domain-specific terms"
            yield st
            t0 = time.time()
            refine_model = ChatModel(s.refine, s.request_timeout, s.max_retries, tracker, log,
                                     reasoning_effort=s.refine_reasoning_effort)
            st.refined, st.corrections, st.profile = refine.refine(
                st.raw, refine_model, hint_list, s.refine_chunk_words, stage_progress("refine"), log,
                concurrency=s.llm_concurrency)
            st.stage_times["refine"] = time.time() - t0
            applied = sum(c.applied for c in st.corrections)
            rejected = len(st.corrections) - applied
            log(f"Refinement: {applied} corrections applied, {rejected} rejected by safety checks")
            yield st

            # 4. documentation (LM #2) -----------------------------------------------
            st.stage, st.message = "document", "Writing minutes, decisions and action items"
            yield st
            t0 = time.time()
            doc_model = ChatModel(s.document, s.request_timeout, s.max_retries, tracker, log,
                                  reasoning_effort=s.document_reasoning_effort)
            map_model = ChatModel(s.document, s.request_timeout, s.max_retries, tracker, log,
                                  reasoning_effort=s.document_map_reasoning_effort)
            meta = {"title_hint": title_hint.strip(), "context": context.strip(), "domain": st.profile.get("domain")}
            st.record, _raw_doc = document.generate_record(st.refined, doc_model, s.document_max_words,
                                                           s.evidence_match_threshold, meta,
                                                           stage_progress("document"), log,
                                                           concurrency=s.llm_concurrency, map_model=map_model)
            if title_hint.strip():
                st.record.title = title_hint.strip()
            st.stage_times["document"] = time.time() - t0
            log(f"Record: {len(st.record.decisions)} decisions, {len(st.record.action_items)} action items, "
                f"{len(st.record.open_items)} open items")

            # 5. export ----------------------------------------------------------------
            st.stage, st.message = "export", "Saving outputs"
            t0 = time.time()
            st.record.metadata = {
                "source_file": Path(audio_path).name,
                "duration_s": round(info.duration, 1),
                "processed_at": datetime.now().isoformat(timespec="seconds"),
                "models": {"stt": st.raw.model, "refine": s.refine.label, "document": s.document.label},
                "domain": st.profile.get("domain", ""),
                "glossary": st.profile.get("glossary", []),
                "corrections_applied": applied,
                "corrections_rejected": rejected,
                "stage_seconds": {k: round(v, 1) for k, v in st.stage_times.items()},
                "usage": tracker.as_dict(),
            }
            st.files = render.write_outputs(st.run_dir, st.raw, st.refined, st.corrections, st.record)
            st.stage_times["export"] = time.time() - t0
            shutil.rmtree(work / "chunks", ignore_errors=True)
            st.status = "done"
            st.message = f"Done in {time.time() - t_start:.0f}s"
            log(st.message)
            if progress:
                progress(1.0, st.message)
            yield st
        except PipelineError as exc:
            st.status, st.error, st.failed_stage = "error", exc.user_message, st.stage
            log(f"ERROR: {exc.user_message}")
            yield st
        except Exception as exc:  # noqa: BLE001 — never crash the UI; surface a readable error
            st.status, st.failed_stage = "error", st.stage
            st.error = f"Unexpected error during {STAGE_LABELS.get(st.stage, st.stage).lower()}: {exc}"
            log("ERROR: " + st.error)
            log(traceback.format_exc(limit=3))
            yield st

    def run(self, audio_path: str | Path, **kwargs) -> RunState:
        state = RunState()
        for state in self.run_iter(audio_path, **kwargs):
            pass
        return state
