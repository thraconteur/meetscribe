"""Stage 3 — meeting documentation (language model #2).

Short meetings go to the documentation model in a single call. Long meetings
use map-reduce: the model extracts evidence-backed notes from each part, then
merges them in order (so later reversals and acceptances win). Either way the
output is verified against the transcript by ``grounding.verify_record``.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from .grounding import verify_record
from .llm import ChatModel, RequestTooLarge
from .refine import _ts
from .schema import MeetingRecord, Transcript

PROMPT_DIR = Path(__file__).parent / "prompts"


def _prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


def numbered(transcript: Transcript, a: int = 0, b: int | None = None) -> str:
    return "\n".join(f"[S{s.id} @ {_ts(s.start)}] {s.text}" for s in transcript.segments[a:b])


def _split(transcript: Transcript, max_words: int) -> list[tuple[int, int]]:
    parts, start, words = [], 0, 0
    for i, s in enumerate(transcript.segments):
        n = len(s.text.split())
        if words and words + n > max_words:
            parts.append((start, i))
            start, words = i, 0
        words += n
    parts.append((start, len(transcript.segments)))
    return parts


def _context_header(meta: dict) -> str:
    lines = []
    if meta.get("title_hint"):
        lines.append(f"Meeting name given by the user: {meta['title_hint']}")
    if meta.get("context"):
        lines.append(f"Context given by the user (background only — do not treat as said in the meeting): {meta['context']}")
    if meta.get("domain"):
        lines.append(f"Detected domain: {meta['domain']}")
    return ("\n".join(lines) + "\n\n") if lines else ""


def generate_record(transcript: Transcript, model: ChatModel, max_words: int, threshold: float,
                    meta: dict | None = None, progress: Callable[[float, str], None] | None = None,
                    log: Callable[[str], None] | None = None, concurrency: int = 1,
                    map_model: ChatModel | None = None) -> tuple[MeetingRecord, dict]:
    meta = meta or {}
    header = _context_header(meta)
    raw: dict | None = None
    if transcript.word_count <= max_words:
        if progress:
            progress(0.1, "Writing minutes, decisions and action items")
        try:
            raw = model.json(_prompt("document.md"), header + "Transcript:\n" + numbered(transcript), max_tokens=12000)
        except RequestTooLarge:
            if log:
                log("Transcript too large for one request on this provider tier; switching to map-reduce")
            raw = None
    if raw is None:
        raw = _map_reduce(transcript, model, max(800, max_words // 2), header, progress, log, concurrency, map_model)

    if progress:
        progress(0.95, "Verifying every item against the transcript")
    record = verify_record(raw if isinstance(raw, dict) else {}, transcript, threshold)
    return record, raw if isinstance(raw, dict) else {}


def _map_reduce(transcript: Transcript, model: ChatModel, part_words: int, header: str,
                progress: Callable[[float, str], None] | None, log: Callable[[str], None] | None,
                concurrency: int = 1, map_model: ChatModel | None = None) -> dict:
    parts = _split(transcript, part_words)
    if log:
        log(f"Long meeting: documenting in {len(parts)} parts, then merging")
    extractor = map_model or model

    def map_part(i: int) -> dict:
        a, b = parts[i]
        data = extractor.json(_prompt("document_map.md"),
                              header + f"Part {i + 1} of {len(parts)}:\n" + numbered(transcript, a, b), max_tokens=6000)
        return {"part": i + 1, "notes": data if isinstance(data, dict) else {}}

    notes: list[dict] = [{}] * len(parts)
    if concurrency <= 1 or len(parts) <= 1:
        for i in range(len(parts)):
            if progress:
                progress(0.05 + 0.7 * i / len(parts), f"Extracting notes from part {i + 1}/{len(parts)}")
            notes[i] = map_part(i)
    else:
        pool = ThreadPoolExecutor(max_workers=min(concurrency, len(parts)))
        try:
            futures = {pool.submit(map_part, i): i for i in range(len(parts))}
            for n, fut in enumerate(as_completed(futures), 1):
                notes[futures[fut]] = fut.result()
                if progress:
                    progress(0.05 + 0.7 * n / len(parts), f"Extracting notes ({n}/{len(parts)} parts)")
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        pool.shutdown()
    if progress:
        progress(0.8, "Merging notes into one record")
    merged = model.json(_prompt("document_reduce.md"), header + json.dumps(notes, ensure_ascii=False), max_tokens=12000)
    return merged if isinstance(merged, dict) else {}
