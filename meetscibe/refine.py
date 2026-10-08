"""Stage 2 — domain-aware transcript refinement (language model #1).

Two passes with the refinement model:
  A. *Domain profile* — infer the meeting's domain, canonical glossary and
     likely mis-hearings from the whole transcript, so corrections are
     consistent across chunks.
  B. *Targeted corrections* — per chunk of segments, the model proposes
     minimal span edits (``original`` → ``corrected``) instead of rewriting.

Every proposed edit then passes deterministic guards before it is applied:
the span must exist in that segment, and the edit may not change numbers,
negation, modality (will/might/...), or known people's names, and may not
be a long rewrite. Rejected edits are kept in the log for transparency.
"""
from __future__ import annotations

import difflib
from concurrent.futures import ThreadPoolExecutor, as_completed
import html
from pathlib import Path
from typing import Callable

from .llm import ChatModel, RequestTooLarge
from .schema import Correction, Transcript
from .textutil import locate, modal_counter, negation_count, number_values, tokens

PROMPT_DIR = Path(__file__).parent / "prompts"
MAX_ORIGINAL_WORDS = 10


def _prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


def _sample_for_profile(transcript: Transcript, max_words: int = 3500) -> str:
    text = transcript.text
    words = text.split()
    if len(words) <= max_words:
        return text
    # Take evenly spaced windows so terms from the whole meeting are seen.
    n = 5
    window = max_words // n
    step = (len(words) - window) // (n - 1)
    return "\n...\n".join(" ".join(words[i * step : i * step + window]) for i in range(n))


def build_profile(model: ChatModel, transcript: Transcript, hints: list[str]) -> dict:
    user = ""
    if hints:
        user += "Hint terms supplied by the user (likely to appear): " + ", ".join(hints) + "\n\n"
    user += "Transcript excerpt:\n" + _sample_for_profile(transcript)
    data = model.json(_prompt("refine_profile.md"), user, max_tokens=3000)
    if not isinstance(data, dict):
        data = {}
    return {
        "domain": str(data.get("domain", "") or ""),
        "glossary": [str(g) for g in (data.get("glossary") or []) if str(g).strip()][:80],
        "suspected_errors": [
            {"heard": str(e.get("heard", "")), "likely": str(e.get("likely", ""))}
            for e in (data.get("suspected_errors") or []) if isinstance(e, dict) and e.get("heard")
        ][:60],
        "names": [str(n) for n in (data.get("names") or []) if str(n).strip()][:40],
    }


def chunk_segments(transcript: Transcript, max_words: int) -> list[tuple[int, int]]:
    """Return (start_idx, end_idx) index ranges over transcript.segments."""
    ranges, start, words = [], 0, 0
    for i, seg in enumerate(transcript.segments):
        w = len(seg.text.split())
        if words and words + w > max_words:
            ranges.append((start, i))
            start, words = i, 0
        words += w
    if start < len(transcript.segments):
        ranges.append((start, len(transcript.segments)))
    return ranges


def _chunk_text(transcript: Transcript, a: int, b: int, context: int = 2) -> str:
    lines = []
    for seg in transcript.segments[max(0, a - context) : a]:
        lines.append(f"[S{seg.id}] (context) {seg.text}")
    for seg in transcript.segments[a:b]:
        lines.append(f"[S{seg.id}] {seg.text}")
    return "\n".join(lines)


def _profile_block(profile: dict, hints: list[str]) -> str:
    parts = []
    if profile.get("domain"):
        parts.append(f"Meeting domain: {profile['domain']}")
    gloss = list(dict.fromkeys(hints + profile.get("glossary", [])))
    if gloss:
        parts.append("Canonical terms: " + ", ".join(gloss))
    if profile.get("suspected_errors"):
        parts.append("Likely mis-hearings seen elsewhere in this meeting: " + "; ".join(
            f'"{e["heard"]}" → "{e["likely"]}"' for e in profile["suspected_errors"]))
    if profile.get("names"):
        parts.append("People named in the meeting: " + ", ".join(profile["names"]))
    return "\n".join(parts)


def guard(correction: Correction, segment_text: str, protected_names: set[str]) -> str:
    """Return '' if the edit is safe to apply, otherwise the reason it was rejected."""
    before, after = correction.before.strip(), correction.after.strip()
    if not before or not after:
        return "empty original or replacement"
    if before == after:
        return "no change"
    if len(before.split()) > MAX_ORIGINAL_WORDS:
        return "edit too long (rewrites rather than corrects a term)"
    if len(after.split()) > len(before.split()) + 4:
        return "replacement adds too many words"
    if number_values(before) != number_values(after):
        return "would change a number"
    if negation_count(before) != negation_count(after):
        return "would change negation"
    if modal_counter(before) != modal_counter(after):
        return "would change a commitment/modality word"
    before_tokens = {t.lower() for t in tokens(before)}
    after_tokens = {t.lower() for t in tokens(after)}
    lost_names = {n for n in protected_names if n.lower() in before_tokens and n.lower() not in after_tokens}
    if lost_names and correction.category != "name":
        return f"would remove the name {', '.join(sorted(lost_names))}"
    if locate(before, segment_text) is None:
        return "original text not found in the segment"
    return ""


def apply_corrections(transcript: Transcript, corrections: list[Correction], protected_names: set[str]) -> Transcript:
    refined = transcript.copy()
    by_id = {s.id: s for s in refined.segments}
    for c in corrections:
        seg = by_id.get(c.segment_id)
        if seg is None:
            c.rejected_because = f"segment S{c.segment_id} does not exist"
            continue
        reason = guard(c, seg.text, protected_names)
        if reason:
            c.rejected_because = reason
            continue
        a, b = locate(c.before, seg.text)
        c.before = seg.text[a:b]  # record exactly what was replaced
        seg.text = seg.text[:a] + c.after.strip() + seg.text[b:]
        c.applied = True
    return refined


def _parse_corrections(data, valid_ids: set[int]) -> list[Correction]:
    items = data.get("corrections", []) if isinstance(data, dict) else data if isinstance(data, list) else []
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        raw_id = str(item.get("segment", "")).strip().lstrip("[Ss").rstrip("]")
        try:
            seg_id = int(raw_id)
        except ValueError:
            continue
        if seg_id not in valid_ids:
            continue
        out.append(Correction(
            segment_id=seg_id,
            before=str(item.get("original", "")),
            after=str(item.get("corrected", "")),
            category=str(item.get("category", "term") or "term"),
            reason=str(item.get("reason", "") or ""),
        ))
    return out


def refine(transcript: Transcript, model: ChatModel, hints: list[str], chunk_words: int,
           progress: Callable[[float, str], None] | None = None,
           log: Callable[[str], None] | None = None,
           concurrency: int = 1) -> tuple[Transcript, list[Correction], dict]:
    if progress:
        progress(0.0, "Inferring meeting domain and glossary")
    profile = build_profile(model, transcript, hints)
    if log and profile.get("domain"):
        log(f"Domain detected: {profile['domain']}")

    system = _prompt("refine_correct.md")
    context = _profile_block(profile, hints)
    def correct_range(a: int, b: int) -> list[Correction]:
        user = (context + "\n\n" if context else "") + "Transcript segments:\n" + _chunk_text(transcript, a, b)
        try:
            data = model.json(system, user, max_tokens=4000)
        except RequestTooLarge:
            if b - a <= 1:
                raise
            mid = (a + b) // 2
            if log:
                log("Refinement request too large for the provider's limits; splitting the chunk")
            return correct_range(a, mid) + correct_range(mid, b)
        return _parse_corrections(data, {s.id for s in transcript.segments[a:b]})

    chunks = chunk_segments(transcript, chunk_words)
    total = len(chunks)
    results: dict[int, list[Correction]] = {}
    if concurrency <= 1 or total <= 1:
        for i, (a, b) in enumerate(chunks):
            if progress:
                progress(0.1 + 0.9 * i / max(total, 1), f"Correcting domain terms ({i + 1}/{total})")
            results[i] = correct_range(a, b)
    else:
        workers = min(concurrency, total)
        if log:
            log(f"Refining {total} chunks with {workers} parallel requests")
        pool = ThreadPoolExecutor(max_workers=workers)
        try:
            futures = {pool.submit(correct_range, a, b): i for i, (a, b) in enumerate(chunks)}
            for n, fut in enumerate(as_completed(futures), 1):
                results[futures[fut]] = fut.result()
                if progress:
                    progress(0.1 + 0.9 * n / total, f"Correcting domain terms ({n}/{total})")
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        pool.shutdown()
    corrections: list[Correction] = [c for i in range(total) for c in results[i]]

    # Drop exact duplicates (same segment + span).
    seen, unique = set(), []
    for c in corrections:
        key = (c.segment_id, c.before.strip().lower(), c.after.strip())
        if key not in seen:
            seen.add(key)
            unique.append(c)

    protected = {n for n in profile.get("names", []) if n[:1].isupper()}
    refined = apply_corrections(transcript, unique, protected)
    refined.model = model.endpoint.label
    return refined, unique, profile


def diff_html(raw: Transcript, refined: Transcript) -> str:
    """Word-level diff of changed segments, for side-by-side inspection in the UI."""
    rows = []
    refined_by_id = {s.id: s for s in refined.segments}
    for seg in raw.segments:
        new = refined_by_id.get(seg.id)
        if new is None or new.text == seg.text:
            continue
        a, b = seg.text.split(), new.text.split()
        out = []
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
            if op == "equal":
                out.append(html.escape(" ".join(a[i1:i2])))
            if op in ("replace", "delete"):
                out.append(f"<del>{html.escape(' '.join(a[i1:i2]))}</del>")
            if op in ("replace", "insert"):
                out.append(f"<ins>{html.escape(' '.join(b[j1:j2]))}</ins>")
        rows.append(f'<div class="diffrow"><span class="ts">[{_ts(seg.start)}] S{seg.id}</span> {" ".join(out)}</div>')
    if not rows:
        return '<div class="diffrow">No domain-term corrections were needed.</div>'
    return "\n".join(rows)


def _ts(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600:d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}" if seconds >= 3600 else \
        f"{seconds // 60:02d}:{seconds % 60:02d}"
