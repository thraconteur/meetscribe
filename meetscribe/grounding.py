"""Deterministic verification of the documentation model's output against the transcript.

The documentation LM is asked to cite verbatim evidence. This module checks
every citation with fuzzy matching and enforces the PS rules in code rather
than trusting the model:

* a decision / task / open item whose evidence cannot be found is removed;
* an owner is kept only if its quoted evidence exists, sits near the task,
  and actually contains that name (pronouns are never owners);
* a deadline is kept only if its quoted evidence exists near the task and
  contains the deadline's words and numbers;
* anything removed or downgraded is written to the verification log.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz

from .schema import (ActionItem, Decision, Evidence, MeetingRecord, MinutesSection, OpenItem,
                     Transcript)
from .textutil import normalise_for_match, number_values

PRONOUNS = {"i", "me", "we", "us", "you", "he", "she", "they", "them", "him", "her", "someone", "somebody",
            "anyone", "anybody", "speaker", "unknown", "tbd", "n/a", "na", "none", "unspecified", "null",
            "not specified", "unassigned"}
DEADLINE_STOPWORDS = {"by", "the", "of", "before", "on", "at", "in", "to", "a", "an", "until", "till", "no", "later",
                      "than", "this", "end", "within"}
HEDGES = ("maybe", "might", "perhaps", "could we", "we could", "think about", "should we", "what if", "not sure",
          "consider", "wondering", "possibly", "one option", "thinking of", "we should probably", "let's discuss")
AGREEMENT = ("agreed", "agree", "decided", "approved", "confirmed", "settled", "final call", "go with", "going with",
             "let's go", "locked", "we will", "we'll", "sounds good", "signed off", "done deal", "let's do")
NEARBY_OWNER = 3      # owner quote must sit within this many segments of the task discussion
NEARBY_DEADLINE = 4


@dataclass
class Match:
    segment_ids: list[int]
    start: float
    score: float


class EvidenceIndex:
    def __init__(self, transcript: Transcript):
        self.segments = transcript.segments
        self.norm = [normalise_for_match(s.text) for s in self.segments]
        self.position = {s.id: i for i, s in enumerate(self.segments)}

    def find(self, quote: str | None, hint_ids: list[int] | None = None) -> Match | None:
        q = normalise_for_match(quote or "")
        if len(q) < 4:
            return None
        n = len(self.segments)
        q_len = len(q)
        per_width: list[Match] = []
        for width in (1, 2, 3, 4):
            best: Match | None = None
            for i in range(0, max(1, n - width + 1)):
                window = " ".join(self.norm[i : i + width])
                if not window:
                    continue
                score = fuzz.partial_ratio(q, window) if q_len <= len(window) else fuzz.ratio(q, window)
                ids = [s.id for s in self.segments[i : i + width]]
                if best is None or score > best.score or (
                    score == best.score and hint_ids and set(ids) & set(hint_ids)
                ):
                    best = Match(ids, self.segments[i].start, score)
            if best:
                per_width.append(best)
                if best.score >= 99.5:
                    break  # an exact match in the tightest window — no need to widen
        if not per_width:
            return None
        top = max(m.score for m in per_width)
        # Prefer the narrowest window that is (almost) as good as the best — precise segment + timestamp.
        chosen = next(m for m in per_width if m.score >= top - 2.0)
        if len(chosen.segment_ids) > 1:
            # Start the citation at the first segment that actually overlaps the quote.
            original_ids = list(chosen.segment_ids)
            for sid in original_ids[:-1]:
                pos = self.position[sid]
                if fuzz.partial_ratio(self.norm[pos], q) >= 80 or fuzz.partial_ratio(q, self.norm[pos]) >= 80:
                    break
                chosen.segment_ids.remove(sid)
            if chosen.segment_ids:
                chosen.start = self.segments[self.position[chosen.segment_ids[0]]].start
        return chosen

    def distance(self, a: Match, b: Match) -> int:
        pa = [self.position[i] for i in a.segment_ids]
        pb = [self.position[i] for i in b.segment_ids]
        return max(0, min(pb) - max(pa), min(pa) - max(pb))

    def segment_text(self, seg_ids: list[int]) -> str:
        return " ".join(self.norm[self.position[i]] for i in seg_ids if i in self.position)


def _int_list(value) -> list[int]:
    out = []
    for v in value or []:
        try:
            out.append(int(str(v).strip().lstrip("[Ss").rstrip("]")))
        except ValueError:
            continue
    return out


def _clean(value) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in PRONOUNS:
        return None
    return s


def _owner_supported(owner: str, owner_quote_norm: str) -> bool:
    name_tokens = [t for t in normalise_for_match(owner).split() if t not in {"the", "team", "and", "of"}]
    if not name_tokens:
        name_tokens = normalise_for_match(owner).split()
    if not name_tokens or all(t in PRONOUNS for t in name_tokens):
        return False
    quote_tokens = set(owner_quote_norm.split())
    hits = 0
    for t in name_tokens:
        if t in quote_tokens or any(fuzz.ratio(t, q) >= 85 for q in quote_tokens if len(q) > 2):
            hits += 1
    return hits >= max(1, (len(name_tokens) + 1) // 2)


def _deadline_supported(deadline: str, deadline_quote_norm: str) -> bool:
    said = number_values(deadline_quote_norm)
    if any(v not in said for v in number_values(deadline)):
        return False  # every number in the deadline must have been said
    words = [t for t in normalise_for_match(deadline).split() if t not in DEADLINE_STOPWORDS and not t.isdigit()]
    if not words:
        return bool(number_values(deadline))
    quote_tokens = set(deadline_quote_norm.split())
    hits = sum(1 for w in words if w in quote_tokens or any(fuzz.ratio(w, q) >= 85 for q in quote_tokens))
    return hits / len(words) >= 0.6


def _verify_evidence(index: EvidenceIndex, quote, segments, threshold: float) -> tuple[Evidence | None, Match | None]:
    quote = (quote or "").strip().strip('"“”')
    match = index.find(quote, _int_list(segments))
    if match is None or match.score < threshold:
        return None, match
    return Evidence(quote=quote, segment_ids=match.segment_ids, start=match.start, verified=True), match


def _ungrounded_numbers(text: str, transcript_text: str, transcript_numbers) -> list[str]:
    """Digits the model wrote that were never said. Digit tokens only, so wording like 'first' can't false-positive."""
    bad = []
    for tok in re.findall(r"\d+(?:[.,]\d+)*", text or ""):
        if tok in transcript_text:
            continue
        if not all(v in transcript_numbers for v in number_values(tok)):
            bad.append(tok)
    return bad


def _is_only_a_proposal(quote: str) -> bool:
    """Evidence that hedges and contains no sign of agreement cannot back a *decision*."""
    q = " " + normalise_for_match(quote) + " "
    return any(f" {h} " in q for h in HEDGES) and not any(f" {a} " in q for a in AGREEMENT)


def _dedupe(items: list, key) -> list:
    out = []
    for it in items:
        if any(fuzz.token_sort_ratio(normalise_for_match(key(it)), normalise_for_match(key(o))) >= 90 for o in out):
            continue
        out.append(it)
    return out



_DASH = "‐‑‒–—\\-"
_REF = rf"S\d+(?:\s*[{_DASH}]\s*S?\d+)?"
_REF_RE = re.compile(rf"\s*[\(\[]\s*{_REF}(?:\s*[,;]\s*{_REF})*\s*[\)\]]"
                     rf"|(?<=\s)S\d+\s*[{_DASH}]\s*S?\d+(?=[\s.,;:)]|$)")


def strip_segment_refs(text: str, transcript_text: str) -> str:
    """Models sometimes leave internal references like “(S108‑S119)” in prose. They are not part of the record.
    A reference is kept only if the meeting itself said that exact token (e.g. “S3 bucket”)."""
    def repl(m: re.Match) -> str:
        ids = re.findall(r"S(\d+)", m.group(0))
        if any(re.search(rf"\bS{i}\b", transcript_text, re.I) for i in ids):
            return m.group(0)
        return ""
    return re.sub(r"\s{2,}", " ", _REF_RE.sub(repl, text)).strip()


def _scrub(raw: dict, transcript_text: str) -> dict:
    """Copy of the model output with references stripped from the prose fields. Quotes (evidence) are left untouched."""
    clean = lambda v: strip_segment_refs(str(v), transcript_text) if isinstance(v, str) else v  # noqa: E731
    out = dict(raw)
    for key in ("title", "summary"):
        if key in out:
            out[key] = clean(out[key])
    for key, field in (("decisions", "decision"), ("action_items", "task"), ("open_items", "item")):
        out[key] = [dict(x, **{field: clean(x.get(field, ""))}) if isinstance(x, dict) else x for x in raw.get(key) or []]
    mins = []
    for sec in raw.get("minutes") or raw.get("topics") or []:
        if isinstance(sec, dict):
            mins.append(dict(sec, topic=clean(sec.get("topic", "")), points=[clean(p) for p in sec.get("points") or []]))
    out["minutes"] = mins
    return out


def _grounded_summary(summary: str, transcript_text: str, transcript_numbers, log: list[str]) -> str:
    kept = []
    for sentence in re.split(r"(?<=[.!?])\s+", summary.strip()):
        bad = _ungrounded_numbers(sentence, transcript_text, transcript_numbers)
        if bad:
            log.append(f"Removed a summary sentence — it states {', '.join(bad)}, which was never said.")
        elif sentence:
            kept.append(sentence)
    return " ".join(kept)


def verify_record(raw: dict, transcript: Transcript, threshold: float = 82.0) -> MeetingRecord:
    raw = _scrub(raw, transcript.text)
    index = EvidenceIndex(transcript)
    log: list[str] = []
    transcript_norm = normalise_for_match(transcript.text)
    transcript_tokens = set(transcript_norm.split())
    transcript_numbers = set(number_values(transcript.text))

    # --- decisions -------------------------------------------------------
    decisions: list[Decision] = []
    demoted: list[OpenItem] = []
    for d in raw.get("decisions") or []:
        if not isinstance(d, dict) or not str(d.get("decision", "")).strip():
            continue
        ev, m = _verify_evidence(index, d.get("evidence"), d.get("segments"), threshold)
        if ev is None:
            log.append(f"Removed decision “{d.get('decision')}” — its supporting quote was not found in the "
                       f"transcript (best match {m.score:.0f}%)." if m else
                       f"Removed decision “{d.get('decision')}” — no supporting quote was given.")
            continue
        bad = _ungrounded_numbers(str(d["decision"]), transcript.text, transcript_numbers)
        if bad:
            log.append(f"Removed decision “{d.get('decision')}” — it states {', '.join(bad)}, which was never said.")
            continue
        if _is_only_a_proposal(ev.quote):
            log.append(f"Moved “{d.get('decision')}” from decisions to open items — the quoted words are tentative, "
                       "with no sign the meeting agreed.")
            demoted.append(OpenItem(item=str(d["decision"]).strip(), kind="proposal", evidence=ev))
            continue
        decisions.append(Decision(decision=str(d["decision"]).strip(), evidence=ev))
    decisions = _dedupe(decisions, lambda x: x.decision)

    # --- action items ----------------------------------------------------
    tasks: list[ActionItem] = []
    for t in raw.get("action_items") or []:
        if not isinstance(t, dict) or not str(t.get("task", "")).strip():
            continue
        task_text = str(t["task"]).strip()
        ev, m = _verify_evidence(index, t.get("evidence"), t.get("segments"), threshold)
        if ev is None:
            log.append(f"Removed task “{task_text}” — its supporting quote was not found in the transcript.")
            continue
        bad = _ungrounded_numbers(task_text, transcript.text, transcript_numbers)
        if bad:
            log.append(f"Removed task “{task_text}” — it states {', '.join(bad)}, which was never said.")
            continue
        item = ActionItem(task=task_text, evidence=ev)

        owner = _clean(t.get("owner"))
        if owner:
            oq = (t.get("owner_evidence") or "").strip()
            om = index.find(oq) if oq else None
            reason = ""
            if not oq:
                reason = "no quote naming the owner"
            elif om is None or om.score < threshold:
                reason = "the owner quote is not in the transcript"
            elif index.distance(om, m) > NEARBY_OWNER:
                reason = "the owner quote is not near the task discussion"
            elif not _owner_supported(owner, normalise_for_match(oq)):
                reason = "the quote does not contain that name"
            elif not any(tok in transcript_tokens for tok in normalise_for_match(owner).split()):
                reason = "that name never occurs in the transcript"
            if reason:
                log.append(f"Task “{task_text}”: owner “{owner}” set to Unspecified — {reason}.")
            else:
                item.owner, item.owner_stated, item.owner_evidence = owner, True, oq
        elif t.get("owner"):
            log.append(f"Task “{task_text}”: owner “{t.get('owner')}” is not a named person/team — set to Unspecified.")

        deadline = _clean(t.get("deadline"))
        if deadline:
            dq = (t.get("deadline_evidence") or "").strip()
            dm = index.find(dq) if dq else None
            reason = ""
            if not dq:
                reason = "no quote stating the deadline"
            elif dm is None or dm.score < threshold:
                reason = "the deadline quote is not in the transcript"
            elif index.distance(dm, m) > NEARBY_DEADLINE:
                reason = "the deadline quote is not near the task discussion"
            elif not _deadline_supported(deadline, normalise_for_match(dq)):
                reason = "the quote does not state that deadline"
            if reason:
                log.append(f"Task “{task_text}”: deadline “{deadline}” set to Unspecified — {reason}.")
            else:
                item.deadline, item.deadline_stated, item.deadline_evidence = deadline, True, dq
        tasks.append(item)
    tasks = _dedupe(tasks, lambda x: x.task)

    # --- open items ------------------------------------------------------
    open_items: list[OpenItem] = []
    decided = [d.decision for d in decisions]
    for o in raw.get("open_items") or []:
        if not isinstance(o, dict) or not str(o.get("item", "")).strip():
            continue
        ev, _ = _verify_evidence(index, o.get("evidence"), o.get("segments"), threshold)
        if ev is None:
            log.append(f"Removed open item “{o.get('item')}” — its supporting quote was not found.")
            continue
        text = str(o["item"]).strip()
        if any(fuzz.token_sort_ratio(normalise_for_match(text), normalise_for_match(d)) >= 85 for d in decided):
            continue  # it was settled — already a decision
        kind = str(o.get("kind", "unresolved")).lower()
        open_items.append(OpenItem(item=text, kind=kind if kind in ("proposal", "question", "unresolved")
                                   else "unresolved", evidence=ev))
    open_items = _dedupe(demoted + open_items, lambda x: x.item)

    # --- participants: only names that occur in the transcript -----------
    participants = []
    for p in raw.get("participants") or []:
        name = _clean(p)
        if not name:
            continue
        toks = [tok for tok in normalise_for_match(name).split() if len(tok) > 1]
        if toks and any(tok in transcript_tokens for tok in toks):
            if name not in participants:
                participants.append(name)
        else:
            log.append(f"Removed participant “{name}” — name not found in the transcript.")

    minutes = []
    for sec in raw.get("minutes") or raw.get("topics") or []:
        if isinstance(sec, dict) and sec.get("topic"):
            points = []
            for p in sec.get("points") or []:
                p = str(p).strip()
                if not p:
                    continue
                bad = _ungrounded_numbers(p, transcript.text, transcript_numbers)
                if bad:
                    log.append(f"Removed a minutes point — it states {', '.join(bad)}, which was never said: “{p[:80]}”")
                    continue
                points.append(p)
            minutes.append(MinutesSection(topic=str(sec["topic"]).strip(), points=points))

    for i, d in enumerate(decisions, 1):
        d.id = f"D{i}"
    for i, t in enumerate(tasks, 1):
        t.id = f"T{i}"

    if not log:
        log.append("All decisions, tasks, owners and deadlines were matched to quotes in the transcript.")

    return MeetingRecord(
        title=str(raw.get("title") or "Meeting record").strip()[:140],
        summary=_grounded_summary(str(raw.get("summary") or ""), transcript.text, transcript_numbers, log),
        participants=participants,
        minutes=minutes,
        decisions=decisions,
        action_items=tasks,
        open_items=open_items,
        verification_log=log,
    )
