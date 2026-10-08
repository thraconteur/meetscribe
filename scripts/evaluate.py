"""Score a MeetScribe run against a ground-truth script and answer key.

    python scripts/evaluate.py runs/<run-folder> \
        --script samples/demo_meeting_script.txt --expected samples/demo_meeting_expected.json

Reports: word error rate of raw vs refined transcript, domain-term accuracy,
decision recall / false decisions, and action-item owner/deadline accuracy.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def reference_text(script: Path) -> str:
    out = []
    for line in script.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            out.append(line.split("|", 1)[-1].strip())
    return " ".join(out)


def norm(text: str) -> str:
    text = text.lower().replace("-", " ")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return " ".join(text.split())


def wer(ref: str, hyp: str) -> float:
    try:
        import jiwer

        return jiwer.wer(norm(ref), norm(hyp))
    except ImportError:  # small fallback
        r, h = norm(ref).split(), norm(hyp).split()
        d = list(range(len(h) + 1))
        for i in range(1, len(r) + 1):
            prev, d[0] = d[0], i
            for j in range(1, len(h) + 1):
                cur = d[j]
                d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
                prev = cur
        return d[len(h)] / max(1, len(r))


def term_accuracy(terms: list[str], ref: str, hyp: str) -> float:
    total = hit = 0
    for term in terms:
        pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])", re.I)
        n_ref = len(pattern.findall(ref))
        if n_ref:
            total += n_ref
            hit += min(n_ref, len(pattern.findall(hyp)))
    return hit / total if total else 1.0


def best(text: str, candidates: list[str]) -> tuple[int, float]:
    scores = [fuzz.token_set_ratio(norm(text), norm(c)) for c in candidates]
    if not scores:
        return -1, 0.0
    i = max(range(len(scores)), key=scores.__getitem__)
    return i, scores[i]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--script", default=str(ROOT / "samples" / "demo_meeting_script.txt"))
    ap.add_argument("--expected", default=str(ROOT / "samples" / "demo_meeting_expected.json"))
    args = ap.parse_args()

    run = Path(args.run_dir)
    record = json.loads((run / "meeting_record.json").read_text(encoding="utf-8"))
    transcripts = json.loads((run / "transcripts.json").read_text(encoding="utf-8"))
    expected = json.loads(Path(args.expected).read_text(encoding="utf-8"))
    ref = reference_text(Path(args.script))
    raw = " ".join(s["text"] for s in transcripts["raw"])
    refined = " ".join(s["text"] for s in transcripts["refined"])

    print("## Transcription\n")
    print("| Metric | Raw (STT) | Refined |\n|---|---|---|")
    print(f"| Word error rate (lower is better) | {wer(ref, raw):.1%} | {wer(ref, refined):.1%} |")
    terms = expected.get("domain_terms", [])
    print(f"| Domain-term accuracy ({len(terms)} terms) | {term_accuracy(terms, ref, raw):.0%} | "
          f"{term_accuracy(terms, ref, refined):.0%} |")

    print("\n## Decisions\n")
    got = [d["decision"] for d in record["decisions"]]
    found = 0
    for d in expected.get("decisions", []):
        i, s = best(d, got)
        ok = s >= 60
        found += ok
        print(f"- {'✓' if ok else '✗'} expected: {d}  →  {got[i] if i >= 0 else '—'} ({s:.0f})")
    false_dec = 0
    for nd in expected.get("not_decisions", []):
        i, s = best(nd, got)
        if s >= 75:
            false_dec += 1
            print(f"- ✗ presented as a decision but was NOT agreed: {got[i]}")
    print(f"\nDecision recall: {found}/{len(expected.get('decisions', []))} · proposals wrongly shown as decisions: "
          f"{false_dec} · total decisions reported: {len(got)}")

    print("\n## Action items\n")
    tasks = record["action_items"]
    names = [t["task"] for t in tasks]
    print("| Expected task | Matched | Owner ok | Deadline ok |\n|---|---|---|---|")
    stats = {"matched": 0, "owner": 0, "deadline": 0}
    for e in expected.get("action_items", []):
        i, s = best(e["task"], names)
        if i < 0 or s < 55:
            print(f"| {e['task']} | ✗ | – | – |")
            continue
        t = tasks[i]
        stats["matched"] += 1
        owner_ok = (not t["owner_stated"]) if e["owner"] is None else (
            t["owner_stated"] and e["owner"].lower() in t["owner"].lower())
        deadline_ok = (not t["deadline_stated"]) if e["deadline"] is None else (
            t["deadline_stated"] and fuzz.token_set_ratio(norm(e["deadline"]), norm(t["deadline"])) >= 60)
        stats["owner"] += owner_ok
        stats["deadline"] += deadline_ok
        print(f"| {e['task']} | {t['id']} ({s:.0f}) | {'✓' if owner_ok else '✗ ' + t['owner']} | "
              f"{'✓' if deadline_ok else '✗ ' + t['deadline']} |")
    n = len(expected.get("action_items", []))
    print(f"\nTasks found: {stats['matched']}/{n} · owners correct: {stats['owner']}/{n} · "
          f"deadlines correct: {stats['deadline']}/{n} · total tasks reported: {len(tasks)}")


if __name__ == "__main__":
    main()
