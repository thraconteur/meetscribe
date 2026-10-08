"""Command-line runner: python cli.py path/to/meeting.mp3 [--hints "Kubernetes, OKR"]"""
from __future__ import annotations

import argparse
import sys

from meetscribe.config import load_settings, with_overrides
from meetscribe.pipeline import STAGE_LABELS, Pipeline


def main() -> int:
    ap = argparse.ArgumentParser(description="Process a meeting recording end to end.")
    ap.add_argument("audio", help="audio or video file")
    ap.add_argument("--hints", default="", help="comma-separated domain terms likely to be said")
    ap.add_argument("--title", default="", help="meeting title")
    ap.add_argument("--context", default="", help="background context (never treated as said in the meeting)")
    ap.add_argument("--out", default=None, help="output directory (default: runs/)")
    args = ap.parse_args()

    settings = load_settings()
    if args.out:
        from pathlib import Path
        settings = with_overrides(settings, output_dir=Path(args.out))

    last = None
    state = None
    for state in Pipeline(settings).run_iter(args.audio, hints=args.hints, title_hint=args.title,
                                             context=args.context):
        if state.stage != last and state.status == "running":
            print(f"→ {STAGE_LABELS[state.stage]}…", flush=True)
            last = state.stage
    if state is None or state.status != "done":
        print(f"\n✗ {state.error if state else 'nothing ran'}", file=sys.stderr)
        return 1
    rec = state.record
    print(f"\n✓ {rec.title}\n\n{rec.summary}\n")
    print(f"Decisions ({len(rec.decisions)}):")
    for d in rec.decisions:
        print(f"  {d.id}. {d.decision}")
    print(f"Action items ({len(rec.action_items)}):")
    for t in rec.action_items:
        print(f"  {t.id}. {t.task}  [owner: {t.owner} | deadline: {t.deadline}]")
    print(f"\nOutputs written to {state.run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
