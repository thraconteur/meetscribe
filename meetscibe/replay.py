"""Re-open a finished run from its saved files (no API calls). Used by the UI's "Open a previous run"
and as a demo-day fallback when the network or a provider is unavailable."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .audio import AudioInfo
from .pipeline import RunState
from .schema import Correction, MeetingRecord, PipelineError, Segment, Transcript

_FILES = {
    "raw_transcript": "raw_transcript.txt", "refined_transcript": "refined_transcript.txt",
    "record_md": "meeting_record.md", "record_html": "meeting_record.html", "record_json": "meeting_record.json",
    "tasks_csv": "action_items.csv", "corrections_csv": "corrections.csv", "transcripts_json": "transcripts.json",
    "bundle": "meeting_outputs.zip",
}


def list_runs(output_dir: Path) -> list[Path]:
    """Finished run folders, newest first."""
    if not output_dir.exists():
        return []
    runs = [p for p in output_dir.iterdir() if (p / "meeting_record.json").is_file()]
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)


def _transcript(rows: list[dict], duration: float, model: str) -> Transcript:
    return Transcript([Segment(int(r["id"]), float(r["start"]), float(r["end"]), str(r["text"])) for r in rows],
                      duration=duration, model=model)


def load_run(folder: Path) -> RunState:
    folder = Path(folder)
    try:
        record = MeetingRecord.model_validate(json.loads((folder / "meeting_record.json").read_text("utf-8")))
        tx = json.loads((folder / "transcripts.json").read_text("utf-8"))
    except FileNotFoundError as exc:
        raise PipelineError(f"“{folder.name}” is not a finished MeetScribe run (missing {Path(exc.filename).name}).") from exc
    except (ValueError, KeyError) as exc:
        raise PipelineError(f"“{folder.name}” could not be read: {exc}") from exc

    meta = record.metadata or {}
    duration = float(meta.get("duration_s") or 0.0)
    models = meta.get("models") or {}
    raw = _transcript(tx["raw"], duration, str(models.get("stt", "")))
    refined = _transcript(tx["refined"], duration, str(models.get("refine", "")))
    raw.notes = list(tx.get("dropped_segments") or [])

    corrections: list[Correction] = []
    csv_path = folder / "corrections.csv"
    if csv_path.is_file():
        with open(csv_path, encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                if not row or not (row.get("segment") or "").strip():
                    continue
                corrections.append(Correction(
                    segment_id=int(row["segment"].lstrip("S")), before=row.get("original", ""),
                    after=row.get("corrected", ""), category=row.get("category", "term"), reason=row.get("reason", ""),
                    applied=str(row.get("applied", "")).strip().lower() == "true",
                    rejected_because=row.get("rejected_because", "")))

    st = RunState(stage="export", status="done", message="Opened saved run", run_dir=folder,
                  raw=raw, refined=refined, corrections=corrections, record=record,
                  profile={"domain": meta.get("domain", ""), "glossary": meta.get("glossary", [])},
                  stage_times={k: float(v) for k, v in (meta.get("stage_seconds") or {}).items()},
                  audio_info=AudioInfo(path=folder / str(meta.get("source_file", "")), duration=duration, codec="",
                                       sample_rate=16000, channels=1, size_mb=0.0),
                  files={k: folder / name for k, name in _FILES.items() if (folder / name).is_file()})
    st.log = [f"Opened saved run {folder.name}"]
    return st
