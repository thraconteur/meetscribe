"""Audio validation, normalisation and silence-aware chunking (ffmpeg/ffprobe)."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .schema import PipelineError

SUPPORTED_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".mp4", ".mpeg", ".mpga", ".webm", ".ogg", ".oga",
    ".opus", ".flac", ".aac", ".wma", ".amr", ".mkv", ".mov", ".3gp", ".aiff", ".aif",
}

SILENT_MAX_VOLUME_DB = -50.0  # anything quieter than this everywhere is treated as silence
MIN_DURATION_S = 1.0


@dataclass
class AudioInfo:
    path: Path
    duration: float
    codec: str
    sample_rate: int
    channels: int
    size_mb: float


def _require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise PipelineError(
                f"'{tool}' was not found on this machine. Install ffmpeg (https://ffmpeg.org/download.html) "
                "and make sure it is on your PATH.",
                stage="audio",
            )


def _run(cmd: list[str], timeout: float = 900) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def probe(path: Path) -> dict:
    proc = _run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)], 120)
    if proc.returncode != 0:
        return {}
    try:
        return json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return {}


def validate_input(path: str | Path, max_upload_mb: float = 500.0, max_duration_min: float = 180.0) -> AudioInfo:
    """Check that the upload is a readable, non-empty English meeting recording candidate.

    Raises PipelineError with a message written for the end user.
    """
    _require_ffmpeg()
    if path is None or str(path).strip() == "":
        raise PipelineError("No file was uploaded. Choose a meeting recording and try again.", "upload")
    path = Path(path)
    if not path.exists():
        raise PipelineError(f"The uploaded file '{path.name}' could not be found on the server.", "upload")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        nice = ", ".join(sorted(e.lstrip(".") for e in SUPPORTED_EXTENSIONS))
        raise PipelineError(
            f"Unsupported file type '{ext or '(no extension)'}'. Upload an audio or video recording ({nice}).",
            "upload",
        )

    size_mb = path.stat().st_size / 1_048_576
    if size_mb == 0:
        raise PipelineError(f"'{path.name}' is empty (0 bytes). Upload the actual recording.", "upload")
    if size_mb > max_upload_mb:
        raise PipelineError(
            f"'{path.name}' is {size_mb:.0f} MB, above the {max_upload_mb:.0f} MB limit. "
            "Trim the recording or export it as compressed audio (e.g. MP3/M4A).",
            "upload",
        )

    meta = probe(path)
    audio_streams = [s for s in meta.get("streams", []) if s.get("codec_type") == "audio"]
    if not meta or not audio_streams:
        raise PipelineError(
            f"'{path.name}' could not be read as audio — the file may be corrupted, truncated, or contain no "
            "audio track.",
            "upload",
        )
    stream = audio_streams[0]
    duration = _duration(meta, stream)
    if duration is None or duration <= 0:
        # Some containers do not report duration; decode to find out.
        duration = _decode_duration(path)
    if duration is None:
        raise PipelineError(f"'{path.name}' could not be decoded. It may be corrupted.", "upload")
    if duration < MIN_DURATION_S:
        raise PipelineError(
            f"'{path.name}' is only {duration:.1f} s long — too short to contain a meeting.", "upload"
        )
    if duration > max_duration_min * 60:
        raise PipelineError(
            f"'{path.name}' is {duration / 60:.0f} minutes long; the limit is {max_duration_min:.0f} minutes.",
            "upload",
        )

    return AudioInfo(
        path=path,
        duration=duration,
        codec=str(stream.get("codec_name", "?")),
        sample_rate=int(stream.get("sample_rate", 0) or 0),
        channels=int(stream.get("channels", 0) or 0),
        size_mb=size_mb,
    )


def _duration(meta: dict, stream: dict) -> float | None:
    for source in (stream.get("duration"), meta.get("format", {}).get("duration")):
        try:
            if source is not None:
                return float(source)
        except (TypeError, ValueError):
            continue
    return None


def _decode_duration(path: Path) -> float | None:
    proc = _run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-vn", "-f", "null", "-"])
    times = re.findall(r"time=(\d+):(\d+):(\d+\.?\d*)", proc.stderr)
    if proc.returncode != 0 or not times:
        return None
    h, m, s = times[-1]
    return int(h) * 3600 + int(m) * 60 + float(s)


def normalise(src: Path, out_dir: Path, enhance: bool = True) -> Path:
    """Convert to 16 kHz mono FLAC (Whisper's native rate, lossless, ~4x smaller than WAV).

    With ``enhance`` a gentle high-pass removes rumble/hum and dynamic loudness
    normalisation evens out quiet and loud speakers — common in meeting rooms
    where people sit at different distances from the microphone.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "audio_16k.flac"
    filters = "highpass=f=70,dynaudnorm=f=200:g=11:p=0.9:m=8" if enhance else "anull"
    proc = _run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
         "-af", filters, "-c:a", "flac", str(dst)]
    )
    if proc.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        raise PipelineError(
            "The recording could not be decoded (ffmpeg error: "
            f"{(proc.stderr or 'unknown').strip().splitlines()[-1][:200]}).",
            "audio",
        )
    return dst


def max_volume_db(path: Path) -> float:
    proc = _run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"])
    m = re.search(r"max_volume:\s*(-?[\d.]+|-inf) dB", proc.stderr)
    if not m:
        return 0.0
    return float("-inf") if m.group(1) == "-inf" else float(m.group(1))


def assert_not_silent(src: Path) -> None:
    """Fail early (before paying for API calls) if the recording is digital silence."""
    if max_volume_db(src) < SILENT_MAX_VOLUME_DB:
        raise PipelineError(
            "The recording appears to be silent — no audible speech was detected. Check that the right file "
            "was uploaded and that the microphone was not muted.",
            "audio",
        )


def silences(path: Path, noise_db: int = -35, min_silence: float = 0.35) -> list[tuple[float, float]]:
    proc = _run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af",
         f"silencedetect=noise={noise_db}dB:d={min_silence}", "-f", "null", "-"]
    )
    starts = [float(x) for x in re.findall(r"silence_start:\s*(-?[\d.]+)", proc.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end:\s*([\d.]+)", proc.stderr)]
    return list(zip(starts, ends))


def plan_chunks(duration: float, chunk_seconds: float, silent_spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Choose chunk boundaries at pauses near every ``chunk_seconds`` so no word is cut in half."""
    if duration <= chunk_seconds * 1.15:
        return [(0.0, duration)]
    mids = sorted((a + b) / 2 for a, b in silent_spans if b - a >= 0.3)
    bounds = [0.0]
    while duration - bounds[-1] > chunk_seconds * 1.15:
        target = bounds[-1] + chunk_seconds
        window = [m for m in mids if target - 0.25 * chunk_seconds <= m <= target + 0.1 * chunk_seconds]
        cut = min(window, key=lambda m: abs(m - target)) if window else target
        bounds.append(cut)
    bounds.append(duration)
    return list(zip(bounds[:-1], bounds[1:]))


def cut_chunks(path: Path, spans: list[tuple[float, float]], out_dir: Path) -> list[tuple[Path, float]]:
    """Write each span to its own FLAC file. Returns (file, start_offset) pairs."""
    if len(spans) == 1:
        return [(path, 0.0)]
    out_dir.mkdir(parents=True, exist_ok=True)
    out: list[tuple[Path, float]] = []
    for i, (a, b) in enumerate(spans):
        dst = out_dir / f"chunk_{i:03d}.flac"
        proc = _run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{a:.3f}", "-to", f"{b:.3f}",
             "-i", str(path), "-c:a", "flac", str(dst)]
        )
        if proc.returncode != 0:
            raise PipelineError(f"Failed to split the recording into chunks: {proc.stderr.strip()[:200]}", "audio")
        out.append((dst, a))
    return out
