"""Synthesise the shareable demo meeting (samples/demo_meeting.mp3) from its script.

Uses Microsoft Edge's free neural voices (needs internet):
    pip install edge-tts
    python scripts/make_sample_meeting.py            # adds light office noise by default
    python scripts/make_sample_meeting.py --clean    # no noise

A real recording of your team reading the script (or any real meeting) is an
even better demo — the evaluator works with any file + script pair.
"""
from __future__ import annotations

import argparse
import asyncio
import random
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VOICES = {
    "Ananya": "en-IN-NeerjaNeural",
    "Rohan": "en-IN-PrabhatNeural",
    "Meera": "en-US-JennyNeural",
    "Kabir": "en-GB-RyanNeural",
}


def read_script(path: Path) -> list[tuple[str, str]]:
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        speaker, text = raw.split("|", 1)
        lines.append((speaker.strip(), text.strip()))
    return lines


async def synth(lines, workdir: Path) -> list[Path]:
    import edge_tts

    files = []
    for i, (speaker, text) in enumerate(lines):
        out = workdir / f"{i:03d}.mp3"
        rate = random.choice(["+0%", "+5%", "-5%", "+8%"])
        await edge_tts.Communicate(text, VOICES.get(speaker, "en-US-GuyNeural"), rate=rate).save(str(out))
        files.append(out)
        print(f"  {speaker}: {text[:60]}…")
    return files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default=str(ROOT / "samples" / "demo_meeting_script.txt"))
    ap.add_argument("--out", default=str(ROOT / "samples" / "demo_meeting.mp3"))
    ap.add_argument("--clean", action="store_true", help="do not add background noise")
    args = ap.parse_args()
    random.seed(7)
    lines = read_script(Path(args.script))
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        clips = asyncio.run(synth(lines, tmp))
        # Natural pauses between turns.
        concat = tmp / "list.txt"
        entries = []
        for i, clip in enumerate(clips):
            wav = tmp / f"{i:03d}.wav"  # same codec/rate for every piece so the concat demuxer is happy
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(clip), "-ar", "24000", "-ac", "1",
                            "-c:a", "pcm_s16le", str(wav)], check=True)
            gap = tmp / f"gap{i}.wav"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                            "anullsrc=r=24000:cl=mono", "-t", f"{random.uniform(0.35, 0.9):.2f}",
                            "-c:a", "pcm_s16le", str(gap)], check=True)
            entries += [f"file '{wav}'", f"file '{gap}'"]
        concat.write_text("\n".join(entries))
        speech = tmp / "speech.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(concat),
                        "-ar", "24000", "-ac", "1", str(speech)], check=True)
        if args.clean:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(speech), "-b:a", "96k", args.out],
                           check=True)
        else:
            # Pink noise at a low level + a slight room reverb to make it less studio-clean.
            subprocess.run([
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(speech), "-f", "lavfi", "-i",
                "anoisesrc=color=pink:amplitude=0.012:sample_rate=24000",
                "-filter_complex", "[0]aecho=0.8:0.5:40:0.18[s];[s][1]amix=inputs=2:duration=first:normalize=0",
                "-b:a", "96k", args.out], check=True)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
