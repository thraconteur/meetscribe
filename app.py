"""MeetScribe — interactive web interface (Gradio).

Run:  python app.py        then open http://127.0.0.1:7860
"""
from __future__ import annotations

import argparse
import os
import threading
import time
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr  # noqa: E402

from meetscribe import ui_views as v  # noqa: E402
from meetscribe.config import load_settings  # noqa: E402
from meetscribe.pipeline import Pipeline, RunState  # noqa: E402
from meetscribe.replay import list_runs, load_run  # noqa: E402
from meetscribe.schema import PipelineError  # noqa: E402

SETTINGS = load_settings()
PIPELINE = Pipeline(SETTINGS)
SAMPLES = sorted(p for p in Path(__file__).parent.joinpath("samples").glob("*")
                 if p.suffix.lower() in {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm"})
DOWNLOADS = ["bundle", "record_html", "record_json", "tasks_csv", "raw_transcript", "refined_transcript"]


def _hero() -> str:
    stt = (f"faster-whisper {SETTINGS.local_whisper_model} (local)" if SETTINGS.stt_backend == "local"
           else SETTINGS.stt.model)
    return v.hero_html(stt, SETTINGS.refine.model, SETTINGS.document.model)


def _outputs(st: RunState | None, started: float | None):
    """One tuple for every UI update: status, header, download bar (row + 6 buttons), tabs, 5 tab bodies."""
    if st is None:
        buttons = [gr.update(value=None)] * len(DOWNLOADS)
        return (v.tracker_html(None), v.empty_html(SETTINGS.max_upload_mb, SETTINGS.max_duration_min),
                gr.update(visible=False), *buttons, gr.update(visible=False), "", "", "", "", "")
    files = st.files or {}
    buttons = [gr.update(value=str(files[k])) if k in files else gr.update(value=None) for k in DOWNLOADS]
    has_results = st.raw is not None
    return (
        v.tracker_html(st, started),
        v.header_html(st) if has_results else v.empty_html(SETTINGS.max_upload_mb, SETTINGS.max_duration_min),
        gr.update(visible=bool(files)), *buttons,
        gr.update(visible=has_results),
        v.overview_html(st), v.actions_html(st), v.compare_html(st), v.transcripts_html(st), v.checks_html(st),
    )


def process(upload, recording, hints, title, context):
    path = upload or recording
    if not path:
        yield _outputs(RunState(status="error", error="No recording provided. Upload an audio or video file, or record one.",
                                failed_stage="validate"), None)
        return
    holder: dict = {"state": RunState(message="Starting…")}

    def worker():
        for state in PIPELINE.run_iter(path, hints=hints or "", title_hint=title or "", context=context or ""):
            holder["state"] = state

    started = time.time()
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    last_key = None
    while thread.is_alive():
        st = holder["state"]
        key = (st.stage, st.message, len(st.stage_times), st.raw is not None, st.refined is not None,
               len(st.log), int(time.time() - started))
        if key != last_key:
            last_key = key
            yield _outputs(st, started)
        time.sleep(0.4)
    thread.join()
    yield _outputs(holder["state"], started)


def open_run(name):
    if not name:
        return _outputs(RunState(status="error", error="Pick a saved run first.", failed_stage="validate"), None)
    try:
        return _outputs(load_run(SETTINGS.output_dir / name), None)
    except PipelineError as exc:
        return _outputs(RunState(status="error", error=exc.user_message, failed_stage="validate"), None)


def _run_choices():
    return gr.update(choices=[p.name for p in list_runs(SETTINGS.output_dir)])


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="MeetScribe — AI meeting assistant") as demo:
        gr.HTML(_hero())
        with gr.Row(equal_height=False):
            with gr.Column(scale=3, min_width=330, elem_id="left"):
                upload = gr.File(label="Meeting recording (audio or video)", type="filepath", file_count="single")
                with gr.Accordion("Record from microphone instead", open=False):
                    recording = gr.Audio(sources=["microphone"], type="filepath", label="Record")
                hints = gr.Textbox(label="Terms to expect (optional)", lines=2,
                                   placeholder="Comma-separated, e.g. Kubernetes, kubectl, OKR, Razorpay")
                with gr.Accordion("Meeting details (optional)", open=False):
                    title = gr.Textbox(label="Meeting title", placeholder="e.g. Sprint 14 planning")
                    context = gr.Textbox(label="Background", lines=2,
                                         placeholder="Helps with jargon only. Never treated as said in the meeting.")
                run = gr.Button("Process meeting", variant="primary", size="lg")
                status = gr.HTML(v.tracker_html(None))
                with gr.Accordion("Open a previous run", open=False):
                    runs = gr.Dropdown(label="Saved runs", choices=[p.name for p in list_runs(SETTINGS.output_dir)])
                    open_btn = gr.Button("Open", size="sm")
                if SAMPLES:
                    gr.Examples([[str(p)] for p in SAMPLES], inputs=[upload], label="Sample recordings")
            with gr.Column(scale=9, min_width=480):
                header = gr.HTML(v.empty_html(SETTINGS.max_upload_mb, SETTINGS.max_duration_min))
                with gr.Row(visible=False, elem_id="dl-row") as dl_row:
                    dl_buttons = [
                        gr.DownloadButton("Everything (.zip)", variant="primary", size="sm"),
                        gr.DownloadButton("Minutes (.html)", size="sm"),
                        gr.DownloadButton("Record (.json)", size="sm"),
                        gr.DownloadButton("Action items (.csv)", size="sm"),
                        gr.DownloadButton("Raw transcript", size="sm"),
                        gr.DownloadButton("Refined transcript", size="sm"),
                    ]
                with gr.Tabs(visible=False) as tabs:
                    with gr.Tab("Overview"):
                        overview = gr.HTML()
                    with gr.Tab("Decisions & action items"):
                        actions = gr.HTML()
                    with gr.Tab("Corrections"):
                        compare = gr.HTML()
                    with gr.Tab("Transcripts"):
                        transcripts = gr.HTML()
                    with gr.Tab("Checks"):
                        checks = gr.HTML()
        outputs = [status, header, dl_row, *dl_buttons, tabs, overview, actions, compare, transcripts, checks]
        run.click(process, inputs=[upload, recording, hints, title, context], outputs=outputs, concurrency_limit=2)
        open_btn.click(open_run, inputs=[runs], outputs=outputs)
        demo.load(_run_choices, outputs=[runs])
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MeetScribe web interface")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="create a temporary public gradio.live link")
    args = parser.parse_args()
    theme = gr.themes.Base(
        primary_hue="cyan", neutral_hue="slate",
        font=[gr.themes.GoogleFont("IBM Plex Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"])
    build_ui().queue().launch(server_name=args.host, server_port=args.port, share=args.share,
                              css=v.CSS, head=v.HEAD, theme=theme,
                              allowed_paths=[str(SETTINGS.output_dir.resolve())])
