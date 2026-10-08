"""The Gradio app's glue: process() must yield a tuple that matches the declared outputs, on success and on error."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from meetscribe.pipeline import Pipeline

from .test_pipeline import _tone, settings, fake_api  # noqa: F401  (fixtures)

N_OUTPUTS = 15  # status, header, download row, 6 download buttons, tabs, 5 tab bodies
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def app(monkeypatch, settings, fake_api):  # noqa: F811
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    spec = importlib.util.spec_from_file_location("meetscribe_app", ROOT / "app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "PIPELINE", Pipeline(settings))
    return mod


def test_ui_builds(app):
    assert app.build_ui() is not None


def test_process_streams_and_finishes(app, tmp_path):
    updates = list(app.process(str(_tone(tmp_path / "meeting.mp3")), None, "", "", ""))
    assert updates and all(len(u) == N_OUTPUTS for u in updates)
    final = updates[-1]
    assert "Saving outputs" in final[0] and "failed" not in final[0]          # tracker finished
    assert final[2]["visible"] is True                                        # download bar shown
    assert all(isinstance(b, dict) and b.get("value") for b in final[3:9])    # all 6 downloads have files
    assert all(Path(b["value"]).is_file() for b in final[3:9])
    assert final[9]["visible"] is True and all(final[10:15])                   # tabs shown and filled


def test_process_without_a_file_shows_a_clear_error(app):
    (update,) = list(app.process(None, None, "", "", ""))
    assert len(update) == N_OUTPUTS and "No recording provided" in update[0]


def test_process_with_a_bad_file_shows_the_error_in_the_tracker(app, tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("not audio")
    final = list(app.process(str(bad), None, "", "", ""))[-1]
    assert len(final) == N_OUTPUTS and "failed" in final[0]


def test_open_run_roundtrip(app, tmp_path):
    list(app.process(str(_tone(tmp_path / "meeting.mp3")), None, "", "", ""))
    app.SETTINGS = app.PIPELINE.settings               # the app lists runs from its own settings
    runs = app.list_runs(app.SETTINGS.output_dir)
    assert runs, "the finished run should be listed"
    out = app.open_run(runs[0].name)
    assert len(out) == N_OUTPUTS and out[9]["visible"] is True


def test_open_run_without_selection_is_a_clear_error(app):
    assert "Pick a saved run" in app.open_run(None)[0]
