"""Run loader and HTML views: round-trip fidelity and escaping of untrusted text."""
from __future__ import annotations

import re

from meetscribe import ui_views as v
from meetscribe.pipeline import Pipeline
from meetscribe.replay import list_runs, load_run
from meetscribe.schema import PipelineError

from .test_pipeline import _tone, settings, fake_api  # noqa: F401  (fixtures)

import pytest


@pytest.fixture
def finished(tmp_path, settings, fake_api):  # noqa: F811
    st = Pipeline(settings).run(_tone(tmp_path / "meeting.mp3"))
    assert st.status == "done", st.error
    return st


def test_reopened_run_matches_the_original(finished):
    re_st = load_run(finished.run_dir)
    assert re_st.status == "done"
    assert [s.text for s in re_st.raw.segments] == [s.text for s in finished.raw.segments]
    assert [s.text for s in re_st.refined.segments] == [s.text for s in finished.refined.segments]
    assert [(c.segment_id, c.before, c.after, c.applied) for c in re_st.corrections] == \
           [(c.segment_id, c.before, c.after, c.applied) for c in finished.corrections]
    assert [d.decision for d in re_st.record.decisions] == [d.decision for d in finished.record.decisions]
    assert [(t.task, t.owner, t.deadline) for t in re_st.record.action_items] == \
           [(t.task, t.owner, t.deadline) for t in finished.record.action_items]
    assert "bundle" in re_st.files and "record_json" in re_st.files


def test_list_runs_finds_finished_runs_only(finished, tmp_path):
    (finished.run_dir.parent / "half-done").mkdir()
    names = [p.name for p in list_runs(finished.run_dir.parent)]
    assert finished.run_dir.name in names and "half-done" not in names


def test_loading_a_non_run_folder_gives_a_clear_error(tmp_path):
    with pytest.raises(PipelineError):
        load_run(tmp_path)


def test_csv_has_no_doubled_carriage_returns(finished):
    for key in ("tasks_csv", "corrections_csv"):
        raw = finished.files[key].read_bytes()
        assert b"\r\r\n" not in raw


def test_all_views_render_for_a_finished_run(finished):
    for fn in (v.header_html, v.overview_html, v.actions_html, v.compare_html, v.transcripts_html, v.checks_html):
        assert fn(finished).strip()


def test_views_escape_untrusted_text(finished):
    evil = '<img src=x onerror=alert(1)><script>alert(2)</script>'
    st = load_run(finished.run_dir)
    st.raw.segments[0].text = evil
    st.refined.segments[0].text = evil + " changed"
    st.record.title = evil
    st.record.summary = evil
    st.record.participants = [evil]
    st.record.minutes[0].topic = evil
    st.record.minutes[0].points = [evil]
    if st.record.decisions:
        st.record.decisions[0].decision = evil
        st.record.decisions[0].evidence.quote = evil
    for t in st.record.action_items:
        t.task, t.owner, t.owner_stated = evil, evil, True
    st.profile = {"domain": evil, "glossary": [evil]}
    st.log = [evil]
    st.message = evil
    for fn in (v.header_html, v.overview_html, v.actions_html, v.compare_html, v.transcripts_html, v.checks_html, v.tracker_html):
        out = fn(st)
        assert "<script" not in out and "<img" not in out, fn.__name__


def test_unspecified_owner_is_shown_as_unspecified(finished):
    html_out = v.actions_html(finished)
    assert "Unspecified" in html_out or not finished.record.action_items


def test_tracker_shows_failure_clearly():
    from meetscribe.pipeline import RunState
    out = v.tracker_html(RunState(status="error", error="The file is empty.", failed_stage="validate"))
    assert "failed" in out and "The file is empty." in out
