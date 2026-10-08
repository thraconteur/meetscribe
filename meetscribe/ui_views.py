"""HTML views for the web UI. Pure functions of a RunState, so they are easy to test.

Everything that comes from the recording or a model is untrusted text and is escaped here.
"""
from __future__ import annotations

import difflib
import html
import time

from .pipeline import STAGE_LABELS, STAGES, RunState
from .refine import _ts
from .schema import UNSPECIFIED

# Which usage bucket belongs to which pipeline stage.
_STAGE_ROLE = {"transcribe": "stt", "refine": "refine", "document": "document"}


def e(value) -> str:
    return html.escape(str(value), quote=True)


def _evidence(ev) -> str:
    if ev is None or not getattr(ev, "quote", ""):
        return ""
    when = f'<span class="ts">{_ts(ev.start)}</span>' if ev.start is not None else ""
    return f'<blockquote class="ev">{when}<q>{e(ev.quote)}</q></blockquote>'


def _chip(text: str, kind: str = "") -> str:
    return f'<span class="chip {kind}">{e(text)}</span>'


def _who(value: str, stated: bool, what: str) -> str:
    if stated and value and value != UNSPECIFIED:
        return f'<span class="val">{e(value)}</span>'
    return f'<span class="val none" title="The recording does not state {e(what)}. Nothing is guessed.">{UNSPECIFIED}</span>'


# --------------------------------------------------------------------------- progress
def tracker_html(st: RunState | None, started: float | None = None) -> str:
    if st is None:
        return '<div class="tracker idle">Choose a recording, then press <b>Process meeting</b>.</div>'
    current = STAGES.index(st.stage) if st.stage in STAGES else 0
    steps = []
    for i, stage in enumerate(STAGES):
        if st.status == "error" and stage == st.failed_stage:
            cls, icon = "error", "!"
        elif st.status == "done" or i < current or stage in st.stage_times:
            cls, icon = "done", "✓"
        elif i == current and st.status == "running":
            cls, icon = "running", '<i class="spin"></i>'
        else:
            cls, icon = "pending", str(i + 1)
        t = st.stage_times.get(stage)
        steps.append(f'<li class="{cls}"><span class="dot">{icon}</span><span class="lbl">{e(STAGE_LABELS[stage])}</span>'
                     f'<span class="t">{f"{t:.0f}s" if t is not None else ""}</span></li>')
    if st.status == "error":
        note = f'<div class="note err"><b>{e(STAGE_LABELS.get(st.failed_stage, "Processing"))} failed.</b> {e(st.error)}</div>'
    else:
        elapsed = f" · {time.time() - started:.0f}s elapsed" if started and st.status == "running" else ""
        live = ""
        if st.status == "running" and st.log and "retrying" in st.log[-1]:
            live = f'<div class="note wait">{e(st.log[-1])}. The provider is rate-limiting us; this is automatic.</div>'
        note = f'<div class="note">{e(st.message)}{elapsed}</div>{live}'
    return f'<div class="tracker"><ol>{"".join(steps)}</ol>{note}</div>'


# --------------------------------------------------------------------------- hero / header
def hero_html(stt_name: str, refine_name: str, document_name: str) -> str:
    chain = "".join(f'<li><small>{e(k)}</small><b>{e(v)}</b></li>' for k, v in
                    [("Speech to text", stt_name), ("Refinement model", refine_name),
                     ("Documentation model", document_name), ("Verifier", "quote checks in code")])
    return (f'<header id="hero"><div><h1>MeetScribe</h1><p>Recording in. Transcript, minutes, decisions and action items out, '
            f'each backed by a quote from the recording.</p></div><ol class="chain">{chain}</ol></header>')


def empty_html(max_mb: float, max_min: float) -> str:
    return (f'<div class="empty"><h2>What you get from one recording</h2><ul>'
            f'<li><b>Raw and refined transcripts</b>, side by side, with every correction listed.</li>'
            f'<li><b>Minutes, decisions and action items</b>. Each one carries the quote and time it came from.</li>'
            f'<li><b>Owners and deadlines only when said.</b> Anything else shows as {UNSPECIFIED}.</li>'
            f'<li><b>Downloads</b> as ZIP, HTML, JSON and CSV.</li></ul>'
            f'<p class="fine">English audio or video up to {max_mb:.0f} MB and {max_min:.0f} minutes. '
            f'Processing a one-hour recording on a free API plan can take several minutes because of rate limits.</p></div>')


def header_html(st: RunState) -> str:
    rec = st.record
    title = rec.title if rec else (st.audio_info.path.name if st.audio_info else "Processing…")
    bits = []
    if st.audio_info and st.audio_info.duration:
        bits.append(f"{st.audio_info.duration / 60:.1f} min of audio")
    if st.raw:
        bits.append(f"{st.raw.word_count:,} words")
    if st.corrections:
        applied = sum(c.applied for c in st.corrections)
        bits.append(f"{applied} term correction{'s' if applied != 1 else ''}")
    counts = ""
    if rec:
        counts = (f'<p class="counts"><b>{len(rec.decisions)}</b> decision{"s" if len(rec.decisions) != 1 else ""} · '
                  f'<b>{len(rec.action_items)}</b> action item{"s" if len(rec.action_items) != 1 else ""} · '
                  f'<b>{len(rec.open_items)}</b> open item{"s" if len(rec.open_items) != 1 else ""}</p>')
    domain = f' <span class="chip">{e(st.profile["domain"])}</span>' if st.profile.get("domain") else ""
    return f'<div class="rhead"><h2>{e(title)}</h2><p class="meta">{e(" · ".join(bits))}{domain}</p>{counts}</div>'


# --------------------------------------------------------------------------- tabs
def overview_html(st: RunState) -> str:
    rec = st.record
    if rec is None:
        return '<div class="pending">Minutes are being written. The transcripts are already available in the Transcripts tab.</div>'
    out = [f'<div class="read"><p class="lede">{e(rec.summary) if rec.summary else "No summary was produced."}</p>']
    if rec.participants:
        out.append('<p class="people"><small>People named in the recording</small> '
                   + " ".join(_chip(p) for p in rec.participants) + "</p>")
    for sec in rec.minutes:
        out.append(f'<section class="topic"><h3>{e(sec.topic)}</h3><ul>'
                   + "".join(f"<li>{e(p)}</li>" for p in sec.points) + "</ul></section>")
    out.append("</div>")
    return "".join(out)


def actions_html(st: RunState) -> str:
    rec = st.record
    if rec is None:
        return '<div class="pending">Decisions and action items appear when the minutes are done.</div>'
    out = []
    out.append(f'<section class="blk"><h3>Decisions <span class="n">{len(rec.decisions)}</span></h3>')
    if rec.decisions:
        for d in rec.decisions:
            out.append(f'<article class="card"><p class="main">{e(d.decision)}</p>{_evidence(d.evidence)}</article>')
    else:
        out.append('<p class="none-note">Nothing in this recording was agreed as a decision. '
                   'Proposals and open questions are listed below, kept separate on purpose.</p>')
    out.append("</section>")

    out.append(f'<section class="blk"><h3>Action items <span class="n">{len(rec.action_items)}</span></h3>')
    if rec.action_items:
        out.append('<div class="tasks">')
        for t in rec.action_items:
            out.append(
                f'<article class="card task"><p class="main">{e(t.task)}</p>'
                f'<dl><div><dt>Owner</dt><dd>{_who(t.owner, t.owner_stated, "an owner")}</dd></div>'
                f'<div><dt>Deadline</dt><dd>{_who(t.deadline, t.deadline_stated, "a deadline")}</dd></div></dl>'
                f'{_evidence(t.evidence)}</article>')
        out.append("</div>")
    else:
        out.append('<p class="none-note">No tasks were assigned or committed to in this recording.</p>')
    out.append("</section>")

    if rec.open_items:
        out.append(f'<section class="blk"><details><summary>Proposals and open questions, not agreed '
                   f'<span class="n">{len(rec.open_items)}</span></summary><div class="open">')
        for o in rec.open_items:
            out.append(f'<article class="card"><p class="main">{_chip(o.kind, "kind")} {e(o.item)}</p>{_evidence(o.evidence)}</article>')
        out.append("</div></details></section>")
    return "".join(out)


def _word_diff(a: str, b: str) -> tuple[str, str]:
    aw, bw = a.split(), b.split()
    left, right = [], []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=aw, b=bw, autojunk=False).get_opcodes():
        if op == "equal":
            left.append(e(" ".join(aw[i1:i2])))
            right.append(e(" ".join(bw[j1:j2])))
        else:
            if i2 > i1:
                left.append(f"<del>{e(' '.join(aw[i1:i2]))}</del>")
            if j2 > j1:
                right.append(f"<ins>{e(' '.join(bw[j1:j2]))}</ins>")
    return " ".join(x for x in left if x), " ".join(x for x in right if x)


def compare_html(st: RunState) -> str:
    if not st.raw:
        return '<div class="pending">Corrections appear after the speech-to-text step.</div>'
    out = []
    p = st.profile or {}
    if p.get("domain") or p.get("glossary"):
        out.append('<section class="blk"><h3>What the refinement model learned about this recording</h3>')
        if p.get("domain"):
            out.append(f'<p>Domain: <b>{e(p["domain"])}</b></p>')
        if p.get("glossary"):
            out.append('<details><summary>Glossary used to judge terms ('
                       f'{len(p["glossary"])})</summary><p class="chips">'
                       + " ".join(_chip(g) for g in p["glossary"]) + "</p></details>")
        out.append("</section>")
    applied = [c for c in st.corrections if c.applied]
    rejected = [c for c in st.corrections if not c.applied]
    out.append(f'<section class="blk"><h3>Corrections <span class="n">{len(applied)} applied</span>'
               + (f' <span class="n warn">{len(rejected)} blocked by safety checks</span>' if rejected else "")
               + "</h3>")
    if not st.corrections:
        out.append('<p class="none-note">No corrections were needed. The speech-to-text output already matched the '
                   "vocabulary of this recording.</p>")
    else:
        out.append('<table class="corr"><thead><tr><th>Time</th><th>Heard</th><th>Corrected to</th><th>Why</th></tr></thead><tbody>')
        for c in applied + rejected:
            seg = st.raw.segment(c.segment_id)
            when = _ts(seg.start) if seg else ""
            if c.applied:
                after = f"<ins>{e(c.after)}</ins>"
                why = e(c.reason)
            else:
                after = f'<span class="kept">kept original</span>'
                why = f'<span class="blocked">Blocked: {e(c.rejected_because)}</span>'
            out.append(f'<tr class="{"" if c.applied else "rej"}"><td class="ts">{when}</td><td><del>{e(c.before)}</del></td>'
                       f'<td>{after}</td><td>{why}</td></tr>')
        out.append("</tbody></table>")
    out.append("</section>")
    return "".join(out)


def transcripts_html(st: RunState) -> str:
    if not st.raw:
        return '<div class="pending">The transcript appears after speech-to-text finishes.</div>'
    refined = {s.id: s.text for s in st.refined.segments} if st.refined else {}
    left, right = [], []
    for s in st.raw.segments:
        new = refined.get(s.id, s.text)
        if new != s.text:
            a, b = _word_diff(s.text, new)
            left.append(f'<p class="seg chg"><span class="ts">{_ts(s.start)}</span> {a}</p>')
            right.append(f'<p class="seg chg"><span class="ts">{_ts(s.start)}</span> {b}</p>')
        else:
            line = f'<p class="seg"><span class="ts">{_ts(s.start)}</span> {e(s.text)}</p>'
            left.append(line)
            right.append(line)
    return ('<div class="tx"><section><h3>Raw <small>speech to text</small></h3><div class="scroll">'
            + "".join(left) + '</div></section><section><h3>Refined <small>domain terms corrected</small></h3>'
            '<div class="scroll">' + ("".join(right) if st.refined else '<p class="pending">Refining…</p>')
            + "</div></section></div>")


def checks_html(st: RunState) -> str:
    out = []
    if st.record:
        out.append('<section class="blk"><h3>Verification</h3><ul class="plain">'
                   + "".join(f"<li>{e(x)}</li>" for x in st.record.verification_log) + "</ul></section>")
    usage = (st.record.metadata.get("usage") if st.record else None) or {}
    models = (st.record.metadata.get("models") if st.record else None) or {}
    if st.stage_times:
        rows, wait_total = [], 0.0
        for stage in STAGES:
            total = st.stage_times.get(stage)
            if total is None:
                continue
            role = _STAGE_ROLE.get(stage)
            api = (usage.get(role) or {}).get("seconds") if role else None
            wait = max(0.0, total - api) if api is not None else None
            if stage in ("refine", "document") and wait:
                wait_total += wait
            tokens = ""
            if role and usage.get(role):
                u = usage[role]
                tokens = f'{u.get("prompt_tokens", 0):,} in / {u.get("completion_tokens", 0):,} out' if u.get("prompt_tokens") \
                    else f'{u.get("calls", 0)} requests'
            rows.append(f'<tr><td>{e(STAGE_LABELS[stage])}</td><td>{total:.0f}s</td>'
                        f'<td>{"" if api is None else f"{api:.0f}s"}</td>'
                        f'<td>{"" if wait is None else f"{wait:.0f}s"}</td><td>{tokens}</td>'
                        f'<td>{e(models.get(role, "")) if role else ""}</td></tr>')
        note = ""
        if wait_total > 30:
            note = (f'<p class="note wait">About {wait_total:.0f}s was spent waiting on the provider’s per-minute token limit, not '
                    'computing. That is typical on a free API plan. A paid plan removes most of it.</p>')
        out.append('<section class="blk"><h3>Where the time went</h3><table class="corr"><thead><tr><th>Step</th><th>Total</th>'
                   '<th>Model working</th><th>Waiting</th><th>Tokens</th><th>Model</th></tr></thead><tbody>'
                   + "".join(rows) + f"</tbody></table>{note}</section>")
    if st.log:
        out.append('<section class="blk"><details><summary>Run log</summary><pre>' + e("\n".join(st.log)) + "</pre></details></section>")
    return "".join(out) or '<div class="pending">Checks appear when processing finishes.</div>'


# --------------------------------------------------------------------------- stylesheet
CSS = """
:root{--ms-bg:#f3f6f9;--ms-surface:#fff;--ms-ink:#14202b;--ms-muted:#586675;--ms-line:#dbe2e9;--ms-accent:#0b6b8a;
--ms-accent-soft:#e3f2f7;--ms-mark:#fdf0b8;--ms-mark-line:#e6c85a;--ms-good:#1d7a57;--ms-bad:#b3412b;--ms-bad-soft:#fbe9e5;
--ms-good-soft:#e2f3ea;--ms-serif:'Newsreader',Georgia,'Iowan Old Style',serif}
.dark{--ms-bg:#0e151c;--ms-surface:#151e27;--ms-ink:#e5ecf2;--ms-muted:#98a6b4;--ms-line:#26323f;--ms-accent:#62c1de;
--ms-accent-soft:#12303b;--ms-mark:#3a3210;--ms-mark-line:#8a742a;--ms-good:#5fcf9d;--ms-bad:#ff9b84;--ms-bad-soft:#3a1d17;--ms-good-soft:#12301f}
.gradio-container{max-width:1680px !important;background:var(--ms-bg) !important}
#hero{display:flex;flex-wrap:wrap;gap:12px 40px;align-items:flex-end;justify-content:space-between;padding:6px 2px 14px}
#hero h1{font-size:1.7rem;margin:0;letter-spacing:-.015em}#hero p{margin:4px 0 0;color:var(--ms-muted);max-width:60ch}
.chain{display:flex;flex-wrap:wrap;gap:8px;list-style:none;margin:0;padding:0}
.chain li{border:1px solid var(--ms-line);background:var(--ms-surface);border-radius:8px;padding:6px 12px;display:flex;flex-direction:column;line-height:1.25}
.chain small{color:var(--ms-muted);font-size:.72rem}.chain b{font-size:.86rem;font-weight:600}
#left{position:sticky;top:12px;align-self:flex-start}
.tracker ol{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:2px}
.tracker li{display:flex;align-items:center;gap:10px;padding:7px 4px;font-size:.92rem}
.tracker .dot{width:22px;height:22px;border-radius:50%;display:inline-grid;place-items:center;font-size:.75rem;font-weight:700;
border:1.5px solid var(--ms-line);color:var(--ms-muted);flex:none}
.tracker .t{margin-left:auto;font-variant-numeric:tabular-nums;color:var(--ms-muted);font-size:.8rem}
.tracker li.done .dot{background:var(--ms-good);border-color:var(--ms-good);color:#fff}
.tracker li.running .dot{border-color:var(--ms-accent)}.tracker li.running .lbl{font-weight:600}
.tracker li.error .dot{background:var(--ms-bad);border-color:var(--ms-bad);color:#fff}.tracker li.pending{opacity:.6}
.spin{width:11px;height:11px;border:2px solid var(--ms-accent);border-right-color:transparent;border-radius:50%;animation:sp .9s linear infinite}
@keyframes sp{to{transform:rotate(360deg)}}@media (prefers-reduced-motion:reduce){.spin{animation:none}}
.tracker.idle{color:var(--ms-muted);font-size:.92rem;padding:4px 2px}
.note{margin-top:8px;font-size:.88rem;color:var(--ms-muted)}.note.err{color:var(--ms-bad);background:var(--ms-bad-soft);padding:8px 10px;border-radius:8px}
.note.wait{background:var(--ms-accent-soft);color:var(--ms-ink);padding:8px 10px;border-radius:8px}
.empty{padding:28px 8px;max-width:64ch}.empty h2{font-size:1.25rem;margin:0 0 10px}.empty ul{padding-left:1.1rem;line-height:1.7;margin:0}
.empty .fine{color:var(--ms-muted);font-size:.88rem;margin-top:14px}
.rhead h2{font-size:1.55rem;line-height:1.25;margin:0 0 4px;letter-spacing:-.01em}.rhead .meta{margin:0;color:var(--ms-muted);font-size:.92rem}
.rhead .counts{margin:8px 0 0;font-size:.95rem}.rhead .counts b{font-weight:700}
.chip{display:inline-block;border:1px solid var(--ms-line);border-radius:999px;padding:1px 9px;font-size:.78rem;background:var(--ms-surface);color:var(--ms-ink)}
.chip.kind{background:var(--ms-accent-soft);border-color:transparent;color:var(--ms-accent)}
.prose.gradio-style{max-width:100% !important}
#dl-row{gap:8px;margin:10px 0 6px;flex-wrap:wrap;justify-content:flex-start}
#dl-row>*{flex:0 0 auto !important;min-width:0 !important;width:auto !important}
#dl-row button,#dl-row a{padding:6px 14px !important;font-size:.84rem !important}
.read{max-width:76ch}.lede{font-family:var(--ms-serif);font-size:1.22rem;line-height:1.62;margin:6px 0 14px}
.people{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:0 0 18px}.people small{color:var(--ms-muted);margin-right:4px}
.topic{margin:18px 0}.topic h3{font-size:1.02rem;margin:0 0 6px}.topic ul{margin:0;padding-left:1.15rem;line-height:1.65;font-family:var(--ms-serif);font-size:1.05rem}
.blk{margin:6px 0 26px}.blk h3{font-size:1.02rem;margin:0 0 10px;display:flex;gap:8px;align-items:baseline}
.n{font-size:.78rem;font-weight:600;color:var(--ms-muted);border:1px solid var(--ms-line);border-radius:999px;padding:0 8px}.n.warn{color:var(--ms-bad);border-color:var(--ms-bad)}
.none-note{color:var(--ms-muted);max-width:70ch;margin:0}
.card{border:1px solid var(--ms-line);background:var(--ms-surface);border-radius:10px;padding:12px 14px;margin:0 0 10px}
.card .main{margin:0 0 8px;font-size:1rem;line-height:1.5}
.ev{margin:0;padding:6px 10px;border-left:3px solid var(--ms-mark-line);background:var(--ms-mark);border-radius:0 6px 6px 0;font-size:.9rem;line-height:1.55}
.ev q{quotes:none;font-family:var(--ms-serif);font-size:.98rem}.ts{font-family:var(--font-mono,monospace);font-size:.75rem;color:var(--ms-muted);margin-right:8px;white-space:nowrap}
.task dl{display:flex;gap:28px;margin:0 0 10px}.task dt{font-size:.74rem;color:var(--ms-muted)}.task dd{margin:0;font-weight:600}
.val.none{font-weight:400;color:var(--ms-muted);border-bottom:1px dotted var(--ms-muted);cursor:help}
details>summary{cursor:pointer;font-weight:600;font-size:1.02rem;margin-bottom:10px}
.corr,.corr th,.corr td{border-left:0 !important;border-right:0 !important;border-top:0 !important}
.corr{width:100%;border-collapse:collapse;font-size:.9rem}.corr th{text-align:left;font-weight:600;color:var(--ms-muted);font-size:.78rem;padding:4px 10px 6px 0;border-bottom:1px solid var(--ms-line)}
.corr td{padding:7px 10px 7px 0;border-bottom:1px solid var(--ms-line);vertical-align:top}
del{background:var(--ms-bad-soft);color:var(--ms-bad);text-decoration:line-through;padding:0 3px;border-radius:3px}
ins{background:var(--ms-good-soft);color:var(--ms-good);text-decoration:none;font-weight:600;padding:0 3px;border-radius:3px}
.kept,.blocked{color:var(--ms-bad)}.corr tr.rej td{opacity:.85}
.tx{display:grid;grid-template-columns:1fr 1fr;gap:18px}@media (max-width:1100px){.tx{grid-template-columns:1fr}}
.tx h3{font-size:1rem;margin:0 0 8px}.tx small{color:var(--ms-muted);font-weight:400;margin-left:6px}
.scroll{max-height:72vh;overflow:auto;border:1px solid var(--ms-line);background:var(--ms-surface);border-radius:10px;padding:8px 14px}
.seg{margin:0;padding:5px 0;line-height:1.6;font-size:.95rem;border-bottom:1px solid transparent}.seg.chg{background:var(--ms-accent-soft);margin:0 -14px;padding:5px 14px}
.pending{color:var(--ms-muted);padding:18px 2px}.plain{margin:0;padding-left:1.1rem;line-height:1.7}
pre{white-space:pre-wrap;font-size:.8rem;max-height:340px;overflow:auto;background:var(--ms-surface);border:1px solid var(--ms-line);border-radius:8px;padding:10px}
:focus-visible{outline:2px solid var(--ms-accent);outline-offset:2px}
"""

HEAD = ('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        '<link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;1,6..72,400&display=swap" rel="stylesheet">')
