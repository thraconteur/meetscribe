# MeetScribe — technical description

MeetScribe turns a recorded English meeting into a raw transcript, a domain-corrected transcript and a verified meeting record. It runs three models in a fixed order — speech-to-text → refinement language model → documentation language model — followed by a deterministic verifier. All model calls go through OpenAI-compatible APIs (default provider: Groq), so each stage's model can be swapped through configuration.

## Models and their roles

| Stage | Model | Why this model |
|---|---|---|
| 1. Speech-to-text | **Whisper large-v3** (`whisper-large-v3`, Groq). Optional local back-end: faster-whisper large-v3 | Most accurate Whisper checkpoint for English; returns segment timestamps and per-segment confidence (`avg_logprob`, `no_speech_prob`, `compression_ratio`) that the pipeline uses for filtering. Served far faster than real time. |
| 2. Transcript refinement | **GPT-OSS 20B** (`openai/gpt-oss-20b`, reasoning effort *low*) | Proposing small term corrections is a narrow task, called once per ~1,400-word chunk, so a fast, cheap model is the right fit. Every proposed edit is then checked by code, so a weaker proposer cannot damage meaning. |
| 3. Meeting documentation | **GPT-OSS 120B** (`openai/gpt-oss-120b`, reasoning effort *medium*) | A reasoning model, used for the judgement-heavy step: separating agreed decisions from proposals, explicit assignments from vague suggestions, and stated deadlines from implied ones. 131K context. |

The two language models are different models (20B and 120B parameters), used in separate stages with separate prompts, so the minutes stage never sees the refinement model's reasoning, only its output. Each model also has its own provider rate-limit budget.

## Data flow

```
upload ─► validate & normalise ─► Whisper ─► raw segments [S1..Sn, timestamps]
                                                    │
                     ┌──────────────────────────────┘
                     ▼
        Refinement LM pass A: domain profile (domain, glossary, likely mis-hearings, names)
        Refinement LM pass B: per chunk → JSON list of span edits {segment, original, corrected, reason}
        Edit guards (code) ─► refined segments (same IDs and timestamps) + correction log
                     │
                     ▼
        Documentation LM: numbered refined transcript ─► JSON record with evidence quotes
        (transcripts > 3,000 words: map per part → reduce in meeting order)
                     │
                     ▼
        Evidence verifier (code) ─► MeetingRecord ─► Markdown / HTML / JSON / CSV / ZIP
```

Segment IDs and timestamps created in stage 1 are carried unchanged through every stage, which is what lets each decision and task point back to the moment it was said.

## Stage details

**1. Audio and speech-to-text.** ffprobe validates the upload (type, size, decodability, audio track, duration). ffmpeg converts to 16 kHz mono FLAC with a high-pass filter and dynamic loudness normalisation, which evens out speakers sitting at different distances from the microphone. Silent files are rejected before any API call. Recordings longer than 10 minutes are split at detected pauses, never mid-word. Whisper receives the user's domain hints and the tail of the previous chunk as its `prompt`, biasing the spelling of rare terms. Known Whisper failure modes are filtered using its own confidence values: low-confidence non-speech segments, stock hallucinations on silence ("Thank you for watching"), and repetition loops.

**2. Domain-aware refinement.** Pass A asks the model to infer the meeting's domain, canonical glossary, likely mis-hearings and the people named, from a sample of the whole meeting. This profile is fed to every chunk in pass B, so a term is corrected the same way throughout. Pass B does not rewrite the transcript. It returns minimal span edits, and code applies them only if they pass these guards:

- the original span must exist in that segment;
- the edit may not change any number (digits and number words are compared by value, so "Q three" → "Q3" passes, "four hundred" → "forty" is blocked);
- the edit may not change negation ("not", "can't", …) or commitment/modality words ("will", "might", "should", …);
- it may not remove a known participant's name or replace a long span (≤ 10 words).

Rejected edits are kept in `corrections.csv` with the reason. The raw transcript is never modified.

**3. Documentation.** The model receives the numbered refined transcript and strict definitions. A *decision* is something explicitly agreed; unaccepted suggestions go to *open items*. A *task* is follow-up work that someone committed to or the group agreed on. An *owner* is filled only if the words name the person or team. A *deadline* is filled only if a time limit was stated, copied as spoken. Every item must cite a verbatim quote. For long meetings, notes are extracted per part and then merged in order, so a proposal accepted later becomes a decision and a reversed decision keeps only its final outcome.

**Evidence verifier.** Code, not the model, enforces "reflect what was said". Each quote is fuzzy-matched (rapidfuzz, ≥ 82%) against windows of 1–4 consecutive segments, and the item gets the matching segment IDs and timestamp. If the quote is not found, the decision, task or open item is removed. An owner is kept only if its own quote exists within 3 segments of the task, contains the owner's name, and the owner is not a pronoun. A deadline is kept only if its quote exists within 4 segments of the task and contains the deadline's words and numbers. Any digit a decision or task states that was never said in the recording gets the item removed. A decision whose only evidence is hedged language ("maybe", "we could", "let's think about it") with no sign of agreement is moved to open items as a proposal. Anything else becomes **Unspecified**. Participants who are never named in the transcript are dropped. Every removal or downgrade is written to the verification log shown in the UI and in the record.

## Outputs and interface

A Gradio web app runs the pipeline in a background thread and streams state to a stage tracker, so raw and refined transcripts appear before the record is finished. Tabs show the minutes, decisions with proposals that were not agreed, action items, raw vs refined transcripts, a word-level diff of corrections, and the verification log. Markdown/HTML (human-readable) and JSON/CSV (machine-readable) are all rendered from the same verified record object, so they always carry identical decisions and tasks. A command-line runner and an offline test-suite are included.

## Robustness

- Rate limits are retried with back-off that honours `Retry-After`.
- If a request exceeds a provider's per-request token cap, the output budget is shrunk to fit, or the chunk is split.
- Malformed JSON is repaired or re-requested.
- If a reasoning model runs out of tokens, it is retried with lower reasoning effort.
- Errors are converted into user-facing messages that name the failing stage.

## Speed and rate limits

On a free Groq plan the cost of a long recording is dominated by the per-minute token limit, not by computation. A measured
71-minute lecture took 591 s end to end; the models themselves worked for about 160 s and the rest was automatic waiting on
the limit (the Checks tab reports this split for every run). Defaults keep the token count low: low reasoning effort for
refinement and for per-part note extraction, and one model per stage so each has its own budget. On a paid plan set
`LLM_CONCURRENCY=4` to run chunk requests in parallel; results are identical to sequential processing (tested).

## Rubric coverage

| Rubric row (points) | What implements it | Guarded by tests |
|---|---|---|
| Speech transcription (20) | Whisper large-v3, temperature 0, language pinned, domain hints as a spelling prompt, splits at pauses, silence hallucinations filtered, raw transcript kept immutable | `test_pipeline` (`test_long_recording_is_chunked_at_pauses`, `test_rejects_silence`, `test_no_speech_recognised`) |
| Transcript refinement (20) | Domain profile first, then small word-level edits only. Code rejects any edit that changes a number, a negation, a commitment word or a name; rejected edits are shown with the reason | `test_pipeline` (`test_number_and_negation_helpers`, `test_end_to_end`), `test_speed` |
| Minutes and decisions (25) | Every decision cites a verbatim quote that code locates in the transcript; hedged quotes become *open items*, not decisions; digits in decisions, tasks, minutes and summary must have been said; internal segment references are stripped | `test_grounding_adversarial` |
| Action items (15) | Owner kept only if a quote naming that person sits within 3 segments; deadline only if its words appear within 4 segments; otherwise shown as *Unspecified*; agenda previews are not tasks | `test_grounding_adversarial` |
| End-to-end application (15) | Two different models in separate stages, one-click UI with live stage status, ZIP/HTML/MD/JSON/CSV exports, clear errors for unsupported, empty, corrupt, silent and non-speech files, reopen past runs | `test_pipeline` (`test_rejects_bad_files`, `test_validate_reports_video_without_audio`, `test_missing_api_key`), `test_llm_errors`, `test_ui_replay` |
| Submission quality (5) | README with setup, this document, prompts in `meetscribe/prompts/`, sample recording script, evaluation script | n/a |

## Limitations

Speakers are not diarised, so owners are assigned only when a name is spoken. This is a deliberate trade-off in favour of never guessing. Accuracy depends on audio quality, and heavy cross-talk degrades speech-to-text. Deadlines are reported as spoken rather than as calendar dates.
