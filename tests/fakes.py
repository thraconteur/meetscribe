"""Offline stand-ins for the OpenAI-compatible APIs. Used ONLY by the test-suite."""
from __future__ import annotations

import json
from types import SimpleNamespace

RAW_SEGMENTS = [
    (0.0, 6.0, "Okay everyone, let's start the sprint planning for the payments service."),
    (6.0, 13.0, "First item, the cube control rollout. We moved the cluster to the new Cubernetes version last week."),
    (13.0, 20.0, "Priya, can you update the helm charts for the staging cluster by Friday?"),
    (20.0, 25.0, "Sure, I'll update the helm charts by Friday."),
    (25.0, 33.0, "Second, the database. Rahul proposed moving from my sequel to post gress."),
    (33.0, 40.0, "After the discussion, we agreed to migrate to post gress in Q three."),
    (40.0, 47.0, "We should not touch the billing tables until the audit is done."),
    (47.0, 54.0, "Someone needs to write the migration runbook, we didn't decide who."),
    (54.0, 60.0, "Maybe we could try graph QL for the public API, but let's think about it."),
    (60.0, 66.0, "Rahul will benchmark the new A.P.I. gateway, it costs about four hundred dollars a month."),
]


class FakeChat:
    def __init__(self, record_override: dict | None = None):
        self.calls = []
        self.record_override = record_override

    def create(self, **kwargs):
        self.calls.append(kwargs)
        system = kwargs["messages"][0]["content"]
        if "terminology analyst" in system:
            content = {"domain": "DevOps / backend", "glossary": ["kubectl", "Kubernetes", "Helm", "MySQL",
                                                                  "Postgres", "GraphQL", "API"],
                       "suspected_errors": [{"heard": "cube control", "likely": "kubectl"}],
                       "names": ["Priya", "Rahul"]}
        elif "proof-reader" in system:
            content = {"corrections": [
                {"segment": 2, "original": "cube control", "corrected": "kubectl", "category": "term", "reason": "tool"},
                {"segment": 2, "original": "Cubernetes", "corrected": "Kubernetes", "category": "term", "reason": "k8s"},
                {"segment": 5, "original": "my sequel", "corrected": "MySQL", "category": "term", "reason": "db"},
                {"segment": 5, "original": "post gress", "corrected": "Postgres", "category": "term", "reason": "db"},
                {"segment": 6, "original": "post gress", "corrected": "Postgres", "category": "term", "reason": "db"},
                {"segment": 6, "original": "Q three", "corrected": "Q3", "category": "term", "reason": "quarter"},
                {"segment": 9, "original": "graph QL", "corrected": "GraphQL", "category": "term", "reason": "api"},
                {"segment": 10, "original": "A.P.I.", "corrected": "API", "category": "acronym", "reason": "acronym"},
                # Unsafe edits the guards must reject:
                {"segment": 7, "original": "should not touch", "corrected": "should touch", "category": "term",
                 "reason": "bad"},
                {"segment": 10, "original": "four hundred dollars", "corrected": "forty dollars", "category": "term",
                 "reason": "bad"},
                {"segment": 4, "original": "I'll update", "corrected": "I might update", "category": "term",
                 "reason": "bad"},
                {"segment": 99, "original": "x", "corrected": "y", "category": "term", "reason": "bad id"},
            ]}
        elif "processing ONE PART" in system:
            content = _record(map_part=True)
        elif "were processed in consecutive parts" in system or "minute-taker" in system:
            content = self.record_override or _record()
        else:
            content = {}
        msg = SimpleNamespace(content=json.dumps(content))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                               usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50))


def _record(map_part: bool = False) -> dict:
    rec = {
        "title": "Payments service sprint planning",
        "summary": "The team reviewed the kubectl rollout and agreed to migrate to Postgres in Q3.",
        "participants": ["Priya", "Rahul", "Ghost Person"],
        "minutes": [{"topic": "Kubernetes rollout", "points": ["Cluster moved to new Kubernetes version"]}],
        "decisions": [
            {"decision": "Migrate from MySQL to Postgres in Q3",
             "evidence": "we agreed to migrate to Postgres in Q3", "segments": [6]},
            {"decision": "Adopt GraphQL for the public API",  # hallucinated quote -> must be removed
             "evidence": "we decided to adopt GraphQL for everything", "segments": [9]},
        ],
        "action_items": [
            {"task": "Update the Helm charts for the staging cluster", "owner": "Priya",
             "owner_evidence": "Priya, can you update the helm charts for the staging cluster",
             "deadline": "by Friday", "deadline_evidence": "update the helm charts by Friday",
             "evidence": "can you update the helm charts for the staging cluster by Friday", "segments": [3]},
            {"task": "Write the migration runbook", "owner": "Rahul",  # invented owner -> Unspecified
             "owner_evidence": "Rahul will write the runbook", "deadline": "next Monday",  # invented deadline
             "deadline_evidence": "by next Monday", "evidence": "Someone needs to write the migration runbook",
             "segments": [8]},
            {"task": "Benchmark the new API gateway", "owner": "Rahul",
             "owner_evidence": "Rahul will benchmark the new API gateway", "deadline": None,
             "deadline_evidence": None, "evidence": "Rahul will benchmark the new API gateway", "segments": [10]},
            {"task": "Freeze billing tables", "owner": "I", "evidence": "We should not touch the billing tables "
             "until the audit is done", "segments": [7]},
        ],
        "open_items": [
            {"item": "Consider GraphQL for the public API", "kind": "proposal",
             "evidence": "Maybe we could try GraphQL for the public API", "segments": [9]},
        ],
    }
    if map_part:
        rec["topics"] = rec.pop("minutes")
    return rec


class FakeTranscriptions:
    def __init__(self, segments=None):
        self.segments = RAW_SEGMENTS if segments is None else segments
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        segs = [{"start": a, "end": b, "text": " " + t, "avg_logprob": -0.2, "no_speech_prob": 0.01,
                 "compression_ratio": 1.3} for a, b, t in self.segments]
        return SimpleNamespace(text=" ".join(t for _, _, t in self.segments), segments=segs, duration=66.0)


class FakeClient:
    def __init__(self, chat: FakeChat | None = None, transcriptions: FakeTranscriptions | None = None):
        self.chat = SimpleNamespace(completions=chat or FakeChat())
        self.audio = SimpleNamespace(transcriptions=transcriptions or FakeTranscriptions())
