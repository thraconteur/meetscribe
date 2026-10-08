"""Data structures shared by every stage of the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

UNSPECIFIED = "Unspecified"


# ---------------------------------------------------------------------------
# Transcript structures (stage 1 / stage 2)
# ---------------------------------------------------------------------------
@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str

    @property
    def tag(self) -> str:
        return f"S{self.id}"


@dataclass
class Transcript:
    segments: list[Segment]
    duration: float = 0.0
    language: str = "en"
    model: str = ""
    notes: list[str] = field(default_factory=list)  # e.g. dropped hallucinated segments

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def segment(self, seg_id: int) -> Segment | None:
        for s in self.segments:
            if s.id == seg_id:
                return s
        return None

    def copy(self) -> "Transcript":
        return Transcript(
            segments=[Segment(s.id, s.start, s.end, s.text) for s in self.segments],
            duration=self.duration,
            language=self.language,
            model=self.model,
            notes=list(self.notes),
        )


@dataclass
class Correction:
    segment_id: int
    before: str
    after: str
    category: str = "term"
    reason: str = ""
    applied: bool = False
    rejected_because: str = ""


# ---------------------------------------------------------------------------
# Meeting record (stage 3) — what the documentation model must return.
# These pydantic models are also the machine-readable output schema.
# ---------------------------------------------------------------------------
class Evidence(BaseModel):
    quote: str = Field(description="Short verbatim excerpt from the transcript that supports the item")
    segment_ids: list[int] = Field(default_factory=list)
    start: float | None = None
    verified: bool = False


class MinutesSection(BaseModel):
    topic: str
    points: list[str] = Field(default_factory=list)


class Decision(BaseModel):
    id: str = ""
    decision: str
    evidence: Evidence


class ActionItem(BaseModel):
    id: str = ""
    task: str
    owner: str = UNSPECIFIED
    owner_stated: bool = False
    deadline: str = UNSPECIFIED
    deadline_stated: bool = False
    evidence: Evidence
    owner_evidence: str | None = None
    deadline_evidence: str | None = None


class OpenItem(BaseModel):
    item: str
    kind: Literal["proposal", "question", "unresolved"] = "unresolved"
    evidence: Evidence | None = None


class MeetingRecord(BaseModel):
    title: str = "Meeting record"
    summary: str = ""
    participants: list[str] = Field(default_factory=list, description="People named in the recording")
    minutes: list[MinutesSection] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    open_items: list[OpenItem] = Field(default_factory=list)
    verification_log: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class PipelineError(Exception):
    """An error with a message that is safe and useful to show to the user."""

    def __init__(self, message: str, stage: str = ""):
        super().__init__(message)
        self.stage = stage
        self.user_message = message
