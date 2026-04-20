"""Pydantic I/O types for every agent.

Kept separate from `state.py` tables: agent inputs/outputs are contracts with
the LLM and evals, not persistence shapes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field
from sqlmodel import Session


Action = Literal[
    "scripted", "probe", "clarify", "acknowledge", "off_topic", "wrap_up"
]


# --- Interviewer -----------------------------------------------------------


@dataclass
class InterviewerDeps:
    """Deps for the interviewer agent. Dataclass (not BaseModel) so we can
    carry a live SQLModel Session through RunContext."""

    call_id: str
    session: Session
    turn_number: int


class InterviewerOutput(BaseModel):
    utterance: str = Field(description="Text spoken back to Vapi.")
    action: Action = Field(description="What kind of turn this is.")
    reasoning: str = Field(
        description="Why this action/utterance was chosen. Not spoken; kept for traces + evals."
    )


# --- Analyst ---------------------------------------------------------------


@dataclass
class AnalystDeps:
    call_id: str
    session: Session


class NewProbe(BaseModel):
    question: str
    priority: int = Field(ge=1, le=3)
    rationale: str


class AnalysisUpdate(BaseModel):
    themes: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    surprises: list[str] = Field(default_factory=list)
    new_probes: list[NewProbe] = Field(default_factory=list)


# --- Synthesis -------------------------------------------------------------


class ThemeWithQuotes(BaseModel):
    theme: str
    quotes: list[str] = Field(default_factory=list)


class ReportOutput(BaseModel):
    summary: str
    themes: list[ThemeWithQuotes] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    key_quotes: list[str] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)


# --- Simulated respondent (used by Tier 3 evals) ---------------------------


class Persona(BaseModel):
    name: str
    system: str


class SimulatedReply(BaseModel):
    text: str


# --- Vapi webhook payloads (minimal subset) --------------------------------


class VapiEvent(BaseModel):
    type: Literal["speech-update", "call-ended", "call-started"]
    call_id: str
    text: str | None = None
    end_reason: str | None = None
