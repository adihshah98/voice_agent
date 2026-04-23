"""Pydantic I/O types for every agent.

Kept separate from `state.py` tables: agent inputs/outputs are contracts with
the LLM and evals, not persistence shapes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, Field
from sqlmodel import Session


Action = Literal[
    "scripted", "probe", "clarify", "off_topic", "wrap_up", "skip_scripted"
]


# --- Interviewer -----------------------------------------------------------


@dataclass
class InterviewerDeps:
    """Deps for the interviewer agent. Dataclass (not BaseModel) so we can
    carry call metadata through RunContext.

    session is optional because turn.py preloads DB context in a short-lived
    session before awaiting the LLM call.
    """

    call_id: str
    session: Session | None
    turn_number: int


class InterviewerOutput(BaseModel):
    utterance: str = Field(description="Text spoken back to Vapi.")
    action: Action = Field(description="What kind of turn this is.")
    reasoning: str = Field(
        description="Why this action/utterance was chosen. Not spoken; kept for traces + evals."
    )
    probe_id_used: int | None = Field(
        default=None,
        description="The id of the PENDING_PROBE used, if action=probe. Must match exactly.",
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
    investor_signals: list[str] = Field(
        default_factory=list,
        description=(
            "Tagged investor-relevant signals. Each entry is prefixed with a category tag: "
            "[PMF], [COMPETITIVE], [REVENUE], [AI-SIGNAL], or [RED-FLAG]. "
            'Example: "[PMF] Uses it daily before every meeting — strong habit formation"'
        ),
    )
    new_probes: list[NewProbe] = Field(default_factory=list)
    covered_subtopics: list[str] = Field(
        default_factory=list,
        description=(
            "Short noun-phrase labels (3-6 words) for every specific subtopic addressed "
            "in this conversation — explicit or organic. Include all prior subtopics from "
            "ESTABLISHED CONTEXT plus any new ones from the NEW TRANSCRIPT. "
            "Name specific entities, not categories: "
            "'Notion vs Google Docs product features' not 'competitor product comparison'; "
            "'Notion vs Google Docs pricing' is a separate entry from product features. "
            "Other examples: 'IT security SOC2 concerns', 'VP of Sales budget ownership', "
            "'day-to-day AE usage workflow'. Never collapse distinct things into one label."
        ),
    )


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
    pmf_score: int = Field(default=0, ge=1, le=5, description="1=no PMF evidence, 5=strong PMF")
    pmf_score_rationale: str = Field(default="")
    competitive_signals: list[str] = Field(default_factory=list)
    revenue_signals: list[str] = Field(default_factory=list)
    ai_adoption_signals: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    investment_thesis_bullets: list[str] = Field(default_factory=list)


# --- Simulated respondent (used by Tier 3 evals) ---------------------------


class Persona(BaseModel):
    name: str
    system: str


class SimulatedReply(BaseModel):
    text: str


# --- Vapi webhook payloads -------------------------------------------------


class VapiCallRef(BaseModel):
    id: str  # Vapi's call ID — maps to Call.vapi_call_id in our DB


class VapiArtifactMessage(BaseModel):
    role: str  # "bot" | "user"
    message: str
    time: float | None = None


class VapiArtifact(BaseModel):
    messages: list[VapiArtifactMessage] = Field(default_factory=list)
    transcript: str | None = None


class VapiStatusUpdate(BaseModel):
    type: Literal["status-update"]
    status: str  # "in-progress" | "ended" | "ringing" | etc.
    call: VapiCallRef


class VapiAssistantRequest(BaseModel):
    type: Literal["assistant-request"]
    call: VapiCallRef
    artifact: VapiArtifact | None = None


class VapiEndOfCallReport(BaseModel):
    type: Literal["end-of-call-report"]
    endedReason: str | None = None
    call: VapiCallRef


VapiMessage = Annotated[
    VapiStatusUpdate | VapiAssistantRequest | VapiEndOfCallReport,
    Field(discriminator="type"),
]


class VapiWebhookPayload(BaseModel):
    """Top-level payload sent by Vapi to the serverUrl webhook."""
    message: VapiMessage | None = None
    model_config = {"extra": "allow"}
