"""Case input/expected-output types for interviewer evals + a YAML loader.

We keep these separate from `models.py` because an eval case is a *seeded DB
state plus a respondent utterance*, not a live `InterviewerDeps`. The loader
builds `pydantic_evals.Case` objects from a flat YAML file so we can add cases without touching Python.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_evals import Case

from voice_agent.models import AnalysisUpdate, InterviewerOutput


class TurnLine(BaseModel):
    speaker: Literal["interviewer", "respondent"]
    text: str
    action: Optional[str] = None


class ProbeSeed(BaseModel):
    question: str
    priority: int = Field(ge=1, le=3)
    rationale: str = ""


class InterviewerCaseInputs(BaseModel):
    """Everything needed to reconstruct a DB state for one turn.

    `last_respondent` is the utterance that triggered this webhook — it is
    what gets passed to `interviewer.run(...)`. It is *not* appended to
    `prior_turns`; the task function appends it when seeding the DB so that
    the prompt's `recent_turns` tool sees the same thing a live call would.
    """

    scripted_questions: list[str] = Field(default_factory=list)
    prior_turns: list[TurnLine] = Field(default_factory=list)
    probes: list[ProbeSeed] = Field(default_factory=list)
    last_respondent: str


def load_cases(
    path: str | Path,
) -> list[Case[InterviewerCaseInputs, InterviewerOutput, None]]:
    """Load YAML cases. Expected output is coerced into `InterviewerOutput`
    with empty `utterance`/`reasoning` so that the task/expected types match.
    Evaluators only read `.action` off the expected object."""
    raw = yaml.safe_load(Path(path).read_text())
    cases: list[Case[InterviewerCaseInputs, InterviewerOutput, None]] = []
    for entry in raw["cases"]:
        exp = entry["expected_output"]
        cases.append(
            Case(
                name=entry["name"],
                inputs=InterviewerCaseInputs.model_validate(entry["inputs"]),
                expected_output=InterviewerOutput(
                    utterance=exp.get("utterance", ""),
                    action=exp["action"],
                    reasoning=exp.get("reasoning", ""),
                ),
            )
        )
    return cases


class AnalystCaseInputs(BaseModel):
    """A transcript to feed to the analyst. No expected_output — quality is
    judged by LLM scorers and deterministic checks, not output matching."""

    transcript: list[TurnLine]


def load_analyst_cases(
    path: str | Path,
) -> list[Case[AnalystCaseInputs, AnalysisUpdate, None]]:
    """Load YAML analyst cases. expected_output is always None."""
    raw = yaml.safe_load(Path(path).read_text())
    cases: list[Case[AnalystCaseInputs, AnalysisUpdate, None]] = []
    for entry in raw["cases"]:
        cases.append(
            Case(
                name=entry["name"],
                inputs=AnalystCaseInputs.model_validate(entry["inputs"]),
            )
        )
    return cases


def get_dataset_version(path: str | Path) -> str:
    raw = yaml.safe_load(Path(path).read_text())
    return raw.get("version", "unversioned")


__all__ = [
    "AnalystCaseInputs",
    "InterviewerCaseInputs",
    "InterviewerOutput",
    "ProbeSeed",
    "TurnLine",
    "get_dataset_version",
    "load_analyst_cases",
    "load_cases",
]