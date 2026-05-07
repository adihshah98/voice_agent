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
    Evaluators read `.action` and optionally `.probe_source` off the expected object."""
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
                    probe_source=exp.get("probe_source"),
                ),
            )
        )
    return cases


class PriorSnapshotSeed(BaseModel):
    """Snapshot established before the new transcript turns.

    Used to test incremental analyst behaviour — the analyst should build on
    the snapshot and not re-probe topics already covered in it.
    """

    after_turn: int
    themes: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    surprises: list[str] = Field(default_factory=list)
    investor_signals: list[str] = Field(default_factory=list)
    covered_subtopics: list[str] = Field(default_factory=list)


class AnalystCaseInputs(BaseModel):
    """A transcript to feed to the analyst.

    `expected_topics` are topics that must be surfaced in at least one generated
    probe (semantic match via LLMJudge). Missing a topic is a coverage failure.

    `prior_snapshot` seeds an `AnalystSnapshot` row before the transcript turns,
    forcing the analyst into incremental mode (only the turns after the snapshot
    are treated as new).
    """

    transcript: list[TurnLine]
    expected_topics: list[str] = Field(default_factory=list)
    prior_snapshot: Optional[PriorSnapshotSeed] = None


def load_analyst_cases(
    path: str | Path,
) -> list[Case[AnalystCaseInputs, AnalysisUpdate, None]]:
    """Load YAML analyst cases. expected_output is always None.

    `expected_topics` and `prior_snapshot` are top-level siblings of `inputs`
    in the YAML (to keep the transcript block readable), so we merge them into
    the inputs dict before validation.
    """
    raw = yaml.safe_load(Path(path).read_text())
    cases: list[Case[AnalystCaseInputs, AnalysisUpdate, None]] = []
    for entry in raw["cases"]:
        inputs_data = dict(entry["inputs"])
        if "expected_topics" in entry:
            inputs_data["expected_topics"] = entry["expected_topics"]
        if "prior_snapshot" in entry:
            inputs_data["prior_snapshot"] = entry["prior_snapshot"]
        cases.append(
            Case(
                name=entry["name"],
                inputs=AnalystCaseInputs.model_validate(inputs_data),
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
    "PriorSnapshotSeed",
    "ProbeSeed",
    "TurnLine",
    "get_dataset_version",
    "load_analyst_cases",
    "load_cases",
]
