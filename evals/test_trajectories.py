"""Tier 3 — focused simulation for 2 persona-specific behaviors.

Only personas that require live simulation (emergent multi-agent behavior)
are tested here. The other 4 personas (power_user, skeptical_buyer,
ai_skeptic, churn_risk) run as deterministic replay evals in test_replay.py.

    contradictory     — analyst must detect contradiction + probe it
    off_topic_rambler — interviewer must fire off_topic redirect

Evaluators:

    CallCompletes          deterministic — reached wrap_up before turn limit
    CoveredAllScripted     deterministic — all scripted questions asked
    CaughtContradiction    deterministic — contradictory: analyst found
                           contradiction AND a priority-1 probe was asked
    RedirectedOffTopic     deterministic — off_topic_rambler: off_topic action
                           fired at least once

Capped at 12 turns per persona. ~24 Sonnet calls ≈ 45s.

Run with:
    uv run pytest evals/test_trajectories.py -v -s -m slow
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from sqlalchemy.pool import StaticPool
from sqlmodel import create_engine, select

load_dotenv()

from voice_agent import state
from voice_agent.agents.analyst import run_analyst_safely
from evals.simulator import load_personas, simulate_turn
from voice_agent.models import AnalystDeps, Persona
import logfire
from voice_agent.turn import run_speech_turn

MAX_TURNS = 12  # capped at 12 turns; ~24 Sonnet calls total across 2 personas

SCRIPTED_QUESTIONS = [
    "How do you currently use this product?",
    "What do you value most about it?",
    "Has anything frustrated you about it recently?",
    "Would you recommend it to someone else?",
    "What would make you use it more often?",
]


# ---------------------------------------------------------------------------
# Trajectory input / output types
# ---------------------------------------------------------------------------


class TrajectoryInputs(BaseModel):
    persona_name: str
    persona_system: str
    scripted_questions: list[str]
    max_turns: int = MAX_TURNS


class TrajectoryResult(BaseModel):
    persona_name: str
    completed: bool
    total_db_turns: int
    scripted_asked_count: int
    scripted_total: int
    off_topic_redirects_at: list[int]
    analyst_found_contradiction: bool
    contradiction_probe_asked: bool
    turn_actions: list[str]


# ---------------------------------------------------------------------------
# Trajectory runner (the eval "task" function)
# ---------------------------------------------------------------------------


async def run_trajectory(inputs: TrajectoryInputs) -> TrajectoryResult:
    """Drive a full simulated conversation and return a structured result."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    state.init_db(engine)
    call_id = f"traj-{inputs.persona_name}"
    persona = Persona(name=inputs.persona_name, system=inputs.persona_system)

    # Seed call
    with state.session_scope(engine) as s:
        s.add(
            state.Call(
                id=call_id,
                scripted_questions=inputs.scripted_questions,
                status="active",
            )
        )

    history: list[dict[str, str]] = []
    turn_actions: list[str] = []
    off_topic_redirects_at: list[int] = []
    analyst_found_contradiction = False
    contradiction_probe_asked = False
    completed = False

    # Interviewer opens with the first scripted question
    db_turn_number = 1
    with state.session_scope(engine) as s:
        opening = state.next_scripted(s, call_id)
        state.mark_scripted_asked(s, call_id)

    opening_text = opening or "Tell me about your experience with this product."
    with state.session_scope(engine) as s:
        s.add(
            state.Turn(
                call_id=call_id,
                turn_number=db_turn_number,
                speaker="interviewer",
                text=opening_text,
                action="scripted",
            )
        )
    history.append({"speaker": "interviewer", "text": opening_text})
    turn_actions.append("scripted")
    db_turn_number += 1

    for _ in range(inputs.max_turns):
        # --- Respondent turn (simulated) ---
        respondent_text = await simulate_turn(persona, history)
        history.append({"speaker": "respondent", "text": respondent_text})

        # --- Interviewer turn via TurnPipeline (production path) ---
        # commit() writes both the respondent and interviewer Turn rows.
        # vapi_messages includes the new respondent utterance as the last "user" message.
        vapi_messages = [
            {"role": "assistant" if t["speaker"] == "interviewer" else "user", "content": t["text"]}
            for t in history
        ]
        out = await run_speech_turn(engine, call_id, vapi_messages=vapi_messages)
        db_turn_number += 2  # commit() wrote respondent + interviewer rows

        history.append({"speaker": "interviewer", "text": out["message"]})
        turn_actions.append(out["action"])

        if out["action"] == "off_topic":
            off_topic_redirects_at.append(db_turn_number - 1)

        # Track contradiction probe: probe asked on a turn where analyst had already flagged one
        if out["action"] == "probe" and analyst_found_contradiction:
            contradiction_probe_asked = True

        # --- Analyst pass — runs after commit(), matching production fire-and-forget order ---
        await run_analyst_safely(AnalystDeps(call_id=call_id, engine=engine))
        with state.session_scope(engine) as s:
            snapshot = state.latest_snapshot(s, call_id)
            if snapshot and snapshot.contradictions:
                analyst_found_contradiction = True

        if out["action"] == "wrap_up":
            completed = True
            break

    # Scripted coverage
    with state.session_scope(engine) as s:
        call = s.get(state.Call, call_id)
        scripted_asked_count = call.scripted_cursor if call else 0

    return TrajectoryResult(
        persona_name=inputs.persona_name,
        completed=completed,
        total_db_turns=db_turn_number - 1,
        scripted_asked_count=scripted_asked_count,
        scripted_total=len(inputs.scripted_questions),
        off_topic_redirects_at=off_topic_redirects_at,
        analyst_found_contradiction=analyst_found_contradiction,
        contradiction_probe_asked=contradiction_probe_asked,
        turn_actions=turn_actions,
    )


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------


@dataclass
class CallCompletes(Evaluator[TrajectoryInputs, TrajectoryResult, None]):
    """Call must reach wrap_up before the turn limit."""

    def evaluate(
        self,
        ctx: EvaluatorContext[TrajectoryInputs, TrajectoryResult, None],
    ) -> bool:
        return ctx.output.completed


@dataclass
class CoveredAllScripted(Evaluator[TrajectoryInputs, TrajectoryResult, None]):
    """All scripted questions should be asked during the call."""

    def evaluate(
        self,
        ctx: EvaluatorContext[TrajectoryInputs, TrajectoryResult, None],
    ) -> bool:
        if ctx.output.scripted_total == 0:
            return True
        return ctx.output.scripted_asked_count >= ctx.output.scripted_total


@dataclass
class CaughtContradiction(Evaluator[TrajectoryInputs, TrajectoryResult, None]):
    """For the contradictory persona: analyst must find the contradiction AND
    a priority-1 probe must be asked as a result.

    Returns True (N/A pass) for all other personas.
    """

    def evaluate(
        self,
        ctx: EvaluatorContext[TrajectoryInputs, TrajectoryResult, None],
    ) -> bool:
        if ctx.inputs.persona_name != "contradictory":
            return True
        return ctx.output.analyst_found_contradiction and ctx.output.contradiction_probe_asked


@dataclass
class RedirectedOffTopic(Evaluator[TrajectoryInputs, TrajectoryResult, None]):
    """For the off_topic_rambler persona: interviewer must use off_topic action
    at least once. Returns True (N/A pass) for other personas."""

    def evaluate(
        self,
        ctx: EvaluatorContext[TrajectoryInputs, TrajectoryResult, None],
    ) -> bool:
        if ctx.inputs.persona_name != "off_topic_rambler":
            return True
        return len(ctx.output.off_topic_redirects_at) > 0


# ---------------------------------------------------------------------------
# Dataset builder — only the 2 personas requiring live simulation
# ---------------------------------------------------------------------------

_SIMULATION_PERSONAS = {"contradictory", "off_topic_rambler"}


def _build_dataset() -> Dataset[TrajectoryInputs, TrajectoryResult, None]:
    all_personas = load_personas()
    sim_personas = [p for p in all_personas if p.name in _SIMULATION_PERSONAS]
    cases: list[Case[TrajectoryInputs, TrajectoryResult, None]] = [
        Case(
            name=p.name,
            inputs=TrajectoryInputs(
                persona_name=p.name,
                persona_system=p.system,
                scripted_questions=SCRIPTED_QUESTIONS,
            ),
        )
        for p in sim_personas
    ]
    return Dataset(
        name="trajectories_tier3_sim",
        cases=cases,
        evaluators=(
            CallCompletes(),
            CoveredAllScripted(),
            CaughtContradiction(),
            RedirectedOffTopic(),
        ),
    )


# ---------------------------------------------------------------------------
# Pytest entry point
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.slow
async def test_tier3_trajectories():
    """Focused simulation for contradictory + off_topic_rambler personas.

    These are the only 2 personas testing emergent multi-agent behavior
    (analyst ↔ interviewer) that replay evals can't cover. The other 4
    personas run as deterministic replay evals in test_replay.py.

    Marked `slow` — ~24 Sonnet calls ≈ 45s. Run with:
        uv run pytest evals/test_trajectories.py -v -s -m slow
    """
    logfire.configure(service_name="voice-agent-evals", send_to_logfire="if-token-present")

    dataset = _build_dataset()
    report = await dataset.evaluate(
        run_trajectory,
        max_concurrency=2,
        progress=False,
    )

    _print_results(report)

    assert not report.failures, (
        f"{len(report.failures)} trajectory(ies) errored: "
        + ", ".join(f.name or "?" for f in report.failures)
    )

    scores = _aggregate(report)
    print("\nAggregate scores:")
    for name, value in sorted(scores.items()):
        print(f"  {name:<28} {value:.3f}")

    assert scores.get("CaughtContradiction", 1) >= 1.0, (
        "contradictory persona: analyst failed to catch the contradiction or probe it"
    )
    assert scores.get("RedirectedOffTopic", 1) >= 1.0, (
        "off_topic_rambler persona: interviewer never redirected the respondent"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_results(report) -> None:
    cols = "  CC  CS  CT  RT"
    header = f"{'persona':<24}{cols}  actions"
    print(f"\nTier 3 simulation results:\n{header}\n{'-' * len(header)}")

    def _b(src, key) -> str:
        item = (src or {}).get(key)
        if item is None:
            return "  - "
        return " ok " if item.value else "FAIL"

    for case in report.cases:
        a = case.assertions or {}
        cc = _b(a, "CallCompletes")
        cs = _b(a, "CoveredAllScripted")
        ct = _b(a, "CaughtContradiction")
        rt = _b(a, "RedirectedOffTopic")
        actions = ", ".join(case.output.turn_actions) if case.output else ""
        print(f"{(case.name or ''):<24}{cc}{cs}{ct}{rt}  {actions[:60]}")

        if case.output and not case.output.completed:
            print(f"  [did not complete — {case.output.total_db_turns} db turns]")
        if case.output and case.output.off_topic_redirects_at:
            print(f"  [off_topic at turns: {case.output.off_topic_redirects_at}]")

    for fail in report.failures:
        msg = (
            (fail.error_message or "unknown error").splitlines()[0][:70]
        )
        print(f"{(fail.name or ''):<24} ERROR  {msg}")


def _aggregate(report) -> dict[str, float]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for case in report.cases:
        for src in (case.assertions, case.scores):
            for key, item in (src or {}).items():
                val = item.value
                if isinstance(val, bool):
                    num = 1.0 if val else 0.0
                elif isinstance(val, (int, float)):
                    num = float(val)
                else:
                    continue
                sums[key] = sums.get(key, 0.0) + num
                counts[key] = counts.get(key, 0) + 1
    return {k: sums[k] / counts[k] for k in sums}