"""Tier 3 — full-conversation trajectory eval.

Drives the interviewer + analyst against a simulated respondent (4 personas)
until `wrap_up` or a turn limit. Evaluators:

    CallCompletes          deterministic — reached wrap_up before turn limit
    CoveredAllScripted     deterministic — all scripted questions asked
    CaughtContradiction    deterministic — contradictory persona: analyst found
                           contradiction AND a priority-1 probe was asked
    RedirectedOffTopic     deterministic — off_topic_rambler: off_topic action
                           fired at least once
    ReportQuality          LLMJudge 1–5 — synthesis summary vs transcript

Run with:
    uv run pytest evals/test_trajectories.py -v -s
    uv run pytest evals/test_trajectories.py -v -s -k chatty_enthusiast   # single persona
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, LLMJudge
from sqlalchemy.pool import StaticPool
from sqlmodel import create_engine, select

load_dotenv()

from voice_agent import state
from voice_agent.agents.analyst import run_analyst_safely
from evals.simulator import load_personas, simulate_turn
from voice_agent.agents.interviewer import run_interviewer
from voice_agent.models import AnalystDeps, InterviewerDeps, Persona
from voice_agent.agents.synthesis import SynthesisDeps, run_synthesis_safely
from voice_agent.tracing import init_tracing

JUDGE_MODEL = "anthropic:claude-opus-4-6"
MAX_TURNS = 20  # respondent turns before we declare the call incomplete

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
    transcript_text: str
    synthesis_summary: str


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
        # --- Respondent turn ---
        respondent_text = await simulate_turn(persona, history)
        with state.session_scope(engine) as s:
            s.add(
                state.Turn(
                    call_id=call_id,
                    turn_number=db_turn_number,
                    speaker="respondent",
                    text=respondent_text,
                )
            )
        history.append({"speaker": "respondent", "text": respondent_text})
        db_turn_number += 1

        # --- Analyst pass (awaited for offline eval — no latency constraint) ---
        await run_analyst_safely(AnalystDeps(call_id=call_id, engine=engine))
        with state.session_scope(engine) as s:
            snapshot = state.latest_snapshot(s, call_id)
            if snapshot and snapshot.contradictions:
                analyst_found_contradiction = True

        # --- Interviewer turn ---
        with state.session_scope(engine) as s:
            deps = InterviewerDeps(
                call_id=call_id,
                session=s,
                turn_number=db_turn_number,
            )
            out = await run_interviewer(deps, respondent_text)

            # Track contradiction probe: probe asked after analyst saw a contradiction
            if out.action == "probe" and analyst_found_contradiction:
                contradiction_probe_asked = True

            s.add(
                state.Turn(
                    call_id=call_id,
                    turn_number=db_turn_number,
                    speaker="interviewer",
                    text=out.utterance,
                    action=out.action,
                    reasoning=out.reasoning,
                )
            )

        history.append({"speaker": "interviewer", "text": out.utterance})
        turn_actions.append(out.action)

        if out.action == "off_topic":
            off_topic_redirects_at.append(db_turn_number)

        db_turn_number += 1

        if out.action == "wrap_up":
            completed = True
            break

    # Scripted coverage
    with state.session_scope(engine) as s:
        call = s.get(state.Call, call_id)
        scripted_asked_count = call.scripted_cursor if call else 0

    # Synthesis (post-call report)
    synthesis_summary = ""
    with state.session_scope(engine) as s:
        s_deps = SynthesisDeps(call_id=call_id, session=s)
        report = await run_synthesis_safely(s_deps)
        if report:
            synthesis_summary = report.summary

    transcript_text = "\n".join(
        f"{t['speaker'].upper()}: {t['text']}" for t in history
    )

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
        transcript_text=transcript_text,
        synthesis_summary=synthesis_summary,
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


def report_quality_judge() -> LLMJudge:
    """LLMJudge: is the synthesis summary accurate and useful vs the transcript?"""
    return LLMJudge(
        rubric=(
            "You are grading a POST-CALL SYNTHESIS SUMMARY produced by an AI "
            "market-research agent after a simulated phone interview.\n\n"
            "The output you are grading contains:\n"
            "  `transcript_text` — the full conversation\n"
            "  `synthesis_summary` — the post-call summary to evaluate\n\n"
            "Score 1–5 on REPORT QUALITY:\n"
            "  5 = accurately reflects the transcript; cites specific details; "
            "no hallucinations; identifies key themes; actionable\n"
            "  4 = mostly accurate; minor omissions or slight over-generalisation\n"
            "  3 = captures the gist but misses important details or makes vague "
            "claims that aren't directly supported\n"
            "  2 = significant omissions or minor inaccuracies relative to the "
            "transcript\n"
            "  1 = does not reflect the transcript, or contains hallucinations, "
            "or synthesis_summary is empty\n\n"
            "Pass (score >= 3) if the summary is a reasonable reflection of what "
            "was discussed. Fail if it introduces facts not in the transcript."
        ),
        model=JUDGE_MODEL,
        include_input=False,
        score={"evaluation_name": "report_quality"},
        assertion=False,
    )


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------


def _build_dataset() -> Dataset[TrajectoryInputs, TrajectoryResult, None]:
    personas = load_personas()
    cases: list[Case[TrajectoryInputs, TrajectoryResult, None]] = [
        Case(
            name=p.name,
            inputs=TrajectoryInputs(
                persona_name=p.name,
                persona_system=p.system,
                scripted_questions=SCRIPTED_QUESTIONS,
            ),
        )
        for p in personas
    ]
    return Dataset(
        name="trajectories_tier3",
        cases=cases,
        evaluators=(
            CallCompletes(),
            CoveredAllScripted(),
            CaughtContradiction(),
            RedirectedOffTopic(),
            report_quality_judge(),
        ),
    )


# ---------------------------------------------------------------------------
# Pytest entry point
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.slow
async def test_tier3_trajectories():
    """Full trajectory eval across all 4 personas.

    Marked `slow` — each run makes ~40–80 LLM calls. Run with:
        uv run pytest evals/test_trajectories.py -v -s -m slow
    """
    init_tracing(service_name="voice-agent-evals", send_to_logfire=False)

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

    assert scores.get("CallCompletes", 0) >= 1, (
        "Not all calls reached wrap_up — interviewer isn't closing calls"
    )
    assert scores.get("CoveredAllScripted", 0) >= 0.75, (
        "Fewer than 75% of calls covered all scripted questions"
    )
    assert scores.get("CaughtContradiction", 1) >= 1.0, (
        "contradictory persona: analyst failed to catch the contradiction or probe it"
    )
    assert scores.get("RedirectedOffTopic", 1) >= 1.0, (
        "off_topic_rambler persona: interviewer never redirected the respondent"
    )

    quality = scores.get("report_quality")
    if quality is not None:
        assert quality >= 3.0, (
            f"Synthesis report quality {quality:.2f} below 3.0/5 — "
            "reports are not accurately reflecting transcripts"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_results(report) -> None:
    cols = "  CC  CS  CT  RT  RQ"
    header = f"{'persona':<24}{cols}  actions"
    print(f"\nTier 3 trajectory results:\n{header}\n{'-' * len(header)}")

    def _b(src, key) -> str:
        item = (src or {}).get(key)
        if item is None:
            return "  - "
        return " ok " if item.value else "FAIL"

    def _n(src, key) -> str:
        item = (src or {}).get(key)
        if item is None:
            return "  - "
        v = item.value
        return f"{v:3.1f}" if isinstance(v, (int, float)) else "  - "

    for case in report.cases:
        a = case.assertions or {}
        sc = case.scores or {}
        cc = _b(a, "CallCompletes")
        cs = _b(a, "CoveredAllScripted")
        ct = _b(a, "CaughtContradiction")
        rt = _b(a, "RedirectedOffTopic")
        rq = _n(sc, "report_quality")
        actions = ", ".join(case.output.turn_actions) if case.output else ""
        print(f"{(case.name or ''):<24}{cc}{cs}{ct}{rt}{rq:>5}  {actions[:60]}")

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