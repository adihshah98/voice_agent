"""Tier 3 — focused simulation for 3 persona-specific behaviors.

Only personas that require live simulation (emergent multi-agent behavior)
are tested here. The other 4 personas (power_user, skeptical_buyer,
ai_skeptic, churn_risk) run as deterministic replay evals in test_replay.py.

    contradictory      — analyst must detect contradiction + probe it
    off_topic_rambler  — interviewer must fire off_topic redirect
    silent_respondent  — interviewer must handle silence gracefully and
                         end the call after 3 consecutive "Still there?"s

Evaluators:

    CallCompletes           deterministic — reached wrap_up before turn limit
    CoveredAllScripted      deterministic — all scripted questions asked
    CaughtContradiction     deterministic — contradictory: analyst found
                            contradiction AND a priority-1 probe was asked
    RedirectedOffTopic      deterministic — off_topic_rambler: off_topic action
                            fired at least once
    HandledSilenceGracefully deterministic — silent_respondent: said "Still
                            there?", said "Take your time.", ended call after
                            repeated silence

Capped at 12 turns per persona. ~36 Sonnet calls ≈ 60s.

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

from pydantic_ai import Agent

from voice_agent import state
from voice_agent.agents.analyst import run_analyst_safely
from evals.simulator import load_personas, simulate_turn
from voice_agent.models import AnalystDeps, Persona
from voice_agent.tracing import init_tracing
from voice_agent.turn import run_speech_turn

_JUDGE_MODEL = "anthropic:claude-sonnet-4-6"

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


class ProbeTurn(BaseModel):
    """Snapshot of one probe turn for utterance-quality evaluation."""
    turn_index: int
    utterance: str
    # Conversation up to and including the respondent turn that triggered this probe
    conversation_so_far: list[dict[str, str]]
    # Whether the probe was generated 3+ turns before it was asked (bridging required)
    stale: bool


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
    # Probe turns for quality evaluation
    probe_turns: list[ProbeTurn] = []
    # Silence handling tracking
    still_there_utterances: list[int] = []    # turn indices where "Still there?" was said
    take_your_time_utterances: list[int] = []  # turn indices where "Take your time." was said
    wrapped_up_after_silence: bool = False     # True if wrap_up fired after repeated silence


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
    probe_turns: list[ProbeTurn] = []
    still_there_utterances: list[int] = []
    take_your_time_utterances: list[int] = []
    wrapped_up_after_silence = False

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
        turn_idx = len(turn_actions) - 1

        if out["action"] == "off_topic":
            off_topic_redirects_at.append(db_turn_number - 1)

        msg_lower = out["message"].lower()
        if "still there" in msg_lower:
            still_there_utterances.append(turn_idx)
        if "take your time" in msg_lower:
            take_your_time_utterances.append(turn_idx)

        # Track contradiction probe: probe asked on a turn where analyst had already flagged one
        if out["action"] == "probe" and analyst_found_contradiction:
            contradiction_probe_asked = True

        # Collect probe turn for quality evaluation
        if out["action"] == "probe":
            # Check if this probe was stale (generated 3+ turns before being asked)
            stale = False
            with state.session_scope(engine) as s:
                # Find most recently asked probe (just marked asked by commit())
                from sqlmodel import select as _select
                from voice_agent import state as _state
                asked_probe = s.exec(
                    _select(_state.Probe)
                    .where(_state.Probe.call_id == call_id, _state.Probe.asked == True)  # noqa: E712
                    .order_by(_state.Probe.asked_at.desc())
                    .limit(1)
                ).first()
                if asked_probe and asked_probe.generated_after_turn is not None:
                    turns_ago = db_turn_number - asked_probe.generated_after_turn
                    stale = turns_ago >= 3
            probe_turns.append(ProbeTurn(
                turn_index=len(turn_actions) - 1,
                utterance=out["message"],
                conversation_so_far=list(history[:-1]),  # exclude the just-added interviewer turn
                stale=stale,
            ))

        # --- Analyst pass — runs after commit(), matching production fire-and-forget order ---
        await run_analyst_safely(AnalystDeps(call_id=call_id, engine=engine))
        with state.session_scope(engine) as s:
            snapshot = state.latest_snapshot(s, call_id)
            if snapshot and snapshot.contradictions:
                analyst_found_contradiction = True

        if out["action"] == "wrap_up":
            completed = True
            # Detect wrap_up triggered by silence (preceded by a "Still there?" clarify)
            if still_there_utterances and turn_idx > 0 and (turn_idx - 1) in still_there_utterances:
                wrapped_up_after_silence = True
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
        probe_turns=probe_turns,
        still_there_utterances=still_there_utterances,
        take_your_time_utterances=take_your_time_utterances,
        wrapped_up_after_silence=wrapped_up_after_silence,
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


@dataclass
class HandledSilenceGracefully(Evaluator[TrajectoryInputs, TrajectoryResult, None]):
    """For the silent_respondent persona: interviewer must:
      1. Respond to silence with "Still there?" (at least once).
      2. Respond to a thinking filler with "Take your time." (at least once).
      3. End the call (wrap_up) after repeated silence, not just keep asking questions.

    Returns True (N/A pass) for other personas.
    """

    def evaluate(
        self,
        ctx: EvaluatorContext[TrajectoryInputs, TrajectoryResult, None],
    ) -> bool:
        if ctx.inputs.persona_name != "silent_respondent":
            return True
        r = ctx.output
        asked_still_there = len(r.still_there_utterances) >= 1
        asked_take_your_time = len(r.take_your_time_utterances) >= 1
        ended_call = r.wrapped_up_after_silence or (
            r.completed and len(r.still_there_utterances) >= 3
        )
        return asked_still_there and asked_take_your_time and ended_call


class _ProbeQualityJudgement(BaseModel):
    specific: bool
    reason: str


_PROBE_SPECIFIC_SYSTEM = (
    "You are grading a single interviewer probe utterance from a market research phone call.\n\n"
    "PASS (specific=true) if the probe references at least one concrete detail the respondent "
    "actually mentioned — a named feature, a number, a person, a competitor, a workflow, "
    "or a concrete event from the conversation.\n\n"
    "FAIL (specific=false) if the probe is vague and could apply to any respondent without "
    "reading this conversation. Examples that should FAIL:\n"
    "  - 'Can you tell me more about your day-to-day use?'\n"
    "  - 'What does that look like for your team?'\n"
    "  - 'Can you elaborate on that?'\n\n"
    "Examples that should PASS:\n"
    "  - 'You mentioned the action items feature — how often does that attribution error happen?'\n"
    "  - 'Earlier you said you were evaluating Fireflies — what specifically prompted that?'\n"
    "  - 'You said it saves about an hour a week — is that per person or across the whole team?'\n\n"
    "Return: specific (bool) and reason (one sentence)."
)

_PROBE_BRIDGED_SYSTEM = (
    "You are grading whether an interviewer properly bridges back to earlier conversation "
    "before asking a probe that was generated 3 or more turns ago.\n\n"
    "The interviewer's instructions say: when a probe is 3–8 turns old, bridge with "
    "'Earlier you mentioned X...' before asking, so it doesn't feel out of nowhere.\n\n"
    "PASS (specific=true) if the utterance:\n"
    "  - references something the respondent said earlier before the probe question, OR\n"
    "  - naturally follows the current turn's content (topic came up again organically)\n\n"
    "FAIL (specific=false) if the utterance jumps straight to a new question with no "
    "reference to prior context, making it feel disconnected.\n\n"
    "Return: specific (bool, True=bridged/passed) and reason (one sentence)."
)


def _get_probe_quality_agent() -> Agent[None, _ProbeQualityJudgement]:
    return Agent(_JUDGE_MODEL, output_type=_ProbeQualityJudgement, system_prompt=_PROBE_SPECIFIC_SYSTEM)


def _get_probe_bridged_agent() -> Agent[None, _ProbeQualityJudgement]:
    return Agent(_JUDGE_MODEL, output_type=_ProbeQualityJudgement, system_prompt=_PROBE_BRIDGED_SYSTEM)


@dataclass
class ProbesAreSpecific(Evaluator[TrajectoryInputs, TrajectoryResult, None]):
    """All probe utterances must reference specific details from the conversation.

    Catches the 'tell me more about your day-to-day' antipattern — generic probes
    that don't tie back to anything the respondent said. Scores as fraction of
    probe turns that pass (0.0–1.0). Cases with no probe turns score 1.0.
    """

    async def evaluate(
        self, ctx: EvaluatorContext[TrajectoryInputs, TrajectoryResult, None]
    ) -> float:
        probe_turns = ctx.output.probe_turns
        if not probe_turns:
            return 1.0

        agent = _get_probe_quality_agent()
        results = []
        for pt in probe_turns:
            conv = "\n".join(
                f"{'Interviewer' if t['speaker'] == 'interviewer' else 'Respondent'}: {t['text']}"
                for t in pt.conversation_so_far
            )
            prompt = f"Conversation:\n{conv}\n\nProbe utterance to grade:\n{pt.utterance}"
            result = await agent.run(prompt)
            results.append(result.output.specific)

        return sum(results) / len(results)


@dataclass
class StaleProbesBridged(Evaluator[TrajectoryInputs, TrajectoryResult, None]):
    """Probes asked 3+ turns after generation must bridge back to prior context.

    Returns 1.0 (N/A pass) when there are no stale probe turns.
    """

    async def evaluate(
        self, ctx: EvaluatorContext[TrajectoryInputs, TrajectoryResult, None]
    ) -> float:
        stale_turns = [pt for pt in ctx.output.probe_turns if pt.stale]
        if not stale_turns:
            return 1.0

        agent = _get_probe_bridged_agent()
        results = []
        for pt in stale_turns:
            conv = "\n".join(
                f"{'Interviewer' if t['speaker'] == 'interviewer' else 'Respondent'}: {t['text']}"
                for t in pt.conversation_so_far
            )
            prompt = (
                f"Conversation so far (this probe was generated 3+ turns ago):\n{conv}"
                f"\n\nProbe utterance to grade:\n{pt.utterance}"
            )
            result = await agent.run(prompt)
            results.append(result.output.specific)

        return sum(results) / len(results)


# ---------------------------------------------------------------------------
# Dataset builder — only the 2 personas requiring live simulation
# ---------------------------------------------------------------------------

_SIMULATION_PERSONAS = {"contradictory", "off_topic_rambler", "silent_respondent"}


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
            HandledSilenceGracefully(),
            ProbesAreSpecific(),
            StaleProbesBridged(),
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
    init_tracing(service_name="voice-agent-evals")

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
    assert scores.get("ProbesAreSpecific", 1.0) >= 0.8, (
        f"probe specificity {scores.get('ProbesAreSpecific', 0):.0%} below 80% — "
        "interviewer asking vague 'tell me more about day-to-day' style probes"
    )
    assert scores.get("StaleProbesBridged", 1.0) >= 0.75, (
        f"stale probe bridging {scores.get('StaleProbesBridged', 0):.0%} below 75% — "
        "interviewer not saying 'earlier you mentioned' when picking up old probes"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_results(report) -> None:
    cols = "  CC  CS  CT  RT  PS  SB"
    header = f"{'persona':<24}{cols}  actions"
    print(f"\nTier 3 simulation results:\n{header}\n{'-' * len(header)}")

    def _b(src, key) -> str:
        item = (src or {}).get(key)
        if item is None:
            return "  - "
        val = item.value
        if isinstance(val, bool):
            return " ok " if val else "FAIL"
        if isinstance(val, (int, float)):
            return f"{val:.2f}"
        return "  - "

    for case in report.cases:
        a = case.assertions or {}
        sc = case.scores or {}
        cc = _b(a, "CallCompletes")
        cs = _b(a, "CoveredAllScripted")
        ct = _b(a, "CaughtContradiction")
        rt = _b(a, "RedirectedOffTopic")
        ps = _b(sc, "ProbesAreSpecific")
        sb = _b(sc, "StaleProbesBridged")
        actions = ", ".join(case.output.turn_actions) if case.output else ""
        print(f"{(case.name or ''):<24}{cc}{cs}{ct}{rt}{ps}{sb}  {actions[:60]}")

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