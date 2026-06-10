"""Tier 3 — focused simulation for 6 persona-specific behaviors.

Only personas that require live simulation (emergent multi-agent behavior)
are tested here. The other 4 personas (power_user, skeptical_buyer,
ai_skeptic, churn_risk) run as deterministic replay evals in test_replay.py.

    contradictory        — analyst must detect contradiction + probe it
    off_topic_rambler    — interviewer must fire off_topic redirect
    silent_respondent    — interviewer must handle silence gracefully and
                           end the call after 3 consecutive "Still there?"s
    context_compression  — analyst accumulates all covered subtopics
    context_drift        — interviewer doesn't re-ask facts stated in turn 1
    cooperative_respondent — call should reach wrap_up naturally

Stage variants (mid/end) pre-seed the DB with canned prior turns to test
behaviors at different call stages. Capped at 25 turns per case.

Evaluators:

    CoveredAllScripted      deterministic — all scripted questions asked
    CaughtContradiction     deterministic — contradictory: analyst found
                            contradiction AND a priority-1 probe was asked
    RedirectedOffTopic      deterministic — off_topic_rambler: off_topic action
                            fired at least once
    HandledSilenceGracefully deterministic — silent_respondent: said "Still
                            there?", said "Take your time.", ended call after
                            repeated silence
    ProbesAreSpecific       LLM judge — probes reference specific details
    StaleProbesBridged      LLM judge — old probes bridged back to prior context
    CoveredSubtopicsAccumulate LLM judge — analyst tracked expected topics
    NoContextDrift          LLM judge — interviewer didn't re-ask known facts

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
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, EvaluationReason
from sqlalchemy.pool import StaticPool
from sqlmodel import create_engine, select

load_dotenv()

from pydantic_ai import Agent

from voice_agent import state
from voice_agent.agents.analyst import run_analyst_safely
from evals.simulator import load_personas, simulate_turn
from voice_agent.models import AnalystDeps, Persona
from voice_agent.turn import run_speech_turn

_JUDGE_MODEL = "anthropic:claude-haiku-4-5-20251001"

MAX_TURNS = 25  # hard cap — conversations run to completion or this limit

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
    # "start" = fresh call; "mid" = partway through; "end" = near wrap-up
    stage: str = "start"


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
    # context_compression: covered_subtopics accumulated by analyst across the call
    final_covered_subtopics: list[str] = []
    # context_drift: all interviewer utterances for drift checking
    interviewer_utterances: list[str] = []
    # cooperative_respondent: turn index when wrap_up fired (None = never)
    wrap_up_turn_index: int | None = None


# ---------------------------------------------------------------------------
# Stage seeding — canned prior turns that set up mid / end starting points
# ---------------------------------------------------------------------------

# Canned turns used to pre-seed mid/end stages. Each entry is
# (interviewer_text, respondent_text). These are generic enough to apply to
# any persona and realistic enough to produce valid DB state.
_CANNED_TURNS: list[tuple[str, str]] = [
    (
        "How do you currently use this product?",
        "I use it mainly for meeting summaries — we have it connected to our Zoom account "
        "so it automatically captures all our team calls. I'd say we get about 20 calls a "
        "week transcribed and summarised.",
    ),
    (
        "What do you value most about it?",
        "Honestly, the time savings are huge. I used to spend 20–30 minutes after every "
        "call writing up notes and action items. Now I just review the AI summary, make a "
        "few edits if needed, and share it. Probably saves me an hour a day across the team.",
    ),
    (
        "Has anything frustrated you about it recently?",
        "The speaker attribution in large group calls can be off — when there are more than "
        "five people it sometimes mixes up who said what. We had an issue last month where "
        "a client call summary attributed a concern to the wrong person and it caused a bit "
        "of confusion. That's my main gripe.",
    ),
    (
        "Would you recommend it to someone else?",
        "Yes, I already have — two colleagues at other companies are now using it. I'd give "
        "it an 8 out of 10. The core functionality is solid; the edge cases like attribution "
        "and the mobile app performance are the things holding it back from a 10.",
    ),
]


def _seed_prior_history(
    engine,
    call_id: str,
    scripted_questions: list[str],
    stage: str,
) -> tuple[list[dict[str, str]], int]:
    """Seed the DB with canned prior turns for mid/end stages.

    Returns (history, next_db_turn_number) so the live loop can continue
    from the correct state.

    mid: 2 scripted asked, analyst snapshot with partial coverage
    end: all-but-last scripted asked, richer analyst snapshot
    """
    if stage == "start":
        return [], 1

    # mid: seed first 2 canned turns; end: seed first 4 (all available)
    turns_to_seed = 2 if stage == "mid" else len(_CANNED_TURNS)
    scripted_to_advance = min(turns_to_seed, len(scripted_questions))

    history: list[dict[str, str]] = []
    db_turn = 1

    with state.session_scope(engine) as s:
        call = s.get(state.Call, call_id)
        # Advance scripted cursor to reflect questions already asked
        call.scripted_cursor = scripted_to_advance

        for i in range(turns_to_seed):
            interviewer_text, respondent_text = _CANNED_TURNS[i]
            s.add(state.Turn(
                call_id=call_id,
                turn_number=db_turn,
                speaker="interviewer",
                text=interviewer_text,
                action="scripted",
            ))
            db_turn += 1
            s.add(state.Turn(
                call_id=call_id,
                turn_number=db_turn,
                speaker="respondent",
                text=respondent_text,
            ))
            db_turn += 1
            history.append({"speaker": "interviewer", "text": interviewer_text})
            history.append({"speaker": "respondent", "text": respondent_text})

        # Seed an analyst snapshot reflecting the seeded conversation
        covered = [
            "daily use for meeting summaries via Zoom integration (~20 calls/week)",
            "time savings: ~1 hour/day across the team",
            "speaker attribution errors in large group calls (5+ people)",
            "NPS/rating: 8/10; would recommend to others",
        ][:turns_to_seed]  # proportional to how many turns were seeded

        themes = ["time savings", "meeting summaries", "speaker attribution quality"]
        if stage == "end":
            themes.append("recommendation intent")

        s.add(state.AnalystSnapshot(
            call_id=call_id,
            after_turn=db_turn - 1,
            after_scripted_cursor=scripted_to_advance,
            themes=themes,
            covered_subtopics=covered,
            contradictions=[],
            surprises=[],
            investor_signals=[],
        ))

    return history, db_turn


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

    # Seed call record
    with state.session_scope(engine) as s:
        s.add(
            state.Call(
                id=call_id,
                scripted_questions=inputs.scripted_questions,
                status="active",
            )
        )

    # Pre-populate DB + history for mid/end stages so the live loop starts
    # partway through a realistic conversation.
    history, db_turn_number = _seed_prior_history(
        engine, call_id, inputs.scripted_questions, inputs.stage
    )

    turn_actions: list[str] = []
    off_topic_redirects_at: list[int] = []
    analyst_found_contradiction = False
    contradiction_probe_asked = False
    completed = False
    probe_turns: list[ProbeTurn] = []
    still_there_utterances: list[int] = []
    take_your_time_utterances: list[int] = []
    wrapped_up_after_silence = False
    interviewer_utterances: list[str] = []
    wrap_up_turn_index: int | None = None

    # For start stage: open with first scripted question (other stages already have history)
    if inputs.stage == "start":
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
        interviewer_utterances.append(out["message"])

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

        # --- Analyst pass — gated by should_run_analyst(), matching production behaviour ---
        with state.session_scope(engine) as s:
            _run_analyst = state.should_run_analyst(s, call_id)
        if _run_analyst:
            await run_analyst_safely(AnalystDeps(call_id=call_id, engine=engine))
        with state.session_scope(engine) as s:
            snapshot = state.latest_snapshot(s, call_id)
            if snapshot and snapshot.contradictions:
                analyst_found_contradiction = True

        if out["action"] == "wrap_up":
            completed = True
            wrap_up_turn_index = turn_idx
            # Detect wrap_up triggered by silence (preceded by a "Still there?" clarify)
            if still_there_utterances and turn_idx > 0 and (turn_idx - 1) in still_there_utterances:
                wrapped_up_after_silence = True
            break

    # Scripted coverage + final analyst snapshot
    with state.session_scope(engine) as s:
        call = s.get(state.Call, call_id)
        scripted_asked_count = call.scripted_cursor if call else 0
        snapshot = state.latest_snapshot(s, call_id)
        final_covered_subtopics = snapshot.covered_subtopics if snapshot else []

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
        final_covered_subtopics=final_covered_subtopics,
        interviewer_utterances=interviewer_utterances,
        wrap_up_turn_index=wrap_up_turn_index,
    )


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------


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


class _SubtopicCoverageJudgement(BaseModel):
    covered: list[str]
    missing: list[str]
    fraction_covered: float


_SUBTOPIC_COVERAGE_SYSTEM = (
    "You are checking whether a qualitative research analyst correctly tracked the topics "
    "discussed in a customer interview. You will receive:\n"
    "  1. EXPECTED_TOPICS — the themes the respondent was supposed to cover (from their persona)\n"
    "  2. COVERED_SUBTOPICS — the list the analyst accumulated in their snapshot\n\n"
    "For each expected topic, determine whether it is semantically represented in "
    "COVERED_SUBTOPICS. A covered_subtopic covers an expected topic if a human reading it "
    "would recognise it addresses the same subject (exact wording not required).\n\n"
    "Return:\n"
    "  covered: list of expected topics found in covered_subtopics\n"
    "  missing: list of expected topics absent from covered_subtopics\n"
    "  fraction_covered: len(covered) / len(expected_topics), or 1.0 if expected_topics is empty"
)

# Topics the context_compression persona is scripted to surface
_CONTEXT_COMPRESSION_EXPECTED_TOPICS = [
    "seat count (35 seats)",
    "respondent is economic buyer, not daily user",
    "SDR team usage (12 reps)",
    "time savings (30 minutes per rep per call)",
    "Salesforce CRM integration issue",
    "Gong competitor evaluation",
    "renewal timeline (2 months)",
    "NPS or rating (7/10)",
    "mobile app performance",
]


def _get_subtopic_coverage_agent() -> Agent[None, _SubtopicCoverageJudgement]:
    return Agent(
        _JUDGE_MODEL,
        output_type=_SubtopicCoverageJudgement,
        system_prompt=_SUBTOPIC_COVERAGE_SYSTEM,
    )


@dataclass
class CoveredSubtopicsAccumulate(Evaluator[TrajectoryInputs, TrajectoryResult, None]):
    """For the context_compression persona: the analyst's final covered_subtopics
    must collectively represent the expected topics discussed during the call.

    Uses an LLM judge to do semantic matching — a subtopic string doesn't need to
    exactly match an expected topic, just cover the same concept. Passes at ≥ 70%
    coverage (allows for organic variation in what the respondent actually said).

    Returns True (N/A pass) for all other personas.
    """

    threshold: float = 0.70

    async def evaluate(
        self,
        ctx: EvaluatorContext[TrajectoryInputs, TrajectoryResult, None],
    ) -> dict:
        if ctx.inputs.persona_name != "context_compression":
            return {"CoveredSubtopicsAccumulate": True}

        covered_subtopics = ctx.output.final_covered_subtopics
        if not covered_subtopics:
            return {"CoveredSubtopicsAccumulate": EvaluationReason(
                value=0.0,
                reason="analyst produced no covered_subtopics at all",
            )}

        topic_lines = "\n".join(f"  - {t}" for t in _CONTEXT_COMPRESSION_EXPECTED_TOPICS)
        subtopic_lines = "\n".join(f"  - {s}" for s in covered_subtopics)
        prompt = (
            f"EXPECTED_TOPICS:\n{topic_lines}\n\n"
            f"COVERED_SUBTOPICS (from analyst snapshot):\n{subtopic_lines}"
        )

        agent = _get_subtopic_coverage_agent()
        result = await agent.run(prompt)
        j = result.output

        reason = (
            f"missing: {j.missing}" if j.missing
            else f"all {len(j.covered)} expected topics found"
        )
        passed = j.fraction_covered >= self.threshold
        return {"CoveredSubtopicsAccumulate": EvaluationReason(
            value=j.fraction_covered,
            reason=reason,
        )}


class _ContextDriftJudgement(BaseModel):
    drifted: bool
    reason: str


_CONTEXT_DRIFT_SYSTEM = (
    "You are auditing an AI interviewer for CONTEXT DRIFT on a market-research phone call.\n\n"
    "The respondent stated these KEY FACTS in their very first turn:\n"
    "  - 50 seats purchased\n"
    "  - Only 8 people actively using it\n"
    "  - The respondent is the main internal champion\n\n"
    "Read ALL of the interviewer's utterances below and determine whether the interviewer "
    "at any point:\n"
    "  (a) asks about something the respondent already clearly answered (e.g. seat count, "
    "      adoption rate, their role), as if the earlier answer was forgotten, OR\n"
    "  (b) states or implies facts that directly contradict what the respondent said "
    "      (e.g. says 'so with your full team using it...' when only 8/50 are active).\n\n"
    "DRIFTED=true if any such forgetting or contradiction occurred.\n"
    "DRIFTED=false if the interviewer correctly remembered and built on the stated facts.\n\n"
    "Return: drifted (bool) and reason (one sentence describing what drifted or confirming consistency)."
)


def _get_context_drift_agent() -> Agent[None, _ContextDriftJudgement]:
    return Agent(_JUDGE_MODEL, output_type=_ContextDriftJudgement, system_prompt=_CONTEXT_DRIFT_SYSTEM)


@dataclass
class NoContextDrift(Evaluator[TrajectoryInputs, TrajectoryResult, None]):
    """For the context_drift persona: interviewer must not re-ask or contradict facts
    the respondent stated clearly in turn 1 (seat count, adoption rate, champion role).

    Uses an LLM judge that reads all interviewer utterances.
    Returns True (N/A pass) for all other personas.
    """

    async def evaluate(
        self,
        ctx: EvaluatorContext[TrajectoryInputs, TrajectoryResult, None],
    ) -> bool:
        if ctx.inputs.persona_name != "context_drift":
            return True
        utterances = ctx.output.interviewer_utterances
        if not utterances:
            return True
        numbered = "\n".join(f"  [{i+1}] {u}" for i, u in enumerate(utterances))
        prompt = f"Interviewer utterances (in order):\n{numbered}"
        agent = _get_context_drift_agent()
        result = await agent.run(prompt)
        return not result.output.drifted


# ---------------------------------------------------------------------------
# Dataset builder — only the personas requiring live simulation
# ---------------------------------------------------------------------------

_SIMULATION_PERSONAS = {
    "contradictory",
    "off_topic_rambler",
    "silent_respondent",
    "context_compression",
    "context_drift",
    "cooperative_respondent",
}

# Personas that get mid/end stage variants in addition to start.
# All others run start-only to keep the suite fast.
_MULTI_STAGE_PERSONAS: dict[str, list[str]] = {
    "context_drift": ["start", "mid"],
    "cooperative_respondent": ["start", "mid"],
}


def _build_dataset(only: set[str] | None = None) -> Dataset[TrajectoryInputs, TrajectoryResult, None]:
    all_personas = load_personas()
    sim_personas = [p for p in all_personas if p.name in _SIMULATION_PERSONAS]
    if only:
        sim_personas = [p for p in sim_personas if p.name in only]
    cases: list[Case[TrajectoryInputs, TrajectoryResult, None]] = []
    for p in sim_personas:
        stages = _MULTI_STAGE_PERSONAS.get(p.name, ["start"])
        for stage in stages:
            case_name = p.name if stage == "start" else f"{p.name}_{stage}"
            cases.append(Case(
                name=case_name,
                inputs=TrajectoryInputs(
                    persona_name=p.name,
                    persona_system=p.system,
                    scripted_questions=SCRIPTED_QUESTIONS,
                    stage=stage,
                ),
            ))
    return Dataset(
        name="trajectories_tier3_sim",
        cases=cases,
        evaluators=(
            CoveredAllScripted(),
            CaughtContradiction(),
            RedirectedOffTopic(),
            HandledSilenceGracefully(),
            ProbesAreSpecific(),
            StaleProbesBridged(),
            CoveredSubtopicsAccumulate(),
            NoContextDrift(),
        ),
    )


# ---------------------------------------------------------------------------
# Pytest entry point
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.slow
async def test_tier3_trajectories(cases_filter: set[str] | None):
    """Focused simulation for 6 personas testing emergent multi-turn behavior.

    Personas testing multi-agent dynamics (analyst ↔ interviewer) and trajectory
    properties that deterministic replay evals can't cover:
      - contradictory: analyst catches contradiction + interviewer probes it
      - off_topic_rambler: interviewer fires off_topic redirect (start + mid)
      - silent_respondent: silence handling + wrap_up after 3× "Still there?"
      - context_compression: analyst accumulates all covered subtopics (start + mid)
      - context_drift: interviewer doesn't re-ask facts from turn 1 (start/mid/end)
      - cooperative_respondent: call should reach wrap_up naturally (start/mid/end)

    Each case is capped at 25 turns. Stage variants (mid/end) pre-seed the DB
    with realistic prior conversation state so we cover calls that start at
    different points. Marked `slow`. Run with:
        uv run pytest evals/test_trajectories.py -v -s -m slow
        uv run pytest evals/test_trajectories.py -v -s -m slow --cases context_drift,context_compression
    """
    dataset = _build_dataset(only=cases_filter)
    report = await dataset.evaluate(
        run_trajectory,
        max_concurrency=6,
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
    assert scores.get("CoveredSubtopicsAccumulate", 1.0) >= 0.70, (
        f"context_compression: analyst only tracked "
        f"{scores.get('CoveredSubtopicsAccumulate', 0):.0%} of expected topics — "
        "covered_subtopics not accumulating across turns"
    )
    assert scores.get("NoContextDrift", 1.0) >= 1.0, (
        "context_drift persona: interviewer re-asked or contradicted facts from turn 1"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_results(report) -> None:
    cols = "  CS  CT  RT  PS  SB  CA  ND"
    header = f"{'case':<32}{cols}  actions"
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
        cs = _b(a, "CoveredAllScripted")
        ct = _b(a, "CaughtContradiction")
        rt = _b(a, "RedirectedOffTopic")
        ps = _b(sc, "ProbesAreSpecific")
        sb = _b(sc, "StaleProbesBridged")
        ca = _b(sc, "CoveredSubtopicsAccumulate")
        nd = _b(a, "NoContextDrift")
        actions = ", ".join(case.output.turn_actions) if case.output else ""
        print(f"{(case.name or ''):<32}{cs}{ct}{rt}{ps}{sb}{ca}{nd}  {actions[:60]}")

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