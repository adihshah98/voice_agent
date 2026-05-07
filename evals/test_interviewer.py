"""Tier 1 — single-turn decision eval for the interviewer agent.

Each case seeds an in-memory SQLite DB from `InterviewerCaseInputs`, runs
`interviewer.run(...)` once, and scores the returned `InterviewerOutput`.

Requires ANTHROPIC_API_KEY in .env. Run with:

    uv run pytest evals/test_interviewer.py -v

Pass criteria:
    - ActionMatches >= 90%  (action + probe_source when specified)
    - UtteranceWarmth average >= 4 / 5
    - SingleQuestion 100% (deterministic)
    - NoLeadingQuestions >= 90%
    - ResponseRelevant >= 90%
"""

from __future__ import annotations

from pathlib import Path

import pytest
from dotenv import load_dotenv
from sqlalchemy.pool import StaticPool
from sqlmodel import create_engine

load_dotenv()

from voice_agent import state
from evals.cases import InterviewerCaseInputs, load_cases
from evals.evaluators import (
    ActionMatches,
    SingleQuestion,
    no_leading_questions_judge,
    response_relevance_judge,
    utterance_warmth_judge,
)
import logfire
from voice_agent.models import InterviewerOutput
from voice_agent.turn import run_speech_turn
from evals.cases import get_dataset_version
from pydantic_evals import Dataset


DATASET_PATH = Path(__file__).parent / "datasets" / "interviewer_turns.yaml"
ACTION_MATCH_THRESHOLD = 0.90
WARMTH_THRESHOLD = 4.0
NON_LEADING_THRESHOLD = 0.90
RESPONSE_RELEVANT_THRESHOLD = 0.90
SINGLE_QUESTION_THRESHOLD = 1.0


def _seed_engine(inputs: InterviewerCaseInputs):
    """Fresh in-memory DB with a single call seeded from `inputs`.

    StaticPool keeps the `:memory:` connection shared across sessions (without
    it, each checkout gets its own empty DB).
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    state.init_db(engine)
    call_id = "eval-call"
    scripted_asks = sum(
        1
        for t in inputs.prior_turns
        if t.speaker == "interviewer" and t.action == "scripted"
    )
    cursor = min(scripted_asks, len(inputs.scripted_questions))
    with state.session_scope(engine) as s:
        s.add(
            state.Call(
                id=call_id,
                phone_number="+15550199",
                scripted_questions=list(inputs.scripted_questions),
                scripted_cursor=cursor,
                status="active",
            )
        )
        for i, t in enumerate(inputs.prior_turns, start=1):
            s.add(
                state.Turn(
                    call_id=call_id,
                    turn_number=i,
                    speaker=t.speaker,
                    text=t.text,
                    action=t.action,
                )
            )
        for p in inputs.probes:
            s.add(
                state.Probe(
                    call_id=call_id,
                    question=p.question,
                    priority=p.priority,
                    rationale=p.rationale,
                )
            )
    return engine, call_id, len(inputs.prior_turns)


async def run_interviewer_on_case(
    inputs: InterviewerCaseInputs,
) -> InterviewerOutput:
    engine, call_id, _ = _seed_engine(inputs)

    # Mirror what Vapi sends as body["messages"] to the LLM endpoint.
    # TurnPipeline.commit() will write the respondent + interviewer Turn rows —
    # do not add last_respondent to prior_turns in _seed_engine.
    vapi_messages: list[dict] = [
        {"role": "assistant" if t.speaker == "interviewer" else "user", "content": t.text}
        for t in inputs.prior_turns
    ]
    vapi_messages.append({"role": "user", "content": inputs.last_respondent})

    result = await run_speech_turn(engine, call_id, vapi_messages=vapi_messages)
    return InterviewerOutput(
        utterance=result["message"],
        action=result["action"],
        reasoning=result["reasoning"],
    )


@pytest.mark.asyncio
async def test_tier1_interviewer_decisions():
    dataset_version = get_dataset_version(DATASET_PATH)
    logfire.set_attribute("dataset_version", dataset_version)

    dataset: Dataset[InterviewerCaseInputs, InterviewerOutput, None] = Dataset(
        name="interviewer_tier1",
        cases=load_cases(DATASET_PATH),
        evaluators=(
            ActionMatches(),
            SingleQuestion(),
            utterance_warmth_judge(),
            no_leading_questions_judge(),
            response_relevance_judge(),
        ),
    )

    report = await dataset.evaluate(
        run_interviewer_on_case,
        max_concurrency=2,
        progress=True,
    )

    import logfire as _logfire
    _logfire.force_flush()

    _print_per_case(report)
    scores = _aggregate(report)
    print("\nAggregate:")
    for name, value in sorted(scores.items()):
        print(f"  {name:24s} {value:.3f}")

    assert not report.failures, (
        f"{len(report.failures)} case(s) errored: "
        + ", ".join(f.name or "?" for f in report.failures)
    )
    assert scores.get("ActionMatches", 0) >= ACTION_MATCH_THRESHOLD, (
        f"ActionMatches {scores.get('ActionMatches'):.2%} below "
        f"{ACTION_MATCH_THRESHOLD:.0%}"
    )
    assert scores.get("SingleQuestion", 0) >= SINGLE_QUESTION_THRESHOLD, (
        "Interviewer stacked multiple questions in a single utterance"
    )
    warmth = scores.get("utterance_warmth")
    if warmth is not None:
        assert warmth >= WARMTH_THRESHOLD, (
            f"Warmth {warmth:.2f} below {WARMTH_THRESHOLD}"
        )
    non_leading = scores.get("non_leading")
    if non_leading is not None:
        assert non_leading >= NON_LEADING_THRESHOLD, (
            f"Non-leading pass rate {non_leading:.2%} below "
            f"{NON_LEADING_THRESHOLD:.0%}"
        )
    relevant = scores.get("response_relevant")
    if relevant is not None:
        assert relevant >= RESPONSE_RELEVANT_THRESHOLD, (
            f"Response relevance pass rate {relevant:.2%} below "
            f"{RESPONSE_RELEVANT_THRESHOLD:.0%}"
        )


def _print_per_case(report) -> None:
    """Print a table: case | expected | actual | scores | utterance, then failures."""
    header = f"{'case':<40} {'exp':>12} {'got':>12} {'AM':>5} {'SQ':>4} {'NL':>4} {'RR':>4} {'W':>5}  utterance"
    print(f"\nTier 1 per-case results:\n{header}\n{'-' * len(header)}")

    def _bool(src, key) -> str:
        item = (src or {}).get(key)
        if item is None:
            return "  - "
        return " ok " if item.value else "FAIL"

    def _num(src, key) -> str:
        item = (src or {}).get(key)
        if item is None:
            return "   -"
        v = item.value
        return f"{v:4.1f}" if isinstance(v, (int, float)) else "   -"

    def _action_label(output) -> str:
        if output is None:
            return "?"
        src = output.probe_source
        return f"{output.action}/{src}" if src else output.action

    for case in report.cases:
        exp = _action_label(case.expected_output)
        got = _action_label(case.output)
        utterance = case.output.utterance if case.output else ""
        if len(utterance) > 50:
            utterance = utterance[:50] + "…"
        am = _num(case.scores, "ActionMatches")
        sq = _bool(case.assertions, "SingleQuestion")
        nl = _bool(case.assertions, "non_leading")
        rr = _bool(case.assertions, "response_relevant")
        w  = _num(case.scores, "utterance_warmth")
        print(f"{(case.name or ''):<40} {exp:>12} {got:>12} {am} {sq} {nl} {rr} {w}  {utterance}")

    for fail in report.failures:
        exp = _action_label(fail.expected_output)
        msg = fail.error_message.splitlines()[0][:60] if fail.error_message else "unknown error"
        print(f"{(fail.name or ''):<40} {exp:>12} {'ERROR':>12}                          {msg}")


def _aggregate(report) -> dict[str, float]:
    """Flatten per-case evaluator results into {name: mean_score}.

    Booleans are treated as 0/1, numeric scores are averaged as-is. Anything
    non-numeric (e.g. string labels) is skipped.
    """
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
