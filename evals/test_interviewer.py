"""Tier 1 — single-turn decision eval for the interviewer agent.

Each case seeds an in-memory SQLite DB from `InterviewerCaseInputs`, runs
`interviewer.run(...)` once, and scores the returned `InterviewerOutput`.

Requires ANTHROPIC_API_KEY in .env. Run with:

    uv run pytest evals/test_interviewer.py -v

Pass criteria (from the plan):
    - ActionMatches >= 90%
    - UtteranceWarmth average >= 4 / 5
    - SingleQuestion 100% (deterministic)
    - NoLeadingQuestions 100% pass
"""

from __future__ import annotations

from pathlib import Path

import pytest
from dotenv import load_dotenv
from sqlalchemy.pool import StaticPool
from sqlmodel import create_engine

load_dotenv()

import state
from evals.cases import InterviewerCaseInputs, load_cases
from evals.evaluators import (
    ActionMatches,
    SingleQuestion,
    no_leading_questions_judge,
    utterance_warmth_judge,
)
from interviewer import run_interviewer
from models import InterviewerDeps, InterviewerOutput
from pydantic_evals import Dataset
from tracing import init_tracing


DATASET_PATH = Path(__file__).parent / "datasets" / "interviewer_turns.yaml"
ACTION_MATCH_THRESHOLD = 0.90
WARMTH_THRESHOLD = 4.0
NON_LEADING_THRESHOLD = 0.90
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
    engine, call_id, prior_count = _seed_engine(inputs)
    with state.session_scope(engine) as session:
        session.add(
            state.Turn(
                call_id=call_id,
                turn_number=prior_count + 1,
                speaker="respondent",
                text=inputs.last_respondent,
            )
        )
        session.flush()
        deps = InterviewerDeps(
            call_id=call_id,
            session=session,
            turn_number=prior_count + 2,
        )
        return await run_interviewer(deps, inputs.last_respondent)


@pytest.mark.asyncio
async def test_tier1_interviewer_decisions():
    init_tracing(service_name="voice-agent-evals", send_to_logfire=False)

    dataset: Dataset[InterviewerCaseInputs, InterviewerOutput, None] = Dataset(
        name="interviewer_tier1",
        cases=load_cases(DATASET_PATH),
        evaluators=(
            ActionMatches(),
            SingleQuestion(),
            utterance_warmth_judge(),
            no_leading_questions_judge(),
        ),
    )

    report = await dataset.evaluate(
        run_interviewer_on_case,
        max_concurrency=2,
        progress=False,
    )

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


def _print_per_case(report) -> None:
    """Print a table: case | expected | actual | scores | utterance, then failures."""
    header = f"{'case':<40} {'exp':>8} {'got':>8} {'AM':>4} {'SQ':>4} {'NL':>4} {'W':>5}  utterance"
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

    for case in report.cases:
        exp = case.expected_output.action if case.expected_output else "?"
        got = case.output.action if case.output else "?"
        utterance = case.output.utterance if case.output else ""
        if len(utterance) > 55:
            utterance = utterance[:55] + "…"
        am = _bool(case.assertions, "ActionMatches")
        sq = _bool(case.assertions, "SingleQuestion")
        nl = _bool(case.assertions, "non_leading")
        w  = _num(case.scores, "utterance_warmth")
        print(f"{(case.name or ''):<40} {exp:>8} {got:>8} {am} {sq} {nl} {w}  {utterance}")

    for fail in report.failures:
        exp = fail.expected_output.action if fail.expected_output else "?"
        msg = fail.error_message.splitlines()[0][:60] if fail.error_message else "unknown error"
        print(f"{(fail.name or ''):<40} {exp:>8} {'ERROR':>8}                        {msg}")


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
