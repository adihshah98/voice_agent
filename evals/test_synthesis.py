"""Layer C — Synthesis quality eval driven by replay transcripts.

Runs the synthesis agent against the 3 canned replay transcripts. Because the
transcripts are fixed, synthesis results are reproducible and this eval is
separable from trajectory evals. No live simulation required.

Evaluators:
    SynthesisNotEmpty     deterministic — summary is non-empty
    HasKeyFields          deterministic — pmf_score and key_quotes present
    ReportQuality         LLMJudge 1–5 — accuracy and usefulness vs transcript

Run with:
    uv run pytest evals/test_synthesis.py -v
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import logfire
import pytest
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, LLMJudge
from sqlalchemy.pool import StaticPool
from sqlmodel import create_engine

load_dotenv()

from voice_agent import state
from voice_agent.agents.synthesis import SynthesisDeps, run_synthesis_safely
from evals.test_replay import ReplayTranscript, load_replay_transcripts, get_dataset_version

JUDGE_MODEL = "anthropic:claude-opus-4-6"
DATASET_PATH = Path(__file__).parent / "datasets" / "replay_transcripts.yaml"
REPORT_QUALITY_THRESHOLD = 3.0


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class SynthesisInputs(BaseModel):
    transcript: ReplayTranscript


class SynthesisResult(BaseModel):
    transcript_name: str
    summary: str
    pmf_score: int | None
    key_quotes: list[str]
    transcript_text: str  # included so LLMJudge can compare


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------


async def run_synthesis_on_transcript(inputs: SynthesisInputs) -> SynthesisResult:
    """Seed in-memory DB from canned transcript, run synthesis, return result."""
    transcript = inputs.transcript
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    state.init_db(engine)
    call_id = f"synth-{transcript.name}"

    with state.session_scope(engine) as s:
        s.add(
            state.Call(
                id=call_id,
                scripted_questions=transcript.scripted_questions,
                status="ended",
            )
        )

    # Seed turns from the fixed transcript.
    # Opening interviewer turn (first scripted question) is turn 1.
    with state.session_scope(engine) as s:
        opening = transcript.scripted_questions[0] if transcript.scripted_questions else "Tell me about your experience."
        s.add(
            state.Turn(
                call_id=call_id,
                turn_number=1,
                speaker="interviewer",
                text=opening,
                action="scripted",
            )
        )
        for i, turn in enumerate(transcript.turns, start=2):
            s.add(
                state.Turn(
                    call_id=call_id,
                    turn_number=i,
                    speaker="respondent" if turn.speaker == "respondent" else "interviewer",
                    text=turn.text,
                )
            )

    transcript_text = f"INTERVIEWER [1]: {opening}\n" + "\n".join(
        f"{t.speaker.upper()} [{i+2}]: {t.text}"
        for i, t in enumerate(transcript.turns)
    )

    report = None
    with state.session_scope(engine) as s:
        deps = SynthesisDeps(call_id=call_id, session=s)
        report = await run_synthesis_safely(deps)

    return SynthesisResult(
        transcript_name=transcript.name,
        summary=report.summary if report else "",
        pmf_score=report.pmf_score if report else None,
        key_quotes=report.key_quotes if report else [],
        transcript_text=transcript_text,
    )


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------


@dataclass
class SynthesisNotEmpty(Evaluator[SynthesisInputs, SynthesisResult, None]):
    """Summary must be non-empty."""

    def evaluate(
        self,
        ctx: EvaluatorContext[SynthesisInputs, SynthesisResult, None],
    ) -> bool:
        return bool(ctx.output.summary.strip())


@dataclass
class HasKeyFields(Evaluator[SynthesisInputs, SynthesisResult, None]):
    """PMF score must be present (1–5) and at least one key quote returned."""

    def evaluate(
        self,
        ctx: EvaluatorContext[SynthesisInputs, SynthesisResult, None],
    ) -> bool:
        has_score = ctx.output.pmf_score is not None and 1 <= ctx.output.pmf_score <= 5
        has_quotes = len(ctx.output.key_quotes) >= 1
        return has_score and has_quotes


def report_quality_judge() -> LLMJudge:
    """LLMJudge: is the synthesis summary accurate and useful vs the canned transcript?"""
    return LLMJudge(
        rubric=(
            "You are grading a POST-CALL SYNTHESIS SUMMARY produced by an AI "
            "market-research agent after a phone interview.\n\n"
            "The output you are grading contains:\n"
            "  `transcript_text` — the full conversation\n"
            "  `summary` — the post-call summary to evaluate\n\n"
            "Score 1–5 on REPORT QUALITY:\n"
            "  5 = accurately reflects the transcript; cites specific details; "
            "no hallucinations; identifies key themes; actionable\n"
            "  4 = mostly accurate; minor omissions or slight over-generalisation\n"
            "  3 = captures the gist but misses important details or makes vague "
            "claims that aren't directly supported\n"
            "  2 = significant omissions or minor inaccuracies relative to the "
            "transcript\n"
            "  1 = does not reflect the transcript, or contains hallucinations, "
            "or summary is empty\n\n"
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


def _build_dataset() -> Dataset[SynthesisInputs, SynthesisResult, None]:
    transcripts = load_replay_transcripts()
    version = get_dataset_version()
    cases: list[Case[SynthesisInputs, SynthesisResult, None]] = [
        Case(
            name=t.name,
            inputs=SynthesisInputs(transcript=t),
            metadata={"dataset_version": version},
        )
        for t in transcripts
    ]
    return Dataset(
        name="synthesis_tier3",
        cases=cases,
        evaluators=(
            SynthesisNotEmpty(),
            HasKeyFields(),
            report_quality_judge(),
        ),
    )


# ---------------------------------------------------------------------------
# Pytest entry point
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesis_quality():
    """Synthesis quality eval on 3 fixed replay transcripts.

    Reproducible: transcript content is fixed so synthesis output is
    deterministic across runs. Run with:
        uv run pytest evals/test_synthesis.py -v
    """
    logfire.configure(service_name="voice-agent-evals", send_to_logfire="if-token-present")

    dataset = _build_dataset()
    report = await dataset.evaluate(
        run_synthesis_on_transcript,
        max_concurrency=2,
        progress=False,
    )

    _print_results(report)

    assert not report.failures, (
        f"{len(report.failures)} transcript(s) errored: "
        + ", ".join(f.name or "?" for f in report.failures)
    )

    scores = _aggregate(report)
    print("\nAggregate scores:")
    for name, value in sorted(scores.items()):
        print(f"  {name:<28} {value:.3f}")

    assert scores.get("SynthesisNotEmpty", 0) >= 1.0, (
        "Synthesis returned empty summaries"
    )
    assert scores.get("HasKeyFields", 0) >= 1.0, (
        "Synthesis missing pmf_score or key_quotes on at least one transcript"
    )
    quality = scores.get("report_quality")
    if quality is not None:
        assert quality >= REPORT_QUALITY_THRESHOLD, (
            f"Synthesis report quality {quality:.2f} below {REPORT_QUALITY_THRESHOLD}/5 — "
            "reports are not accurately reflecting transcripts"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_results(report) -> None:
    header = f"{'transcript':<28} {'NE':>4} {'KF':>4} {'RQ':>5}  pmf  summary"
    print(f"\nSynthesis eval results:\n{header}\n{'-' * len(header)}")

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
        ne = _b(a, "SynthesisNotEmpty")
        kf = _b(a, "HasKeyFields")
        rq = _n(sc, "report_quality")
        pmf = str(case.output.pmf_score) if case.output else "-"
        summary = (case.output.summary[:55] + "…") if case.output and len(case.output.summary) > 55 else (case.output.summary if case.output else "")
        print(f"{(case.name or ''):<28} {ne} {kf} {rq:>5}  {pmf:<4} {summary}")

    for fail in report.failures:
        msg = (fail.error_message or "unknown error").splitlines()[0][:70]
        print(f"{(fail.name or ''):<28} ERROR  {msg}")


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
