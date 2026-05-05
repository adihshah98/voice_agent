"""Layer A — Replay eval harness (Tier 3, deterministic).

Runs canned full transcripts where respondent lines are fixed. Only the
interviewer runs live (via `run_speech_turn`). Tests multi-turn state effects
that Tier 1 can't cover:
  - covered_subtopics accumulation
  - scripted cursor advancement
  - probe staleness
  - barge-in reconciliation

Runtime: ~30 Haiku calls ≈ 20–30s. CI-safe.

Run with:
    uv run pytest evals/test_replay.py -v -m replay
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import logfire
import pytest
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from sqlalchemy.pool import StaticPool
from sqlmodel import create_engine

load_dotenv()

from voice_agent import state
from voice_agent.turn import run_speech_turn

DATASET_PATH = Path(__file__).parent / "datasets" / "replay_transcripts.yaml"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class ReplayTurn(BaseModel):
    speaker: str
    text: str
    expected_actions: list[str]


class ReplayTranscript(BaseModel):
    name: str
    scripted_questions: list[str]
    turns: list[ReplayTurn]


class ReplayInputs(BaseModel):
    transcript: ReplayTranscript


class ReplayResult(BaseModel):
    transcript_name: str
    turn_results: list[dict[str, Any]]  # per-turn: text, action, expected_actions, pass
    all_actions_valid: bool
    scripted_cursor_advanced: bool
    final_scripted_cursor: int
    scripted_total: int


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


def load_replay_transcripts(path: str | Path = DATASET_PATH) -> list[ReplayTranscript]:
    raw = yaml.safe_load(Path(path).read_text())
    return [ReplayTranscript.model_validate(t) for t in raw["transcripts"]]


def get_dataset_version(path: str | Path = DATASET_PATH) -> str:
    raw = yaml.safe_load(Path(path).read_text())
    return raw.get("version", "unversioned")


# ---------------------------------------------------------------------------
# Replay runner
# ---------------------------------------------------------------------------


async def run_replay(inputs: ReplayInputs) -> ReplayResult:
    """Drive one canned transcript through the live interviewer and record actions."""
    transcript = inputs.transcript
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    state.init_db(engine)
    call_id = f"replay-{transcript.name}"

    with state.session_scope(engine) as s:
        s.add(
            state.Call(
                id=call_id,
                scripted_questions=transcript.scripted_questions,
                status="active",
            )
        )

    # Seed the opening scripted question as the first interviewer turn so the
    # history mirrors what a real call would have seen.
    with state.session_scope(engine) as s:
        opening = state.next_scripted(s, call_id)
        state.mark_scripted_asked(s, call_id)

    opening_text = opening or "Tell me about your experience with this product."
    with state.session_scope(engine) as s:
        s.add(
            state.Turn(
                call_id=call_id,
                turn_number=1,
                speaker="interviewer",
                text=opening_text,
                action="scripted",
            )
        )

    # history is kept as vapi_messages format for run_speech_turn
    history: list[dict[str, str]] = [{"role": "assistant", "content": opening_text}]
    turn_results: list[dict[str, Any]] = []

    for canned_turn in transcript.turns:
        # Feed canned respondent utterance as the next user message
        history.append({"role": "user", "content": canned_turn.text})

        result = await run_speech_turn(engine, call_id, vapi_messages=history)
        action = result["action"]

        turn_results.append({
            "respondent_text": canned_turn.text[:60],
            "action": action,
            "expected_actions": canned_turn.expected_actions,
            "pass": action in canned_turn.expected_actions,
        })

        history.append({"role": "assistant", "content": result["message"]})

        if action == "wrap_up":
            break

    with state.session_scope(engine) as s:
        call = s.get(state.Call, call_id)
        final_cursor = call.scripted_cursor if call else 0

    all_pass = all(t["pass"] for t in turn_results)

    return ReplayResult(
        transcript_name=transcript.name,
        turn_results=turn_results,
        all_actions_valid=all_pass,
        scripted_cursor_advanced=final_cursor > 1,  # opened with cursor=1 already
        final_scripted_cursor=final_cursor,
        scripted_total=len(transcript.scripted_questions),
    )


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------


@dataclass
class AllActionsValid(Evaluator[ReplayInputs, ReplayResult, None]):
    """Every interviewer turn produced one of the expected actions."""

    def evaluate(
        self,
        ctx: EvaluatorContext[ReplayInputs, ReplayResult, None],
    ) -> bool:
        return ctx.output.all_actions_valid


@dataclass
class ScriptedCursorAdvanced(Evaluator[ReplayInputs, ReplayResult, None]):
    """Scripted cursor moved past the opening question (interviewer asked at least one more)."""

    def evaluate(
        self,
        ctx: EvaluatorContext[ReplayInputs, ReplayResult, None],
    ) -> bool:
        return ctx.output.scripted_cursor_advanced


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------


def _build_dataset() -> Dataset[ReplayInputs, ReplayResult, None]:
    transcripts = load_replay_transcripts()
    version = get_dataset_version()
    cases: list[Case[ReplayInputs, ReplayResult, None]] = [
        Case(
            name=t.name,
            inputs=ReplayInputs(transcript=t),
            metadata={"dataset_version": version},
        )
        for t in transcripts
    ]
    return Dataset(
        name="replay_tier3",
        cases=cases,
        evaluators=(
            AllActionsValid(),
            ScriptedCursorAdvanced(),
        ),
    )


# ---------------------------------------------------------------------------
# Pytest entry point
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.replay
async def test_replay_transcripts():
    """Replay eval: canned respondent turns, live interviewer.

    Marked `replay` — fast (~20–30s), CI-safe. Run with:
        uv run pytest evals/test_replay.py -v -m replay
    """
    logfire.configure(service_name="voice-agent-evals", send_to_logfire="if-token-present")

    dataset = _build_dataset()
    report = await dataset.evaluate(
        run_replay,
        max_concurrency=3,
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

    assert scores.get("AllActionsValid", 0) >= 1.0, (
        "Replay transcripts: interviewer produced unexpected actions on at least one turn"
    )
    assert scores.get("ScriptedCursorAdvanced", 0) >= 1.0, (
        "Replay transcripts: scripted cursor never advanced past the opening question"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_results(report) -> None:
    header = f"{'transcript':<28} {'AV':>4} {'SC':>4}  turn actions"
    print(f"\nReplay eval results:\n{header}\n{'-' * len(header)}")

    def _b(src, key) -> str:
        item = (src or {}).get(key)
        if item is None:
            return "  - "
        return " ok " if item.value else "FAIL"

    for case in report.cases:
        av = _b(case.assertions, "AllActionsValid")
        sc = _b(case.assertions, "ScriptedCursorAdvanced")
        actions = ", ".join(t["action"] for t in (case.output.turn_results if case.output else []))
        print(f"{(case.name or ''):<28} {av} {sc}  {actions[:60]}")

        if case.output:
            fails = [t for t in case.output.turn_results if not t["pass"]]
            for f in fails:
                print(f"  MISMATCH turn: got={f['action']} expected={f['expected_actions']}  [{f['respondent_text']}]")

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
