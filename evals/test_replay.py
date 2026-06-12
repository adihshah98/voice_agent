"""Layer A — Replay eval harness (Tier 3, deterministic).

Runs canned full transcripts where respondent lines are fixed. Only the
interviewer runs live (via `run_speech_turn`). Tests multi-turn state effects
that Tier 1 & 2 can't cover:
  - covered_subtopics accumulation
  - scripted cursor advancement
  - probe staleness
  - skip_scripted on organic coverage
  - non-happy-path: silence and vague answer handling
  - loop guard: no repeated "Still there?" or re-asked scripted topics

Runtime: ~60 Haiku calls ≈ 40–60s. CI-safe.

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
from voice_agent.agents.analyst import run_analyst
from voice_agent.models import AnalystDeps
from voice_agent.turn import run_speech_turn

DATASET_PATH = Path(__file__).parent / "datasets" / "replay_transcripts.yaml"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class ReplayTurn(BaseModel):
    speaker: str
    text: str
    expected_actions: list[str]


class ProbeSeed(BaseModel):
    question: str
    priority: int = 2
    rationale: str = ""
    generated_after_turn: int = 0


class ReplayTranscript(BaseModel):
    name: str
    scripted_questions: list[str]
    turns: list[ReplayTurn]
    probe_seeds: list[ProbeSeed] = []


class ReplayInputs(BaseModel):
    transcript: ReplayTranscript


class ReplayResult(BaseModel):
    transcript_name: str
    turn_results: list[dict[str, Any]]  # per-turn: text, action, utterance, expected_actions, pass
    all_actions_valid: bool
    scripted_cursor_advanced: bool
    final_scripted_cursor: int
    scripted_total: int
    wrap_up_after_all_scripted: bool
    still_there_repeated: bool
    # Probe-turn utterances where action=probe (for BridgingPhrasePresent)
    probe_utterances: list[str] = []
    # Index of the turn wrap_up fired (None if never fired)
    wrap_up_turn_index: int | None = None


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
        for ps in transcript.probe_seeds:
            s.add(
                state.Probe(
                    call_id=call_id,
                    question=ps.question,
                    priority=ps.priority,
                    rationale=ps.rationale,
                    generated_after_turn=ps.generated_after_turn,
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
    probe_utterances: list[str] = []
    wrap_up_cursor: int | None = None
    wrap_up_turn_index: int | None = None

    for turn_idx, canned_turn in enumerate(transcript.turns):
        history.append({"role": "user", "content": canned_turn.text})

        result = await run_speech_turn(engine, call_id, vapi_messages=history)
        action = result["action"]
        utterance = result["message"]

        if action == "wrap_up" and wrap_up_cursor is None:
            wrap_up_turn_index = turn_idx
            with state.session_scope(engine) as s:
                call = s.get(state.Call, call_id)
                wrap_up_cursor = call.scripted_cursor if call else 0

        if action == "probe":
            probe_utterances.append(utterance)

        turn_results.append({
            "respondent_text": canned_turn.text[:60],
            "action": action,
            "utterance": utterance,
            "expected_actions": canned_turn.expected_actions,
            "pass": action in canned_turn.expected_actions,
        })

        history.append({"role": "assistant", "content": utterance})

        if result.get("should_run_analyst"):
            await run_analyst(AnalystDeps(call_id=call_id, engine=engine))

        if action == "wrap_up":
            break

    with state.session_scope(engine) as s:
        call = s.get(state.Call, call_id)
        final_cursor = call.scripted_cursor if call else 0

    all_pass = all(t["pass"] for t in turn_results)

    # Wrap-up only after all scripted covered
    if wrap_up_cursor is not None:
        wrap_up_after_all_scripted = wrap_up_cursor >= len(transcript.scripted_questions)
    else:
        # No wrap_up fired — not a violation of this specific check
        wrap_up_after_all_scripted = True

    # Loop guard: detect runaway re-engagement. The model may re-prompt a stalling
    # respondent ("Take your time." → "Still there?"), but a run of 4+ consecutive
    # clarify utterances means it's stuck re-engaging instead of advancing.
    actions = [t["action"] for t in turn_results]
    max_consec = 0
    cur = 0
    for a in actions:
        if a == "clarify":
            cur += 1
            max_consec = max(max_consec, cur)
        else:
            cur = 0
    still_there_repeated = max_consec > 3

    return ReplayResult(
        transcript_name=transcript.name,
        turn_results=turn_results,
        all_actions_valid=all_pass,
        scripted_cursor_advanced=final_cursor > 1,  # opened with cursor=1 already
        final_scripted_cursor=final_cursor,
        scripted_total=len(transcript.scripted_questions),
        wrap_up_after_all_scripted=wrap_up_after_all_scripted,
        still_there_repeated=still_there_repeated,
        probe_utterances=probe_utterances,
        wrap_up_turn_index=wrap_up_turn_index,
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


@dataclass
class WrapUpOnlyAfterAllScripted(Evaluator[ReplayInputs, ReplayResult, None]):
    """When wrap_up fires, all scripted questions must have been asked (or skipped).

    A call that wraps up at turn 3 with 3 scripted questions unasked is a failure —
    the interviewer gave up too early. Returns True (N/A pass) when no wrap_up fired
    within the transcript's turn limit.

    Skipped (N/A pass) for disengaged_early_exit, where early wrap_up is the
    correct behaviour — DisengagementWrapsUp covers that case instead.
    """

    _EXEMPT = {"disengaged_early_exit"}

    def evaluate(
        self,
        ctx: EvaluatorContext[ReplayInputs, ReplayResult, None],
    ) -> bool:
        if ctx.inputs.transcript.name in self._EXEMPT:
            return True
        return ctx.output.wrap_up_after_all_scripted


@dataclass
class NoStillThereLoop(Evaluator[ReplayInputs, ReplayResult, None]):
    """Interviewer must not re-engage more than 3 times consecutively.

    Re-prompting a stalling respondent ("Take your time." → "Still there?") is fine,
    but a run of 4+ consecutive clarify turns means the interviewer is stuck
    re-engaging instead of advancing or wrapping up.
    """

    def evaluate(
        self,
        ctx: EvaluatorContext[ReplayInputs, ReplayResult, None],
    ) -> bool:
        return not ctx.output.still_there_repeated


@dataclass
class BridgingPhrasePresent(Evaluator[ReplayInputs, ReplayResult, None]):
    """For transcripts with probe_seeds, at least one probe utterance must
    contain a bridging phrase anchoring back to an earlier respondent statement.

    The interviewer prompt instructs: when turns_ago ≥ 3, bridge with
    'Earlier you mentioned X...' before asking. This checks the stale_probe_bridging
    transcript where the probe is seeded at turn 1 and used at turn 5+.

    Returns True (N/A pass) for transcripts with no probe_seeds.
    """

    _BRIDGE_PHRASES = ("earlier", "you mentioned", "going back", "you said", "you brought up")

    def evaluate(
        self,
        ctx: EvaluatorContext[ReplayInputs, ReplayResult, None],
    ) -> bool:
        if ctx.inputs.transcript.name != "stale_probe_bridging":
            return True
        if not ctx.output.probe_utterances:
            return False
        return any(
            any(phrase in u.lower() for phrase in self._BRIDGE_PHRASES)
            for u in ctx.output.probe_utterances
        )


@dataclass
class DisengagementWrapsUp(Evaluator[ReplayInputs, ReplayResult, None]):
    """For the disengaged_early_exit transcript: wrap_up must fire within the
    5-turn transcript even though scripted questions remain unanswered.

    This evaluator intentionally passes when wrap_up fires *before* all
    scripted are covered — that's the correct behaviour for a disengaged
    respondent. It conflicts with WrapUpOnlyAfterAllScripted, so that
    evaluator is skipped for this transcript in _build_dataset().

    Returns True (N/A pass) for all other transcripts.
    """

    def evaluate(
        self,
        ctx: EvaluatorContext[ReplayInputs, ReplayResult, None],
    ) -> bool:
        if ctx.inputs.transcript.name != "disengaged_early_exit":
            return True
        return ctx.output.wrap_up_turn_index is not None


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
            WrapUpOnlyAfterAllScripted(),
            NoStillThereLoop(),
            BridgingPhrasePresent(),
            DisengagementWrapsUp(),
        ),
    )


# ---------------------------------------------------------------------------
# Pytest entry point
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.replay
async def test_replay_transcripts():
    """Replay eval: canned respondent turns, live interviewer.

    Marked `replay` — fast (~40–60s), CI-safe. Run with:
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

    assert not report.failures, (
        f"{len(report.failures)} transcript(s) errored: "
        + ", ".join(f.name or "?" for f in report.failures)
    )
    assert scores.get("AllActionsValid", 0) >= 1.0, (
        "Replay transcripts: interviewer produced unexpected actions on at least one turn"
    )
    assert scores.get("ScriptedCursorAdvanced", 0) >= 1.0, (
        "Replay transcripts: scripted cursor never advanced past the opening question"
    )
    assert scores.get("WrapUpOnlyAfterAllScripted", 0) >= 1.0, (
        "Replay transcripts: wrap_up fired before all scripted questions were covered"
    )
    assert scores.get("NoStillThereLoop", 0) >= 1.0, (
        "Replay transcripts: interviewer repeated 'Still there?' on consecutive turns"
    )
    assert scores.get("BridgingPhrasePresent", 1.0) >= 1.0, (
        "stale_probe_bridging: interviewer did not bridge back with 'Earlier you mentioned…'"
    )
    assert scores.get("DisengagementWrapsUp", 1.0) >= 1.0, (
        "disengaged_early_exit: interviewer kept asking scripted questions despite disengagement"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_results(report) -> None:
    header = f"{'transcript':<30} {'AV':>4} {'SC':>4} {'WU':>4} {'SL':>4} {'BP':>4} {'DE':>4}  turn actions"
    print(f"\nReplay eval results:\n{header}\n{'-' * len(header)}")

    def _b(src, key) -> str:
        item = (src or {}).get(key)
        if item is None:
            return "  - "
        return " ok " if item.value else "FAIL"

    for case in report.cases:
        av = _b(case.assertions, "AllActionsValid")
        sc = _b(case.assertions, "ScriptedCursorAdvanced")
        wu = _b(case.assertions, "WrapUpOnlyAfterAllScripted")
        sl = _b(case.assertions, "NoStillThereLoop")
        bp = _b(case.assertions, "BridgingPhrasePresent")
        de = _b(case.assertions, "DisengagementWrapsUp")
        actions = ", ".join(t["action"] for t in (case.output.turn_results if case.output else []))
        print(f"{(case.name or ''):<30} {av} {sc} {wu} {sl} {bp} {de}  {actions[:60]}")

        if case.output:
            fails = [t for t in case.output.turn_results if not t["pass"]]
            for f in fails:
                print(f"  MISMATCH turn: got={f['action']} expected={f['expected_actions']}  [{f['respondent_text']}]")
                print(f"    utterance: {f['utterance'][:100]}")

    for fail in report.failures:
        msg = (fail.error_message or "unknown error").splitlines()[0][:70]
        print(f"{(fail.name or ''):<30} ERROR  {msg}")


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
