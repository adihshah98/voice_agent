"""Tier 2 — analyst probe quality eval.

Each case seeds an in-memory DB from `AnalystCaseInputs`, runs `run_analyst`,
and grades the returned `AnalysisUpdate` on five dimensions:

    HasProbes            deterministic  — at least 1 probe generated
    NoDuplicateProbes    deterministic  — no two probes share >60% word overlap
    probes_specific      LLMJudge 1–5  — probes reference transcript specifics
    probes_non_leading   LLMJudge bool — all probes are open and neutral
    priority_calibrated  LLMJudge bool — priority-1 only on contradictions/surprises

Pass thresholds (from the plan):
    HasProbes              100%
    NoDuplicateProbes      100%
    probes_specific        >= 4.0 / 5
    probes_non_leading     >= 90%
    priority_calibrated    >= 90%

Run with:
    uv run pytest evals/test_analyst.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest
from dotenv import load_dotenv
from sqlalchemy.pool import StaticPool
from sqlmodel import create_engine

load_dotenv()

from voice_agent import state
from voice_agent.agents.analyst import load_latest_analysis, run_analyst
from evals.cases import AnalystCaseInputs, load_analyst_cases
from evals.evaluators import (
    HasProbes,
    NoDuplicateProbes,
    priority_calibrated_judge,
    probes_non_leading_judge,
    probes_specific_judge,
)
from voice_agent.models import AnalysisUpdate, AnalystDeps
from pydantic_evals import Dataset
from voice_agent.tracing import init_tracing


DATASET_PATH = Path(__file__).parent / "datasets" / "analyst_probes.yaml"

SPECIFICITY_THRESHOLD = 4.0
NON_LEADING_THRESHOLD = 0.90
PRIORITY_THRESHOLD = 0.90


def _seed_engine(inputs: AnalystCaseInputs):
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    state.init_db(engine)
    call_id = "eval-analyst-call"

    with state.session_scope(engine) as session:
        session.add(
            state.Call(
                id=call_id,
                scripted_questions=[],
                status="active",
            )
        )
        for i, turn in enumerate(inputs.transcript, start=1):
            session.add(
                state.Turn(
                    call_id=call_id,
                    turn_number=i,
                    speaker=turn.speaker,
                    text=turn.text,
                    action=turn.action,
                )
            )

    return engine, call_id


async def run_analyst_on_case(inputs: AnalystCaseInputs) -> AnalysisUpdate:
    engine, call_id = _seed_engine(inputs)
    await run_analyst(AnalystDeps(call_id=call_id, engine=engine))
    return load_latest_analysis(engine, call_id)


@pytest.mark.asyncio
async def test_tier2_analyst_probe_quality():
    init_tracing(service_name="voice-agent-evals", send_to_logfire=False)

    dataset: Dataset[AnalystCaseInputs, AnalysisUpdate, None] = Dataset(
        name="analyst_tier2",
        cases=load_analyst_cases(DATASET_PATH),
        evaluators=(
            HasProbes(),
            NoDuplicateProbes(),
            probes_specific_judge(),
            probes_non_leading_judge(),
            priority_calibrated_judge(),
        ),
    )

    report = await dataset.evaluate(
        run_analyst_on_case,
        max_concurrency=2,
        progress=False,
    )

    _print_per_case(report)
    scores = _aggregate(report)
    print("\nAggregate:")
    for name, value in sorted(scores.items()):
        print(f"  {name:28s} {value:.3f}")

    assert not report.failures, (
        f"{len(report.failures)} case(s) errored: "
        + ", ".join(f.name or "?" for f in report.failures)
    )
    assert scores.get("HasProbes", 0) >= 1.0, "Some cases produced zero probes"
    assert scores.get("NoDuplicateProbes", 0) >= 1.0, "Duplicate probes detected"

    specificity = scores.get("probes_specific")
    if specificity is not None:
        assert specificity >= SPECIFICITY_THRESHOLD, (
            f"Probe specificity {specificity:.2f} below {SPECIFICITY_THRESHOLD}"
        )

    non_leading = scores.get("probes_non_leading")
    if non_leading is not None:
        assert non_leading >= NON_LEADING_THRESHOLD, (
            f"Non-leading pass rate {non_leading:.2%} below {NON_LEADING_THRESHOLD:.0%}"
        )

    priority = scores.get("priority_calibrated")
    if priority is not None:
        assert priority >= PRIORITY_THRESHOLD, (
            f"Priority calibration {priority:.2%} below {PRIORITY_THRESHOLD:.0%}"
        )


def _print_per_case(report) -> None:
    header = (
        f"{'case':<35} {'HP':>4} {'ND':>4} {'Sp':>5} {'NL':>4} {'PC':>4}  "
        "probes (priority: question[:60])"
    )
    print(f"\nTier 2 per-case results:\n{header}\n{'-' * len(header)}")

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
        hp = _bool(case.assertions, "HasProbes")
        nd = _bool(case.assertions, "NoDuplicateProbes")
        sp = _num(case.scores, "probes_specific")
        nl = _bool(case.assertions, "probes_non_leading")
        pc = _bool(case.assertions, "priority_calibrated")
        print(f"{(case.name or ''):<35} {hp} {nd} {sp} {nl} {pc}")
        if case.output:
            for p in case.output.new_probes:
                q = p.question[:60]
                print(f"  [{p.priority}] {q}")

    for fail in report.failures:
        msg = (
            fail.error_message.splitlines()[0][:60]
            if fail.error_message
            else "unknown error"
        )
        print(f"{(fail.name or ''):<35} {'ERROR':>40}  {msg}")


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
