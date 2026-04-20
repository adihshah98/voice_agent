"""Synthesis agent — final post-call report.

Triggered on call-ended; reads all turns + most-recent analyst snapshot,
produces a structured report, and upserts a SynthesisReport row.

Use run_synthesis_safely from server.py — exceptions are caught + logged.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import logfire
from dotenv import load_dotenv
from pydantic_ai import Agent
from sqlmodel import Session, select

load_dotenv()

from voice_agent import state
from voice_agent.config import SYNTHESIS_MODEL
from voice_agent.models import ReportOutput

SYNTHESIS_PROMPT = """\
You are a senior qualitative researcher writing a post-call synthesis report.

You will receive the full interview transcript and any analyst notes produced
during the call (themes, contradictions, surprises).

Your job is to produce a concise, actionable report:

SUMMARY — 2-3 sentence overview of what was learned.
THEMES — recurring ideas, each paired with direct quotes that support it.
CONTRADICTIONS — specific places where the respondent contradicted themselves.
KEY_QUOTES — 3-5 verbatim quotes that best capture the respondent's perspective.
FOLLOW_UP_QUESTIONS — 3-5 questions worth exploring in a follow-up call.

Be specific and evidence-based. Cite the transcript directly.
Do not hallucinate or infer beyond what was said.
"""


@dataclass
class SynthesisDeps:
    call_id: str
    session: Session


def _build_prompt(session: Session, call_id: str) -> str:
    turns = state.recent_turns(session, call_id, n=200)
    lines = [f"{t.speaker.upper()} [{t.turn_number}]: {t.text}" for t in turns]
    transcript = "\n".join(lines) if lines else "(no turns recorded)"

    latest_snap = session.exec(
        select(state.AnalystSnapshot)
        .where(state.AnalystSnapshot.call_id == call_id)
        .order_by(state.AnalystSnapshot.id.desc())
        .limit(1)
    ).first()

    prompt = f"TRANSCRIPT:\n{transcript}"
    if latest_snap:
        notes: list[str] = []
        if latest_snap.themes:
            notes.append("ANALYST THEMES:\n" + "\n".join(f"- {t}" for t in latest_snap.themes))
        if latest_snap.contradictions:
            notes.append("ANALYST CONTRADICTIONS:\n" + "\n".join(f"- {c}" for c in latest_snap.contradictions))
        if latest_snap.surprises:
            notes.append("ANALYST SURPRISES:\n" + "\n".join(f"- {s}" for s in latest_snap.surprises))
        if notes:
            prompt += "\n\n" + "\n\n".join(notes)

    return prompt


synthesis_agent = Agent(
    SYNTHESIS_MODEL,
    deps_type=SynthesisDeps,
    output_type=ReportOutput,
    system_prompt=SYNTHESIS_PROMPT,
    instrument=True,
)


async def run_synthesis(deps: SynthesisDeps) -> ReportOutput:
    t0 = time.perf_counter()
    prompt = _build_prompt(deps.session, deps.call_id)
    result = await synthesis_agent.run(prompt, deps=deps)
    report: ReportOutput = result.output
    latency_ms = int((time.perf_counter() - t0) * 1000)

    existing = deps.session.exec(
        select(state.SynthesisReport).where(state.SynthesisReport.call_id == deps.call_id)
    ).first()

    row_data = dict(
        summary=report.summary,
        themes=[t.model_dump() for t in report.themes],
        contradictions=report.contradictions,
        key_quotes=report.key_quotes,
        follow_up_questions=report.follow_up_questions,
    )
    if existing:
        for k, v in row_data.items():
            setattr(existing, k, v)
        deps.session.add(existing)
    else:
        deps.session.add(state.SynthesisReport(call_id=deps.call_id, **row_data))

    deps.session.commit()

    logfire.info(
        "synthesis_complete",
        call_id=deps.call_id,
        themes_count=len(report.themes),
        latency_ms=latency_ms,
    )
    return report


async def run_synthesis_safely(deps: SynthesisDeps) -> ReportOutput | None:
    try:
        return await run_synthesis(deps)
    except Exception:
        logfire.exception("synthesis_error", call_id=deps.call_id)
        return None
