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
You are a senior investment research analyst writing a post-call memo for an investor
doing due diligence on a B2B SaaS or AI product. You will receive the full interview
transcript and analyst notes (themes, contradictions, surprises, investor signals).

Produce a concise, actionable investor memo with these sections:

SUMMARY — 3-5 bullets sentences answering: "What did we learn that's investment-relevant?"
  Focus on PMF strength, competitive position, and revenue signals.

THEMES — recurring ideas, each paired with direct transcript quotes.

CONTRADICTIONS — specific self-contradictions in the respondent's account.
  Flag gaps between stated satisfaction and actual usage behavior.

KEY_QUOTES — 3-5 verbatim quotes that best capture investment-relevant perspective.

FOLLOW_UP_QUESTIONS — 3-5 questions worth exploring in a follow-up call.

PMF_SCORE — integer 1-5 with explicit rationale:
  5 = daily habit, "can't live without it", organic word-of-mouth, unprompted advocacy
  4 = strong regular use, would miss it, considering expansion to more teams
  3 = satisfied user, uses it regularly, neutral on switching, hasn't advocated for it
  2 = sporadic or limited use, wouldn't miss it much, evaluating alternatives
  1 = minimal use, considering switching, no clear demonstrated value

PMF_SCORE_RATIONALE — one sentence explaining the score, citing transcript evidence.

COMPETITIVE_SIGNALS — why this product was chosen over alternatives, what would make
  the respondent switch, named competitors and how they were characterized.

REVENUE_SIGNALS — budget ownership, contract structure (annual/monthly), seat count
  and utilization, expansion signals or blockers, pricing sensitivity.

AI_ADOPTION_SIGNALS — trust level in AI outputs, verification habits, ROI clarity,
  adoption barriers (security, compliance, hallucinations), workflow integration depth.

RED_FLAGS — anything suggesting churn risk, weak adoption, or poor PMF:
  low seat utilization, implementation stalls, IT/security issues, evaluating alternatives.

INVESTMENT_THESIS_BULLETS — 2-4 crisp bullets stating what this customer proves or disproves
  about the investment thesis. Start each with "SUPPORTS:" or "QUESTIONS:".

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
        if latest_snap.investor_signals:
            notes.append("ANALYST INVESTOR SIGNALS:\n" + "\n".join(f"- {s}" for s in latest_snap.investor_signals))
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
        pmf_score=report.pmf_score,
        pmf_score_rationale=report.pmf_score_rationale,
        competitive_signals=report.competitive_signals,
        revenue_signals=report.revenue_signals,
        ai_adoption_signals=report.ai_adoption_signals,
        red_flags=report.red_flags,
        investment_thesis_bullets=report.investment_thesis_bullets,
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
