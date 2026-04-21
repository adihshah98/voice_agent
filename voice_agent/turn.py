"""Shared speech-turn pipeline — used by the Vapi LLM endpoint and local play.py."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from sqlalchemy import func
from sqlmodel import Session, select

from voice_agent import state
from voice_agent.agents.analyst import run_analyst_safely
from voice_agent.agents.interviewer import run_interviewer_with_timeout
from voice_agent.models import AnalystDeps, InterviewerDeps
from voice_agent.tracing import agent_span, log_interviewer_decision, turn_span


def _next_turn_number(session: Session, call_id: str) -> int:
    result = session.exec(
        select(func.max(state.Turn.turn_number)).where(state.Turn.call_id == call_id)
    ).one()
    return (result or 0) + 1


async def _analyst_task(engine, call_id: str) -> None:
    session = Session(engine)
    try:
        with agent_span("analyst", call_id):
            deps = AnalystDeps(call_id=call_id, session=session)
            await run_analyst_safely(deps)
    finally:
        session.close()


async def run_speech_turn(engine, call_id: str, respondent_text: str) -> dict[str, Any]:
    """Process one respondent turn through the interviewer agent.

    Returns {message, action, reasoning}. If respondent_text is empty (opening
    turn), no respondent Turn row is written.
    """
    with state.session_scope(engine) as session:
        turn_number = _next_turn_number(session, call_id)

        deps = InterviewerDeps(
            call_id=call_id,
            session=session,
            turn_number=turn_number,
        )

        with turn_span(call_id, turn_number, respondent_text=respondent_text or None):
            t0 = time.perf_counter()
            with agent_span("interviewer", call_id, turn_number=turn_number):
                reply = await run_interviewer_with_timeout(deps, respondent_text)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            log_interviewer_decision(
                call_id=call_id,
                turn_number=turn_number,
                action=reply.action,
                utterance=reply.utterance,
                reasoning=reply.reasoning,
                latency_ms=latency_ms,
            )

        # Fire analyst after a scripted question is answered — that's when a new
        # topic has been fully explored and synthesis has something worth doing.
        if respondent_text and reply.action == "scripted":
            asyncio.create_task(_analyst_task(engine, call_id))

        turns: list[state.Turn] = []
        interviewer_turn_number = turn_number
        if respondent_text:
            turns.append(state.Turn(
                call_id=call_id,
                turn_number=turn_number,
                speaker="respondent",
                text=respondent_text,
            ))
            interviewer_turn_number = turn_number + 1

        turns.append(state.Turn(
            call_id=call_id,
            turn_number=interviewer_turn_number,
            speaker="interviewer",
            text=reply.utterance,
            action=reply.action,
            reasoning=reply.reasoning,
            latency_ms=latency_ms,
        ))
        session.add_all(turns)

    return {
        "message": reply.utterance,
        "action": reply.action,
        "reasoning": reply.reasoning,
    }
