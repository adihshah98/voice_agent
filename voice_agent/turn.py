"""Shared speech-turn pipeline — used by the Vapi LLM endpoint and local play.py."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from sqlmodel import Session

from voice_agent import state
from voice_agent.agents.analyst import run_analyst_safely
from voice_agent.agents.interviewer import run_interviewer_with_timeout
from voice_agent.models import AnalystDeps, InterviewerDeps
from voice_agent.tracing import agent_span


async def _analyst_task(engine, call_id: str) -> None:
    import logfire
    session = Session(engine)
    try:
        with agent_span("analyst", call_id):
            deps = AnalystDeps(call_id=call_id, session=session)
            await run_analyst_safely(deps)
    except Exception:
        logfire.exception("analyst_task_error", call_id=call_id)
    finally:
        session.close()


async def run_speech_turn(engine, call_id: str, respondent_text: str) -> dict[str, Any]:
    """Process one respondent turn through the interviewer agent.

    Returns {message, action, reasoning}. If respondent_text is empty (opening
    turn), no respondent Turn row is written.

    Two short sessions bracket the LLM call so SQLite is never write-locked
    during the 5 s interviewer budget. Phase 1 increments the turn counter and
    reads context; phase 2 persists turns and side effects.
    """
    import logfire

    # --- Phase 1a: commit turn-counter increment immediately so the write lock
    #     is released before the LLM call (which can take up to 5 s). ---
    with state.session_scope(engine) as session:
        turn_number = state.next_turn_number(session, call_id)
    logfire.info("turn_started", call_id=call_id, turn_number=turn_number)

    # --- Phase 1b: read session for context fetch + LLM call (no writes) ---
    with state.session_scope(engine) as session:
        deps = InterviewerDeps(
            call_id=call_id,
            session=session,
            turn_number=turn_number,
        )

        with agent_span(
            "interviewer",
            call_id,
            turn_number=turn_number,
            respondent_text=respondent_text or None,
        ) as span:
            t0 = time.perf_counter()
            reply = await run_interviewer_with_timeout(deps, respondent_text)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            span.set_attribute("action", reply.action)
            span.set_attribute("utterance", reply.utterance)
            span.set_attribute("reasoning", reply.reasoning)
            span.set_attribute("latency_ms", latency_ms)

    # --- Phase 2: write session (persist turns + interviewer side effects) ---
    with logfire.span(
        "turn_persist",
        call_id=call_id,
        turn_number=turn_number,
        action=reply.action,
        probe_id_used=reply.probe_id_used,
    ):
        with state.session_scope(engine) as session:
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

            if reply.action in ("scripted", "skip_scripted"):
                state.mark_scripted_asked(session, call_id)
            if reply.probe_id_used is not None:
                state.mark_probe_asked(session, reply.probe_id_used)

    # Fire analyst after scripted turn — topic fully explored, analyst has something to do.
    if respondent_text and reply.action == "scripted":
        asyncio.create_task(_analyst_task(engine, call_id))

    return {
        "message": reply.utterance,
        "action": reply.action,
        "reasoning": reply.reasoning,
    }
