"""Shared speech-turn pipeline — used by the Vapi LLM endpoint and local play.py."""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from typing import Any

import logfire

from voice_agent import state
from voice_agent.agents.interviewer import run_interviewer_with_timeout, stream_interviewer_utterance
from voice_agent.models import InterviewerDeps, InterviewerOutput
from voice_agent.tracing import agent_span


async def run_speech_turn(
    engine,
    call_id: str,
    respondent_text: str,
    vapi_messages: list[dict] | None = None,
) -> dict[str, Any]:
    """Process one respondent turn through the interviewer agent.

    Returns {message, action, reasoning}.

    vapi_messages: the OpenAI-formatted messages[] Vapi sends with each LLM
    request. When provided, conversation history comes from Vapi (source of
    truth for what was actually spoken). When None (play.py / evals), the
    interviewer falls back to reading recent_turns from the DB.

    Turn rows are NOT written here. Confirmed turns are written by the server
    from conversation-update webhook events, which fire only after Vapi has
    played the response — eliminating provisional/ghost turns from rapid-fire
    utterance boundaries.

    Two short sessions bracket the LLM call so SQLite is never write-locked
    during the 5 s interviewer budget.
    """
    # --- Phase 1a: increment turn counter before the LLM call so the write
    #     lock is released immediately. ---
    with state.session_scope(engine) as session:
        turn_number = state.next_turn_number(session, call_id)
    logfire.info("turn_started", call_id=call_id, turn_number=turn_number)

    # --- Phase 1b: read-only session — fetch study state + run LLM ---
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
            reply = await run_interviewer_with_timeout(deps, respondent_text, vapi_messages=vapi_messages)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            is_fallback = reply.reasoning.startswith("fallback:")
            span.set_attribute("action", reply.action)
            span.set_attribute("utterance", reply.utterance)
            span.set_attribute("reasoning", reply.reasoning)
            span.set_attribute("latency_ms", latency_ms)
            span.set_attribute("fallback", is_fallback)
            span.set_attribute("probe_source", "analyst" if reply.probe_id_used else "interviewer" if reply.action == "probe" else None)

    # --- Phase 2: apply study side-effects only (no Turn writes) ---
    with logfire.span(
        "turn_persist",
        call_id=call_id,
        turn_number=turn_number,
        action=reply.action,
        probe_id_used=reply.probe_id_used,
    ):
        with state.session_scope(engine) as session:
            if reply.action in ("scripted", "skip_scripted"):
                state.mark_scripted_asked(session, call_id)
            if reply.probe_id_used is not None:
                probe = session.get(state.Probe, reply.probe_id_used)
                if probe is not None:
                    lag = turn_number - (probe.generated_after_turn or 0)
                    logfire.info(
                        "probe_used",
                        call_id=call_id,
                        turn_number=turn_number,
                        probe_id=reply.probe_id_used,
                        probe_priority=probe.priority,
                        analyst_lag_turns=lag,
                    )
                state.mark_probe_asked(session, reply.probe_id_used)

    return {
        "message": reply.utterance,
        "action": reply.action,
        "reasoning": reply.reasoning,
        "latency_ms": latency_ms,
    }


async def run_speech_turn_stream(
    engine,
    call_id: str,
    respondent_text: str,
    vapi_messages: list[dict] | None = None,
) -> AsyncGenerator[str | dict[str, Any], None]:
    """Streaming variant of run_speech_turn for the SSE path.

    Async generator: yields str text chunks as they arrive from the LLM,
    then yields a single dict as the last item once side effects are applied.
    The caller distinguishes by isinstance.
    """
    with state.session_scope(engine) as session:
        turn_number = state.next_turn_number(session, call_id)
    logfire.info("turn_started", call_id=call_id, turn_number=turn_number)

    t0 = time.perf_counter()
    first_token_ms: int | None = None

    with state.session_scope(engine) as session:
        deps = InterviewerDeps(call_id=call_id, session=session, turn_number=turn_number)
        gen = stream_interviewer_utterance(deps, respondent_text, vapi_messages=vapi_messages)

    reply: InterviewerOutput | None = None
    async for item in gen:
        if isinstance(item, str):
            if first_token_ms is None:
                first_token_ms = int((time.perf_counter() - t0) * 1000)
                logfire.info(
                    "interviewer_first_token",
                    call_id=call_id,
                    turn_number=turn_number,
                    ttft_ms=first_token_ms,
                )
            yield item
        else:
            reply = item

    assert reply is not None
    llm_latency_ms = int((time.perf_counter() - t0) * 1000)

    logfire.info(
        "interviewer_stream_done",
        call_id=call_id,
        turn_number=turn_number,
        action=reply.action,
        utterance=reply.utterance,
        reasoning=reply.reasoning,
        llm_latency_ms=llm_latency_ms,
        ttft_ms=first_token_ms,
        fallback=reply.reasoning.startswith("fallback:"),
        probe_source="analyst" if reply.probe_id_used else "interviewer" if reply.action == "probe" else None,
    )

    persist_t0 = time.perf_counter()
    with logfire.span("turn_persist", call_id=call_id, turn_number=turn_number, action=reply.action, probe_id_used=reply.probe_id_used):
        with state.session_scope(engine) as session:
            if reply.action in ("scripted", "skip_scripted"):
                state.mark_scripted_asked(session, call_id)
            if reply.probe_id_used is not None:
                probe = session.get(state.Probe, reply.probe_id_used)
                if probe is not None:
                    lag = turn_number - (probe.generated_after_turn or 0)
                    logfire.info("probe_used", call_id=call_id, turn_number=turn_number, probe_id=reply.probe_id_used, probe_priority=probe.priority, analyst_lag_turns=lag)
                state.mark_probe_asked(session, reply.probe_id_used)
    persist_ms = int((time.perf_counter() - persist_t0) * 1000)

    yield {
        "action": reply.action,
        "reasoning": reply.reasoning,
        "latency_ms": llm_latency_ms,
        "llm_latency_ms": llm_latency_ms,
        "ttft_ms": first_token_ms,
        "persist_ms": persist_ms,
    }
