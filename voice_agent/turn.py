"""Shared speech-turn pipeline — used by the Vapi LLM endpoint and local play.py."""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

import logfire
from sqlmodel import select
from voice_agent import state
from voice_agent.agents.interviewer import (
    InterviewerStream,
    prepare_interviewer_turn_concurrent,
)
from voice_agent.models import InterviewerDeps, InterviewerOutput


@dataclass
class StreamTurnResult:
    action: str
    reasoning: str
    llm_latency_ms: int
    ttft_ms: int | None
    persist_ms: int
    should_run_analyst: bool


async def run_speech_turn(
    engine,
    call_id: str,
    vapi_messages: list[dict] | None = None,
) -> dict[str, Any]:
    """Non-streaming turn for play.py and evals — drives TurnPipeline and buffers tokens."""
    pipeline = TurnPipeline(engine, call_id, vapi_messages=vapi_messages)
    message = "".join([tok async for tok in pipeline.stream_tokens()])
    result = await pipeline.commit()
    return {
        "message": message,
        "action": result.action,
        "reasoning": result.reasoning,
        "llm_latency_ms": result.llm_latency_ms,
        "should_run_analyst": result.should_run_analyst,
    }


class TurnPipeline:
    """Two-phase streaming turn: stream tokens first, then commit side effects.

    Usage:
        pipeline = TurnPipeline(engine, call_id, vapi_messages)
        async for token in pipeline.stream_tokens():
            ...send token to client...
        result = await pipeline.commit()
    """

    def __init__(self, engine: Any, call_id: str, vapi_messages: list[dict] | None = None) -> None:
        self._engine = engine
        self._call_id = call_id
        self._vapi_messages = vapi_messages
        self._reply: InterviewerOutput | None = None
        self._turn_number: int = 0
        self._respondent_text: str = ""
        self._llm_latency_ms: int = 0
        self._first_token_ms: int | None = None

    async def stream_tokens(self) -> AsyncGenerator[str, None]:
        """Yield LLM text tokens as they arrive. Must be fully consumed before commit()."""
        call_id = self._call_id
        vapi_messages = self._vapi_messages

        last = vapi_messages[-1] if vapi_messages else None
        respondent_text = last.get("content", "") if last and last.get("role") == "user" else ""
        self._respondent_text = respondent_text

        with state.session_scope(self._engine) as session:
            self._turn_number = state.next_turn_number(session, call_id)
        logfire.info("turn_started", call_id=call_id, turn_number=self._turn_number)

        with logfire.span("interviewer_db_prep", call_id=call_id, turn_number=self._turn_number):
            prepared = await prepare_interviewer_turn_concurrent(
                self._engine,
                call_id,
                self._turn_number,
                respondent_text=respondent_text,
                vapi_messages=vapi_messages,
            )

        deps = InterviewerDeps(call_id=call_id, session=None, turn_number=self._turn_number)
        stream = InterviewerStream(deps, prepared)

        t0 = time.perf_counter()
        async for token in stream.tokens():
            if self._first_token_ms is None:
                self._first_token_ms = int((time.perf_counter() - t0) * 1000)
                logfire.info(
                    "interviewer_first_token",
                    call_id=call_id,
                    turn_number=self._turn_number,
                    ttft_ms=self._first_token_ms,
                )
            yield token
        self._reply = stream.output

        if self._reply is None:
            raise RuntimeError("InterviewerStream produced no output")
        self._llm_latency_ms = int((time.perf_counter() - t0) * 1000)
        logfire.info(
            "interviewer_stream_done",
            call_id=call_id,
            turn_number=self._turn_number,
            action=self._reply.action,
            utterance=self._reply.utterance,
            reasoning=self._reply.reasoning,
            llm_latency_ms=self._llm_latency_ms,
            ttft_ms=self._first_token_ms,
            fallback=self._reply.is_fallback,
            probe_source="analyst" if self._reply.probe_id_used else "interviewer" if self._reply.action == "probe" else None,
        )

    async def commit(self) -> StreamTurnResult:
        """Persist side effects after stream_tokens() is fully consumed."""
        if self._reply is None:
            raise RuntimeError("commit() called before stream_tokens() completed")
        reply = self._reply
        call_id = self._call_id
        vapi_messages = self._vapi_messages
        turn_number = self._turn_number

        persist_t0 = time.perf_counter()
        should_run_analyst = False
        with logfire.span("turn_persist", call_id=call_id, turn_number=turn_number, action=reply.action, probe_id_used=reply.probe_id_used):
            with state.session_scope(self._engine) as session:
                if reply.action in ("scripted", "skip_scripted"):
                    state.mark_scripted_asked(session, call_id)
                if reply.probe_id_used is not None:
                    probe = session.get(state.Probe, reply.probe_id_used)
                    if probe is not None:
                        lag = turn_number - (probe.generated_after_turn or 0)
                        logfire.info("probe_used", call_id=call_id, turn_number=turn_number, probe_id=reply.probe_id_used, probe_priority=probe.priority, analyst_lag_turns=lag)
                    state.mark_probe_asked(session, reply.probe_id_used)

                if vapi_messages is not None:
                    next_num = turn_number
                    if self._respondent_text:
                        session.add(state.Turn(call_id=call_id, turn_number=next_num, speaker="respondent", text=self._respondent_text))
                        next_num += 1
                    session.add(state.Turn(call_id=call_id, turn_number=next_num, speaker="interviewer", text=reply.utterance))
                    should_run_analyst = state.should_run_analyst(session, call_id)
        persist_ms = int((time.perf_counter() - persist_t0) * 1000)

        return StreamTurnResult(
            action=reply.action,
            reasoning=reply.reasoning,
            llm_latency_ms=self._llm_latency_ms,
            ttft_ms=self._first_token_ms,
            persist_ms=persist_ms,
            should_run_analyst=should_run_analyst,
        )
