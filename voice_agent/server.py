"""FastAPI server — Vapi webhook + call lifecycle endpoints.

Webhook flow per turn:
  speech-update  → fire-and-forget analyst | await interviewer (≤1.8 s) → reply
  call-ended     → mark call complete | fire-and-forget synthesis

Local echo test (no Vapi needed):
  POST /echo  {"call_id": "...", "text": "respondent says this"}
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import logfire
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import func
from sqlmodel import Session, select

load_dotenv()

from voice_agent import state
from voice_agent.agents.analyst import run_analyst_safely as _analyst_safely
from voice_agent.agents.interviewer import run_interviewer_with_timeout
from voice_agent.agents.synthesis import SynthesisDeps, run_synthesis_safely as _synthesis_safely
from voice_agent.config import ANALYST_EVERY_N_TURNS
from voice_agent.models import AnalystDeps, InterviewerDeps, VapiEvent
from voice_agent.tracing import agent_span, init_tracing, log_interviewer_decision, turn_span

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///voice_agent.db")
VAPI_API_KEY = os.getenv("VAPI_API_KEY", "")
LOGFIRE_BASE_URL = "https://logfire.pydantic.dev"
LOGFIRE_PROJECT = os.getenv("LOGFIRE_PROJECT", "voice-agent")

engine = state.make_engine(DATABASE_URL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.init_db(engine)
    init_tracing(app=app, engine=engine)
    yield


app = FastAPI(title="voice-agent", lifespan=lifespan)


# --- Session helper ---------------------------------------------------------


def _new_session() -> Session:
    return Session(engine)


# --- Next turn number -------------------------------------------------------


def _next_turn_number(session: Session, call_id: str) -> int:
    result = session.exec(
        select(func.max(state.Turn.turn_number)).where(state.Turn.call_id == call_id)
    ).one()
    return (result or 0) + 1


# --- Background task wrappers -----------------------------------------------
# Each creates its own session so it can run independently of the request session.


async def _analyst_task(call_id: str) -> None:
    session = _new_session()
    try:
        with agent_span("analyst", call_id):
            deps = AnalystDeps(call_id=call_id, session=session)
            await _analyst_safely(deps)
    finally:
        session.close()


async def _synthesis_task(call_id: str) -> None:
    session = _new_session()
    try:
        with agent_span("synthesis", call_id):
            deps = SynthesisDeps(call_id=call_id, session=session)
            await _synthesis_safely(deps)
    finally:
        session.close()


# --- Webhook ----------------------------------------------------------------


@app.post("/vapi/webhook")
async def vapi_webhook(event: VapiEvent) -> dict[str, Any]:
    with logfire.span("vapi_event", type=event.type, call_id=event.call_id):

        if event.type == "call-started":
            with state.session_scope(engine) as session:
                call = session.get(state.Call, event.call_id)
                if call is None:
                    session.add(state.Call(id=event.call_id, status="active"))
                elif call.status == "pending":
                    call.status = "active"
                    session.add(call)
            return {"ok": True}

        if event.type == "speech-update":
            if not event.text:
                return {"message": ""}

            with state.session_scope(engine) as session:
                turn_number = _next_turn_number(session, event.call_id)

                # Fire analyst every ANALYST_EVERY_N_TURNS respondent turns.
                # turn_number is 1-based and odd for respondent turns (1, 3, 5, …).
                respondent_index = (turn_number - 1) // 2  # 0, 1, 2, …
                if respondent_index % ANALYST_EVERY_N_TURNS == 0:
                    asyncio.create_task(_analyst_task(event.call_id))

                deps = InterviewerDeps(
                    call_id=event.call_id,
                    session=session,
                    turn_number=turn_number,
                )

                with turn_span(event.call_id, turn_number, respondent_text=event.text):
                    t0 = time.perf_counter()
                    with agent_span("interviewer", event.call_id, turn_number=turn_number):
                        reply = await run_interviewer_with_timeout(deps, event.text)
                    latency_ms = int((time.perf_counter() - t0) * 1000)
                    log_interviewer_decision(
                        call_id=event.call_id,
                        turn_number=turn_number,
                        action=reply.action,
                        utterance=reply.utterance,
                        reasoning=reply.reasoning,
                        latency_ms=latency_ms,
                    )

                session.add_all([
                    state.Turn(
                        call_id=event.call_id,
                        turn_number=turn_number,
                        speaker="respondent",
                        text=event.text,
                    ),
                    state.Turn(
                        call_id=event.call_id,
                        turn_number=turn_number + 1,
                        speaker="interviewer",
                        text=reply.utterance,
                        action=reply.action,
                        reasoning=reply.reasoning,
                        latency_ms=latency_ms,
                    ),
                ])

            return {"message": reply.utterance}

        if event.type == "call-ended":
            with state.session_scope(engine) as session:
                call = session.get(state.Call, event.call_id)
                if call:
                    call.status = "ended"
                    call.end_reason = event.end_reason
                    call.ended_at = datetime.now(timezone.utc)
                    session.add(call)
            asyncio.create_task(_synthesis_task(event.call_id))
            return {"ok": True}

    return {"ok": True}


# --- Call lifecycle ---------------------------------------------------------


class StartCallRequest(BaseModel):
    scripted_questions: list[str]
    phone_number: str | None = None
    call_id: str | None = None


@app.post("/calls/start")
async def start_call(req: StartCallRequest) -> dict[str, str]:
    """Seed scripted questions and optionally dial out via Vapi."""
    call_id = req.call_id or str(uuid.uuid4())

    with state.session_scope(engine) as session:
        existing = session.get(state.Call, call_id)
        if existing:
            raise HTTPException(status_code=409, detail=f"Call {call_id} already exists")
        session.add(
            state.Call(
                id=call_id,
                phone_number=req.phone_number,
                scripted_questions=req.scripted_questions,
                status="pending",
            )
        )

    logfire.info("call_created", call_id=call_id, phone_number=req.phone_number)

    if req.phone_number and VAPI_API_KEY:
        await _dial_vapi(call_id, req.phone_number)

    return {"call_id": call_id}


@app.get("/calls/{call_id}/report")
async def get_report(call_id: str) -> JSONResponse:
    """Return synthesis report or 202 if not ready yet."""
    with state.session_scope(engine) as session:
        report = session.exec(
            select(state.SynthesisReport).where(state.SynthesisReport.call_id == call_id)
        ).first()

        if report is None:
            call = session.get(state.Call, call_id)
            if call is None:
                raise HTTPException(status_code=404, detail="Call not found")
            return JSONResponse(status_code=202, content={"status": "pending"})

        return JSONResponse(content={
            "call_id": call_id,
            "summary": report.summary,
            "themes": report.themes,
            "contradictions": report.contradictions,
            "key_quotes": report.key_quotes,
            "follow_up_questions": report.follow_up_questions,
        })


@app.get("/calls/{call_id}/trace")
async def get_trace(call_id: str) -> dict[str, str]:
    """Return a Logfire query URL for this call's spans."""
    with state.session_scope(engine) as session:
        call = session.get(state.Call, call_id)
        if call is None:
            raise HTTPException(status_code=404, detail="Call not found")

    import urllib.parse
    query = urllib.parse.quote(f'call_id="{call_id}"')
    url = f"{LOGFIRE_BASE_URL}/{LOGFIRE_PROJECT}/live?filter={query}"
    return {"call_id": call_id, "trace_url": url}


# --- Local echo test (no Vapi required) -------------------------------------


class EchoRequest(BaseModel):
    call_id: str
    text: str


@app.post("/echo")
async def echo(req: EchoRequest) -> dict[str, Any]:
    """Simulate a speech-update turn locally for testing without Vapi.

    Example:
        curl -s -X POST http://localhost:8000/echo \\
          -H 'Content-Type: application/json' \\
          -d '{"call_id": "test-1", "text": "I mostly use it for my commute."}'
    """
    event = VapiEvent(type="speech-update", call_id=req.call_id, text=req.text)
    return await vapi_webhook(event)


# --- Vapi dial-out helper (stub until Vapi key is configured) ---------------


async def _dial_vapi(call_id: str, phone_number: str) -> None:
    """POST to Vapi's outbound call API. Requires VAPI_API_KEY + VAPI_PHONE_NUMBER_ID."""
    import httpx

    phone_number_id = os.getenv("VAPI_PHONE_NUMBER_ID", "")
    assistant_id = os.getenv("VAPI_ASSISTANT_ID", "")
    webhook_url = os.getenv("WEBHOOK_URL", "")

    if not phone_number_id:
        logfire.warning("vapi_dial_skipped", reason="VAPI_PHONE_NUMBER_ID not set", call_id=call_id)
        return

    payload: dict[str, Any] = {
        "phoneNumberId": phone_number_id,
        "customer": {"number": phone_number},
    }
    if assistant_id:
        payload["assistantId"] = assistant_id
    if webhook_url:
        payload.setdefault("assistant", {})["serverUrl"] = webhook_url

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.vapi.ai/call/phone",
            headers={"Authorization": f"Bearer {VAPI_API_KEY}"},
            json=payload,
            timeout=10,
        )
    resp.raise_for_status()
    vapi_call_id = resp.json().get("id")

    with state.session_scope(engine) as session:
        call = session.get(state.Call, call_id)
        if call and vapi_call_id:
            call.vapi_call_id = vapi_call_id
            session.add(call)

    logfire.info("vapi_dial_initiated", call_id=call_id, vapi_call_id=vapi_call_id)


# --- Dev entrypoint ---------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("voice_agent.server:app", host="0.0.0.0", port=8000, reload=True)
