"""FastAPI server — Vapi webhook + call lifecycle endpoints.

Vapi wiring:
  /vapi/llm/chat/completions  per-turn OpenAI-shaped request from Vapi's custom-LLM
  /vapi/webhook               lifecycle events (status-update, end-of-call-report)

Local simulation: scripts/play.py drives turns in-process (no HTTP needed).
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import logfire
from opentelemetry import trace
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from sqlmodel import Session, select

load_dotenv()

from voice_agent import state
from voice_agent.agents.synthesis import SynthesisDeps, run_synthesis_safely as _synthesis_safely
from voice_agent.config import ENABLE_SYNTHESIS_REPORT
from voice_agent.tracing import agent_span, init_tracing
from voice_agent.turn import run_speech_turn

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


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # HTTPException and RequestValidationError are expected control flow — don't noise-up Logfire.
    if isinstance(exc, (HTTPException, RequestValidationError)):
        raise exc
    logfire.exception(
        "unhandled_exception",
        method=request.method,
        path=request.url.path,
        exc_type=type(exc).__name__,
    )
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


async def _synthesis_task(call_id: str) -> None:
    session = Session(engine)
    try:
        with agent_span("synthesis", call_id):
            deps = SynthesisDeps(call_id=call_id, session=session)
            await _synthesis_safely(deps)
    except Exception:
        logfire.exception("synthesis_task_error", call_id=call_id)
    finally:
        session.close()


# --- Webhook ----------------------------------------------------------------


@app.post("/vapi/webhook")
async def vapi_webhook(request: Request) -> dict[str, Any]:
    body = await request.json()
    msg = body.get("message", {})
    event_type = msg.get("type")
    vapi_call_id = msg.get("call", {}).get("id")

    with logfire.span("vapi_event", type=event_type, vapi_call_id=vapi_call_id) as event_span:

        if event_type == "status-update":
            status = msg.get("status")
            event_span.set_attribute("status", status)
            if status == "in-progress" and vapi_call_id:
                with state.session_scope(engine) as session:
                    call = session.exec(
                        select(state.Call).where(state.Call.vapi_call_id == vapi_call_id)
                    ).first()
                    if call and call.status == "pending":
                        call.status = "active"
                        session.add(call)
                        event_span.set_attribute("call_id", call.id)
                        logfire.info("call_activated", call_id=call.id, vapi_call_id=vapi_call_id)
            return {}

        if event_type == "end-of-call-report":
            if not vapi_call_id:
                logfire.warning("end_of_call_missing_vapi_id")
                return {}
            call_id: str | None = None
            with state.session_scope(engine) as session:
                call = session.exec(
                    select(state.Call).where(state.Call.vapi_call_id == vapi_call_id)
                ).first()
                if call:
                    call.status = "ended"
                    call.end_reason = msg.get("endedReason")
                    call.ended_at = datetime.now(timezone.utc)
                    session.add(call)
                    call_id = call.id
                    event_span.set_attribute("call_id", call_id)
                    event_span.set_attribute("ended_reason", msg.get("endedReason"))
                else:
                    logfire.warning("end_of_call_unknown_vapi_id", vapi_call_id=vapi_call_id)
            if call_id:
                logfire.info("call_ended", call_id=call_id, ended_reason=msg.get("endedReason"))
                if ENABLE_SYNTHESIS_REPORT:
                    asyncio.create_task(_synthesis_task(call_id))
                else:
                    logfire.info("synthesis_skipped", call_id=call_id, reason="ENABLE_SYNTHESIS_REPORT=false")
            return {}

        if event_type == "speech-update":
            logfire.info(
                "vapi_speech_update",
                vapi_call_id=vapi_call_id,
                status=msg.get("status"),
                role=msg.get("role"),
            )
            return {}

        if event_type == "conversation-update":
            logfire.info(
                "vapi_conversation_update",
                vapi_call_id=vapi_call_id,
                message_count=len(msg.get("messages", [])),
            )
            return {}

        logfire.debug("vapi_event_ignored", type=event_type, payload=msg)
        return {}


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

    logfire.info("call_created", call_id=call_id, phone_number=req.phone_number,
                 question_count=len(req.scripted_questions))

    if req.phone_number and VAPI_API_KEY:
        await _dial_vapi(call_id, req.phone_number)

    return {"call_id": call_id}


@app.get("/calls/{call_id}/report")
async def get_report(call_id: str) -> JSONResponse:
    """Return synthesis report, 202 if still generating, or 200 stub when synthesis is disabled."""
    with state.session_scope(engine) as session:
        report = session.exec(
            select(state.SynthesisReport).where(state.SynthesisReport.call_id == call_id)
        ).first()

        if report is None:
            call = session.get(state.Call, call_id)
            if call is None:
                raise HTTPException(status_code=404, detail="Call not found")
            if call.status == "ended" and not ENABLE_SYNTHESIS_REPORT:
                return JSONResponse(
                    status_code=200,
                    content={
                        "call_id": call_id,
                        "status": "disabled",
                        "summary": "",
                        "themes": [],
                        "contradictions": [],
                        "key_quotes": [],
                        "follow_up_questions": [],
                    },
                )
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
    project_path = os.getenv("LOGFIRE_PROJECT_PATH", LOGFIRE_PROJECT)
    url = f"{LOGFIRE_BASE_URL}/{project_path}/live?filter={query}"
    return {"call_id": call_id, "trace_url": url}


# --- Vapi Custom LLM endpoint -----------------------------------------------


@app.post("/vapi/llm/chat/completions")
async def vapi_llm(request: Request):
    """Custom LLM endpoint called by Vapi for every turn.

    Vapi sends an OpenAI-compatible chat completion request, usually with
    stream=true. We return either SSE chunks or a single JSON response.
    """
    import json

    body = await request.json()
    vapi_call_id = body.get("call", {}).get("id")
    messages = body.get("messages", [])
    stream = bool(body.get("stream", False))

    user_msgs = [m for m in messages if m.get("role") == "user"]
    respondent_text = user_msgs[-1].get("content", "") if user_msgs else ""
    respondent_hash = (
        hashlib.sha256(respondent_text.encode("utf-8")).hexdigest()[:16]
        if respondent_text
        else ""
    )

    with state.session_scope(engine) as session:
        call = session.exec(
            select(state.Call).where(state.Call.vapi_call_id == vapi_call_id)
        ).first()
        if call is None:
            logfire.warning(
                "vapi_llm_unknown_call",
                vapi_call_id=vapi_call_id,
                message_count=len(messages),
            )
            content = "Thank you for your time."
            call_id = None
        else:
            call_id = call.id

    with logfire.span(
        "vapi_llm_request",
        call_id=call_id,
        vapi_call_id=vapi_call_id,
        respondent_chars=len(respondent_text),
        respondent_sha256_16=respondent_hash,
        message_count=len(messages),
        stream=stream,
    ) as req_span:
        otel_span = trace.get_current_span()
        trace_id = format(otel_span.get_span_context().trace_id, "032x")

        if call_id is not None:
            t0 = time.perf_counter()
            result = await run_speech_turn(engine, call_id, respondent_text)
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            content = result["message"]
            req_span.set_attribute("action", result["action"])
            req_span.set_attribute("reply_chars", len(content))
            req_span.set_attribute("elapsed_ms", elapsed_ms)
        else:
            req_span.set_attribute("action", "fallback_unknown_call")

    headers = {"X-Trace-ID": trace_id}

    if not stream:
        return JSONResponse(
            {
                "id": "interviewer",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            },
            headers=headers,
        )

    async def sse():
        chunk = {
            "id": "interviewer",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        done = {
            "id": "interviewer",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(done)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream", headers=headers)


# --- Vapi dial-out helper ---------------


def _vapi_assistant_voice() -> dict[str, Any]:
    """Build Vapi `assistant.voice` (ElevenLabs or Vapi catalog voices).

    - ``provider=vapi`` + ``voiceId=Elliot`` matches the dashboard "Vapi" voice library
      (names like Elliot, Savannah — see GET /assistant JSON).
    - ``provider=11labs`` + opaque ``voiceId`` is standard ElevenLabs.
    Optional tuning keys apply only to 11labs; see .env.example.
    """
    raw_provider = os.getenv("VAPI_VOICE_PROVIDER")
    if raw_provider is None or not str(raw_provider).strip():
        provider = "11labs"
    else:
        provider = str(raw_provider).strip().lower()
    default_id = "21m00Tcm4TlvDq8ikWAM" if provider == "11labs" else "Elliot"
    voice_id = os.getenv("VAPI_VOICE_ID", default_id)
    voice: dict[str, Any] = {"provider": provider, "voiceId": voice_id}
    if provider != "11labs":
        return voice
    optional_floats: tuple[tuple[str, str], ...] = (
        ("stability", "VAPI_VOICE_STABILITY"),
        ("similarityBoost", "VAPI_VOICE_SIMILARITY_BOOST"),
        ("style", "VAPI_VOICE_STYLE"),
        ("speed", "VAPI_VOICE_SPEED"),
    )
    for json_key, env_name in optional_floats:
        raw = os.getenv(env_name)
        if raw is None or raw.strip() == "":
            continue
        try:
            voice[json_key] = float(raw)
        except ValueError:
            logfire.warning("vapi_voice_param_invalid", env=env_name, value=raw)
    return voice


async def _dial_vapi(call_id: str, phone_number: str) -> None:
    """POST to Vapi's outbound call API. Requires VAPI_API_KEY + VAPI_PHONE_NUMBER_ID + WEBHOOK_URL."""
    import httpx

    phone_number_id = os.getenv("VAPI_PHONE_NUMBER_ID", "")
    webhook_url = os.getenv("WEBHOOK_URL", "")

    if not phone_number_id:
        logfire.warning("vapi_dial_skipped", reason="VAPI_PHONE_NUMBER_ID not set", call_id=call_id)
        return
    if not webhook_url:
        logfire.warning("vapi_dial_skipped", reason="WEBHOOK_URL not set", call_id=call_id)
        return

    payload: dict[str, Any] = {
        "phoneNumberId": phone_number_id,
        "customer": {"number": phone_number},
        "assistant": {
            "model": {
                "provider": "custom-llm",
                "url": f"{webhook_url}/vapi/llm",
                "model": "interviewer",
            },
            "serverUrl": f"{webhook_url}/vapi/webhook",
            "voice": _vapi_assistant_voice(),
            "transcriber": {"provider": "deepgram", "model": "nova-2", "language": "en"},
            "firstMessageMode": "assistant-speaks-first",
        },
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.vapi.ai/call/phone",
            headers={"Authorization": f"Bearer {VAPI_API_KEY}"},
            json=payload,
            timeout=10,
        )
    if not resp.is_success:
        logfire.error("vapi_dial_error", call_id=call_id, status=resp.status_code, body=resp.text[:4000])
        raise HTTPException(status_code=502, detail=f"Vapi error {resp.status_code}: {resp.text}")

    vapi_call_id = resp.json().get("id")

    with state.session_scope(engine) as session:
        call = session.get(state.Call, call_id)
        if call and vapi_call_id:
            call.vapi_call_id = vapi_call_id
            session.add(call)

    logfire.info("vapi_dial_initiated", call_id=call_id, vapi_call_id=vapi_call_id, phone_number=phone_number)


# --- Dev entrypoint ---------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("voice_agent.server:app", host="0.0.0.0", port=8000, reload=True)
