"""FastAPI server — Vapi webhook + call lifecycle endpoints.

Vapi wiring:
  /vapi/llm/chat/completions  per-turn OpenAI-shaped request from Vapi's custom-LLM
  /vapi/webhook               lifecycle events (status-update, end-of-call-report)

Local simulation: scripts/play.py drives turns in-process (no HTTP needed).
"""

from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import httpx
import logfire
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlmodel import Session, select
from voice_agent import state
from voice_agent.agents.synthesis import SynthesisDeps, run_synthesis_safely as _synthesis_safely
from voice_agent.config import ENABLE_SYNTHESIS_REPORT, settings
from voice_agent.tracing import agent_span, init_tracing
from voice_agent.turn import TurnPipeline

LOGFIRE_BASE_URL = "https://logfire.pydantic.dev"

engine = state.make_engine(settings.database_url)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.init_db(engine)
    init_tracing(app=app, engine=engine)
    yield


app = FastAPI(title="voice-agent", lifespan=lifespan)



def _end_call_on_error(call_id: str) -> None:
    """Mark a call ended and fire Vapi DELETE — best-effort, never raises."""
    try:
        with state.session_scope(engine) as session:
            call = session.get(state.Call, call_id)
            if call and call.status != "ended":
                vapi_call_id = call.vapi_call_id
                call.status = "ended"
                call.end_reason = "server_error"
                call.ended_at = datetime.now(timezone.utc)
                session.add(call)
        if vapi_call_id and settings.vapi_api_key:
            async def _delete():
                async with httpx.AsyncClient() as client:
                    await client.delete(
                        f"https://api.vapi.ai/call/{vapi_call_id}",
                        headers={"Authorization": f"Bearer {settings.vapi_api_key}"},
                        timeout=5,
                    )
            asyncio.create_task(_delete())
    except Exception:
        logfire.exception("end_call_on_error_failed", call_id=call_id)


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, HTTPException):
        return await http_exception_handler(request, exc)
    if isinstance(exc, RequestValidationError):
        call_id = getattr(request.state, "call_id", None)
        if call_id:
            _end_call_on_error(call_id)
        return await request_validation_exception_handler(request, exc)
    call_id = getattr(request.state, "call_id", None)
    logfire.exception(
        "unhandled_exception",
        method=request.method,
        path=request.url.path,
        exc_type=type(exc).__name__,
        call_id=call_id,
    )
    if call_id:
        _end_call_on_error(call_id)
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


async def _analyst_task(call_id: str) -> None:
    from voice_agent.agents.analyst import run_analyst_safely
    from voice_agent.models import AnalystDeps
    with agent_span("analyst", call_id):
        await run_analyst_safely(AnalystDeps(call_id=call_id, engine=engine))



# --- Webhook ----------------------------------------------------------------


def _call_id_from_event(msg: dict) -> str | None:
    """Extract our internal call_id from Vapi's metadata field."""
    return msg.get("call", {}).get("assistant", {}).get("metadata", {}).get("call_id")


@app.post("/vapi/webhook")
async def vapi_webhook(request: Request) -> dict[str, Any]:
    body = await request.json()
    msg = body.get("message", {})
    event_type = msg.get("type")
    vapi_call_id = msg.get("call", {}).get("id")
    call_id = _call_id_from_event(msg)

    request.state.call_id = call_id
    with logfire.span("vapi_event", type=event_type, call_id=call_id, vapi_call_id=vapi_call_id) as event_span:

        if event_type == "status-update":
            status = msg.get("status")
            event_span.set_attribute("status", status)
            if status == "in-progress" and call_id:
                with state.session_scope(engine) as session:
                    call = session.get(state.Call, call_id)
                    if call and call.status == "pending":
                        call.status = "active"
                        session.add(call)
                        logfire.info("call_activated", call_id=call_id, vapi_call_id=vapi_call_id)
            return {}

        if event_type == "end-of-call-report":
            if not call_id:
                logfire.warning("end_of_call_missing_call_id", vapi_call_id=vapi_call_id)
                return {}
            ended_reason = msg.get("endedReason")
            with state.session_scope(engine) as session:
                call = session.get(state.Call, call_id)
                if call:
                    # Guard is a no-op if DELETE /calls/{id} already fired — both paths are valid.
                    if call.status == "ended":
                        return {}
                    call.status = "ended"
                    call.end_reason = ended_reason
                    call.ended_at = datetime.now(timezone.utc)
                    session.add(call)
                    event_span.set_attribute("ended_reason", ended_reason)
                    probes_asked, probes_total = state.probe_utilization(session, call_id)
                else:
                    logfire.warning("end_of_call_unknown_call_id", call_id=call_id)
                    return {}
            logfire.info(
                "call_ended",
                call_id=call_id,
                vapi_call_id=vapi_call_id,
                ended_reason=ended_reason,
                probes_asked=probes_asked,
                probes_total=probes_total,
                probe_utilization_pct=round(100 * probes_asked / probes_total) if probes_total else None,
            )
            if ENABLE_SYNTHESIS_REPORT:
                asyncio.create_task(_synthesis_task(call_id))
            else:
                logfire.info("synthesis_skipped", call_id=call_id, reason="ENABLE_SYNTHESIS_REPORT=false")
            return {}

        if event_type == "speech-update":
            logfire.info(
                "vapi_speech_update",
                call_id=call_id,
                vapi_call_id=vapi_call_id,
                status=msg.get("status"),
                role=msg.get("role"),
                turn=msg.get("turn"),
                tts_started=(msg.get("role") == "assistant" and msg.get("status") == "started"),
            )
            return {}

        if event_type == "conversation-update":
            raw_messages = msg.get("messages", [])
            last_role_raw = raw_messages[-1].get("role") if raw_messages else None
            logfire.debug(
                "vapi_conversation_update",
                call_id=call_id,
                vapi_call_id=vapi_call_id,
                message_count=len(raw_messages),
                last_role=last_role_raw,
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
            logfire.error("call_already_exists", call_id=call_id)
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

    if req.phone_number and settings.vapi_api_key:
        await _dial_vapi(call_id, req.phone_number)

    return {"call_id": call_id}


@app.get("/calls/{call_id}/report")
async def get_report(request: Request, call_id: str) -> JSONResponse:
    """Return synthesis report, 202 if still generating, or 200 stub when synthesis is disabled."""
    request.state.call_id = call_id
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
            "pmf_score": report.pmf_score,
            "pmf_score_rationale": report.pmf_score_rationale,
            "competitive_signals": report.competitive_signals,
            "revenue_signals": report.revenue_signals,
            "ai_adoption_signals": report.ai_adoption_signals,
            "red_flags": report.red_flags,
            "investment_thesis_bullets": report.investment_thesis_bullets,
        })


@app.get("/calls/{call_id}/trace")
async def get_trace(request: Request, call_id: str) -> dict[str, str]:
    """Return a Logfire query URL for this call's spans."""
    request.state.call_id = call_id
    with state.session_scope(engine) as session:
        call = session.get(state.Call, call_id)
        if call is None:
            raise HTTPException(status_code=404, detail="Call not found")

    import urllib.parse
    query = urllib.parse.quote(f'call_id="{call_id}"')
    url = f"{LOGFIRE_BASE_URL}/{settings.effective_logfire_project_path}/live?filter={query}"
    return {"call_id": call_id, "trace_url": url}


# --- Vapi Custom LLM endpoint -----------------------------------------------


class _VapiMeta(BaseModel):
    model_config = ConfigDict(extra="allow")
    call_id: str | None = None


class _VapiAssistant(BaseModel):
    model_config = ConfigDict(extra="allow")
    metadata: _VapiMeta = Field(default_factory=_VapiMeta)


class _VapiCall(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str | None = None
    assistant: _VapiAssistant = Field(default_factory=_VapiAssistant)


class VapiLLMBody(BaseModel):
    model_config = ConfigDict(extra="allow")
    call: _VapiCall = Field(default_factory=_VapiCall)
    messages: list[dict[str, Any]] = []


@app.post("/vapi/llm/chat/completions")
async def vapi_llm(request: Request, body: VapiLLMBody):
    """Custom LLM endpoint called by Vapi for every turn.

    Vapi sends an OpenAI-compatible chat completion request, usually with
    stream=true. We return either SSE chunks or a single JSON response.
    """
    vapi_call_id = body.call.id
    call_id = body.call.assistant.metadata.call_id
    request.state.call_id = call_id
    messages = body.messages

    if call_id is None:
        logfire.warning(
            "vapi_llm_unknown_call",
            vapi_call_id=vapi_call_id,
            message_count=len(messages),
        )

    async def sse():
        stream_cancelled = False
        if call_id is None:
            fallback = "Thank you for your time."
            chunk = {"id": "interviewer", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"role": "assistant", "content": fallback}, "finish_reason": None}]}
            yield f"data: {json.dumps(chunk)}\n\n"
        else:
            with logfire.span(
                "vapi_llm_request",
                call_id=call_id,
                vapi_call_id=vapi_call_id,
                message_count=len(messages),
            ) as req_span:
                reply_chars = 0
                pipeline = TurnPipeline(engine, call_id, vapi_messages=messages)
                try:
                    async for token in pipeline.stream_tokens():
                        reply_chars += len(token)
                        chunk = {"id": "interviewer", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"role": "assistant", "content": token}, "finish_reason": None}]}
                        yield f"data: {json.dumps(chunk)}\n\n"
                    result = await pipeline.commit()
                    req_span.set_attribute("action", result.action)
                    req_span.set_attribute("reply_chars", reply_chars)
                    if result.ttft_ms is not None:
                        req_span.set_attribute("ttft_ms", result.ttft_ms)
                    if result.should_run_analyst:
                        asyncio.create_task(_analyst_task(call_id))
                except asyncio.CancelledError:
                    stream_cancelled = True
                    logfire.info(
                        "vapi_llm_stream_cancelled",
                        call_id=call_id,
                        vapi_call_id=vapi_call_id,
                        reply_chars=reply_chars,
                    )
                    return
                except Exception:
                    logfire.exception(
                        "vapi_llm_stream_error",
                        call_id=call_id,
                        vapi_call_id=vapi_call_id,
                        reply_chars=reply_chars,
                    )
                    _end_call_on_error(call_id)
                    return

        if stream_cancelled:
            return
        done = {"id": "interviewer", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        yield f"data: {json.dumps(done)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


@app.delete("/calls/{call_id}")
async def delete_call(request: Request, call_id: str) -> dict[str, str]:
    """Cancel an in-flight call. Fires DELETE to Vapi if the call is still active."""
    request.state.call_id = call_id

    with state.session_scope(engine) as session:
        call = session.get(state.Call, call_id)
        if call is None:
            raise HTTPException(status_code=404, detail="Call not found")
        if call.status == "ended":
            return {"call_id": call_id, "status": "already_ended"}
        vapi_call_id = call.vapi_call_id
        call.status = "ended"
        call.end_reason = "deleted"
        call.ended_at = datetime.now(timezone.utc)
        session.add(call)

    if vapi_call_id and settings.vapi_api_key:
        async with httpx.AsyncClient() as client:
            resp = await client.delete(
                f"https://api.vapi.ai/call/{vapi_call_id}",
                headers={"Authorization": f"Bearer {settings.vapi_api_key}"},
                timeout=10,
            )
        if not resp.is_success:
            logfire.warning("vapi_cancel_failed", call_id=call_id, vapi_call_id=vapi_call_id,
                            status=resp.status_code)

    logfire.info("call_deleted", call_id=call_id, vapi_call_id=vapi_call_id)
    return {"call_id": call_id, "status": "ended"}


# --- Vapi dial-out helper ---------------


def _vapi_assistant_voice() -> dict[str, Any]:
    provider = settings.vapi_voice_provider
    if not settings.vapi_voice_id:
        raise ValueError("VAPI_VOICE_ID must be set in .env")
    voice: dict[str, Any] = {"provider": provider, "voiceId": settings.vapi_voice_id}
    if provider != "11labs":
        return voice
    for json_key, value in (
        ("stability", settings.vapi_voice_stability),
        ("similarityBoost", settings.vapi_voice_similarity_boost),
        ("style", settings.vapi_voice_style),
        ("speed", settings.vapi_voice_speed),
    ):
        if value is not None:
            voice[json_key] = value
    return voice


async def _dial_vapi(call_id: str, phone_number: str) -> None:
    """POST to Vapi's outbound call API. Requires VAPI_API_KEY + VAPI_PHONE_NUMBER_ID + WEBHOOK_URL."""
    if not settings.vapi_phone_number_id:
        logfire.warning("vapi_dial_skipped", reason="VAPI_PHONE_NUMBER_ID not set", call_id=call_id)
        return
    if not settings.webhook_url:
        logfire.warning("vapi_dial_skipped", reason="WEBHOOK_URL not set", call_id=call_id)
        return

    payload: dict[str, Any] = {
        "phoneNumberId": settings.vapi_phone_number_id,
        "customer": {"number": phone_number},
        "assistant": {
            "model": {
                "provider": "custom-llm",
                "url": f"{settings.webhook_url}/vapi/llm/chat/completions",
                "model": "interviewer",
            },
            "serverUrl": f"{settings.webhook_url}/vapi/webhook",
            "voice": _vapi_assistant_voice(),
            "transcriber": {"provider": "deepgram", "model": "nova-2", "language": "en"},
            "firstMessage": "Hey! Thank you for getting on the call — just want to check if you can hear me before we get started.",
            "firstMessageMode": "assistant-speaks-first",
            "metadata": {"call_id": call_id},
        },
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.vapi.ai/call",
            headers={"Authorization": f"Bearer {settings.vapi_api_key}"},
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
