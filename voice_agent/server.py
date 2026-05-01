"""FastAPI server — Vapi webhook + call lifecycle endpoints.

Vapi wiring:
  /vapi/llm/chat/completions  per-turn OpenAI-shaped request from Vapi's custom-LLM
  /vapi/webhook               lifecycle events (status-update, end-of-call-report)

Local simulation: scripts/play.py drives turns in-process (no HTTP needed).
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import httpx
import logfire
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlmodel import Session, select
from voice_agent import state
from voice_agent.agents.synthesis import SynthesisDeps, run_synthesis_safely as _synthesis_safely
from voice_agent.config import ENABLE_SYNTHESIS_REPORT, VAPI_TIMESTAMP_TOLERANCE_S, settings
from voice_agent.tracing import agent_span, init_tracing
from voice_agent.turn import TurnPipeline

LOGFIRE_BASE_URL = "https://logfire.pydantic.dev"

engine = state.make_engine(settings.database_url)

_background_tasks: set[asyncio.Task] = set()

# Per-call speech timing for latency instrumentation.
# Keyed by call_id; cleaned up on end-of-call-report.
_speech_ts: dict[str, dict[str, Any]] = {}

# One asyncio.Task per call: fires Vapi DELETE if user stays silent after assistant stops speaking.
_silence_watch_tasks: dict[str, asyncio.Task] = {}


def _fire(coro, *, name: str | None = None) -> asyncio.Task:
    task = asyncio.create_task(coro, name=name)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.init_db(engine)
    init_tracing(app=app, engine=engine)
    yield


app = FastAPI(title="voice-agent", lifespan=lifespan)



def _cancel_silence_watch(call_id: str) -> None:
    """Stop the extended-silence task for this call if one is running."""
    task = _silence_watch_tasks.pop(call_id, None)
    if task and not task.done():
        task.cancel()


def _end_call_vapi_delete(call_id: str, end_reason: str) -> None:
    """Mark a call ended in the DB, cancel silence watch, and fire Vapi DELETE — best-effort, never raises."""
    vapi_call_id: str | None = None
    try:
        with state.session_scope(engine) as session:
            call = session.get(state.Call, call_id)
            if call and call.status != "ended":
                vapi_call_id = call.vapi_call_id
                call.status = "ended"
                call.end_reason = end_reason
                call.ended_at = datetime.now(timezone.utc)
                session.add(call)
    except Exception:
        logfire.exception("end_call_vapi_delete_db_failed", call_id=call_id, end_reason=end_reason)
        return

    _cancel_silence_watch(call_id)
    _speech_ts.pop(call_id, None)

    if not vapi_call_id or not settings.vapi_api_key:
        if vapi_call_id and not settings.vapi_api_key:
            logfire.warning("vapi_delete_skipped_no_key", call_id=call_id, vapi_call_id=vapi_call_id, end_reason=end_reason)
        return

    async def _delete() -> None:
        try:
            async with httpx.AsyncClient() as client:
                await client.delete(
                    f"https://api.vapi.ai/call/{vapi_call_id}",
                    headers={"Authorization": f"Bearer {settings.vapi_api_key}"},
                    timeout=5,
                )
        except Exception:
            logfire.exception("vapi_delete_call_failed", vapi_call_id=vapi_call_id, end_reason=end_reason)

    _fire(_delete(), name=f"vapi-delete-call-{end_reason}")


def _end_call_on_error(call_id: str) -> None:
    """Mark a call ended and fire Vapi DELETE — best-effort, never raises."""
    _end_call_vapi_delete(call_id, "server_error")


def _schedule_silence_watch_if_enabled(call_id: str, vapi_call_id: str | None) -> None:
    """Start (or replace) a timer: end the call if the user does not start/stop speaking before the deadline.

    Fires after each assistant TTS `stopped` event. Cancelled when the user or assistant next speaks.
    """
    if settings.vapi_extended_silence_seconds <= 0 or not call_id:
        return

    _cancel_silence_watch(call_id)

    async def _run() -> None:
        try:
            await asyncio.sleep(settings.vapi_extended_silence_seconds)
        except asyncio.CancelledError:
            return
        try:
            with state.session_scope(engine) as session:
                call = session.get(state.Call, call_id)
                if not call or call.status == "ended":
                    return
            logfire.info("extended_silence_timeout", call_id=call_id, vapi_call_id=vapi_call_id)
            _end_call_vapi_delete(call_id, "extended_silence")
        except Exception:
            logfire.exception("silence_watch_failed", call_id=call_id)

    task = _fire(_run(), name=f"silence-watch-{call_id}")
    _silence_watch_tasks[call_id] = task

    def _on_done(t: asyncio.Task) -> None:
        if _silence_watch_tasks.get(call_id) is t:
            _silence_watch_tasks.pop(call_id, None)

    task.add_done_callback(_on_done)


@app.exception_handler(HTTPException)
async def _http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    call_id = getattr(request.state, "call_id", None)
    ctx = dict(status_code=exc.status_code, detail=exc.detail, method=request.method, path=request.url.path, call_id=call_id)
    if exc.status_code >= 500:
        logfire.error("http_error", **ctx)
    elif exc.status_code >= 400:
        logfire.warning("http_error", **ctx)
    return await http_exception_handler(request, exc)


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, HTTPException):
        return await _http_exception_handler(request, exc)
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
    try:
        with agent_span("synthesis", call_id):
            with state.session_scope(engine) as session:
                deps = SynthesisDeps(call_id=call_id, session=session)
                await _synthesis_safely(deps)
    except Exception:
        logfire.exception("synthesis_task_error", call_id=call_id)


async def _warmup_groq() -> None:
    """Fire a minimal Groq request so llama is loaded onto a GPU before the first real user turn.

    Groq de-allocates models after ~20 min of inactivity. The first request to a cold
    model is unstable (structured output fails). This throwaway completion warms it up
    while the assistant is still delivering its greeting message.
    """
    try:
        import groq as groq_sdk
        client = groq_sdk.AsyncGroq()
        await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )
        logfire.debug("groq_warmup_done")
    except Exception:
        logfire.debug("groq_warmup_failed")  # best-effort, never affects the call


async def _analyst_task(call_id: str) -> None:
    from voice_agent.agents.analyst import run_analyst_safely
    from voice_agent.models import AnalystDeps
    with agent_span("analyst", call_id):
        await run_analyst_safely(AnalystDeps(call_id=call_id, engine=engine))



# --- Vapi signature verification --------------------------------------------


async def _require_vapi_signature(request: Request) -> None:
    """Verify Vapi HMAC signature. No-op when VAPI_WEBHOOK_SECRET is unset (dev mode).

    Vapi sends the signature as "v1{hex}" in the configured header. The HMAC payload is
    "{timestamp}.{body}" when VAPI_TIMESTAMP_HEADER is set, otherwise just the raw body.
    Timestamps outside the tolerance window are rejected to prevent replay attacks.
    """
    secret = settings.vapi_webhook_secret
    if not secret:
        return

    sig = request.headers.get(settings.vapi_signature_header, "")
    if not sig:
        logfire.warning("vapi_signature_missing", path=request.url.path)
        raise HTTPException(status_code=403, detail="Missing signature header")

    body = await request.body()

    if settings.vapi_timestamp_header:
        ts = request.headers.get(settings.vapi_timestamp_header, "")
        if not ts:
            logfire.warning("vapi_timestamp_missing", path=request.url.path)
            raise HTTPException(status_code=403, detail="Missing timestamp header")
        try:
            ts_s = int(ts) / 1000
            age_s = time.time() - ts_s
            if abs(age_s) > VAPI_TIMESTAMP_TOLERANCE_S:
                logfire.warning(
                    "vapi_timestamp_stale",
                    path=request.url.path,
                    ts_raw=ts,
                    ts_s=round(ts_s, 3),
                    server_time_s=round(time.time(), 3),
                    age_s=round(age_s, 3),
                    tolerance_s=VAPI_TIMESTAMP_TOLERANCE_S,
                )
                raise HTTPException(status_code=403, detail="Request timestamp too old")
        except ValueError:
            raise HTTPException(status_code=403, detail="Invalid timestamp")
        payload = f"{ts}.{body.decode()}".encode()
    else:
        payload = body

    # Vapi prefixes the hex digest with "v1"
    sig_hex = sig.removeprefix("v1")
    expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig_hex):
        logfire.warning("vapi_signature_invalid", path=request.url.path)
        raise HTTPException(status_code=403, detail="Invalid signature")


# --- Static secret for custom LLM endpoint ----------------------------------


async def _require_llm_secret(request: Request) -> None:
    """Verify the static secret Vapi sends via model.headers. No-op when LLM_SECRET_TOKEN is unset (dev)."""
    secret = settings.llm_secret_token
    if not secret:
        return
    if not hmac.compare_digest(request.headers.get("X-Vapi-Secret", ""), secret):
        logfire.warning("llm_secret_invalid", path=request.url.path)
        raise HTTPException(status_code=401, detail="Invalid secret")


# --- Webhook ----------------------------------------------------------------


def _call_id_from_event(msg: dict) -> str | None:
    """Extract our internal call_id from Vapi's metadata field."""
    return msg.get("call", {}).get("assistant", {}).get("metadata", {}).get("call_id")


@app.post("/vapi/webhook")
async def vapi_webhook(request: Request, _: None = Depends(_require_vapi_signature)) -> dict[str, Any]:
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
                        _fire(_warmup_groq(), name="groq-warmup")
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
                        _cancel_silence_watch(call_id)
                        _speech_ts.pop(call_id, None)
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
            _cancel_silence_watch(call_id)
            _speech_ts.pop(call_id, None)

            # Extract Vapi's per-turn TurnLatency breakdown (seconds → ms).
            # Metrics live under msg.artifact.performanceMetrics, not msg.analysis.
            artifact = msg.get("artifact") or {}
            perf = artifact.get("performanceMetrics") or {}
            turn_latencies: list[dict] = perf.get("turnLatencies") or []
            vapi_latency: dict[str, Any] = {}
            if turn_latencies:
                def _avg_ms(field: str) -> int:
                    # Vapi sends latency values already in milliseconds.
                    vals = [t[field] for t in turn_latencies if field in t]
                    return round(sum(vals) / len(vals)) if vals else 0
                # vapi_turn_ms = endpointing + stt + llm + tts = true E2E from last audio byte
                vapi_latency = {
                    "vapi_turns": len(turn_latencies),
                    "vapi_endpointing_ms_avg": _avg_ms("endpointingLatency"),
                    "vapi_stt_ms_avg": _avg_ms("transcriberLatency"),
                    "vapi_llm_ms_avg": _avg_ms("modelLatency"),
                    "vapi_tts_ms_avg": _avg_ms("voiceLatency"),
                    "vapi_turn_ms_avg": _avg_ms("turnLatency"),  # true E2E from last audio byte
                }
                for i, t in enumerate(turn_latencies):
                    logfire.info(
                        "vapi_turn_latency",
                        call_id=call_id,
                        vapi_turn_index=i,
                        vapi_endpointing_ms=round(t.get("endpointingLatency", 0)),
                        vapi_stt_ms=round(t.get("transcriberLatency", 0)),
                        vapi_llm_ms=round(t.get("modelLatency", 0)),
                        vapi_tts_ms=round(t.get("voiceLatency", 0)),
                        vapi_turn_ms=round(t.get("turnLatency", 0)),
                    )
            else:
                logfire.warning(
                    "vapi_perf_metrics_absent",
                    call_id=call_id,
                    msg_keys=list(msg.keys()),
                    artifact_keys=list(artifact.keys()),
                    perf_keys=list(perf.keys()),
                )

            logfire.info(
                "call_ended",
                call_id=call_id,
                vapi_call_id=vapi_call_id,
                ended_reason=ended_reason,
                probes_asked=probes_asked,
                probes_total=probes_total,
                probe_utilization_pct=round(100 * probes_asked / probes_total) if probes_total else None,
                **vapi_latency,
            )
            if ENABLE_SYNTHESIS_REPORT:
                _fire(_synthesis_task(call_id), name=f"synthesis-{call_id}")
            else:
                logfire.info("synthesis_skipped", call_id=call_id, reason="ENABLE_SYNTHESIS_REPORT=false")
            return {}

        if event_type == "speech-update":
            role = msg.get("role")
            status = msg.get("status")
            vapi_turn = msg.get("turn")
            now = time.time()
            turnaround_ms: int | None = None
            tts_duration_ms: int | None = None

            if call_id:
                entry = _speech_ts.setdefault(call_id, {})
                if role == "user" and status == "stopped":
                    entry["user_stopped_at"] = now
                    entry["user_stopped_turn"] = vapi_turn
                    _cancel_silence_watch(call_id)
                elif role == "user" and status == "started":
                    _cancel_silence_watch(call_id)
                elif role == "assistant" and status == "started":
                    # Pop so the next turn can't accidentally reuse a stale value when
                    # speech-update(user, stopped) races or arrives after the LLM call.
                    user_stopped = entry.pop("user_stopped_at", None)
                    if user_stopped is not None:
                        turnaround_ms = int((now - user_stopped) * 1000)
                        vapi_pipeline_ms = entry.pop("vapi_pipeline_ms", None)
                        llm_ttft_ms = entry.pop("llm_ttft_ms", None)
                        filler_injected = entry.pop("filler_injected", False)
                        if vapi_pipeline_ms is not None and llm_ttft_ms is not None:
                            tts_ttft_ms = turnaround_ms - vapi_pipeline_ms - llm_ttft_ms
                            logfire.info(
                                "turn_latency",
                                call_id=call_id,
                                e2e_ms=turnaround_ms,
                                vapi_pipeline_ms=vapi_pipeline_ms,
                                llm_ttft_ms=llm_ttft_ms,
                                tts_ttft_ms=tts_ttft_ms,
                                filler_injected=filler_injected,
                            )
                            entry["e2e_ms"] = turnaround_ms
                    entry["assistant_started_at"] = now
                    _cancel_silence_watch(call_id)
                elif role == "assistant" and status == "stopped":
                    assistant_started = entry.get("assistant_started_at")
                    if assistant_started is not None:
                        tts_duration_ms = int((now - assistant_started) * 1000)
                    entry.pop("assistant_started_at", None)
                    _schedule_silence_watch_if_enabled(call_id, vapi_call_id)

            logfire.info(
                "vapi_speech_update",
                call_id=call_id,
                vapi_call_id=vapi_call_id,
                status=status,
                role=role,
                turn=vapi_turn,
                tts_started=(role == "assistant" and status == "started"),
                turnaround_ms=turnaround_ms,
                tts_duration_ms=tts_duration_ms,
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
async def vapi_llm(request: Request, body: VapiLLMBody, _: None = Depends(_require_llm_secret)):
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
        elif not messages or messages[-1].get("role") != "user":
            # Ghost turn: Vapi called the LLM but the last message is from the assistant,
            # meaning no real user utterance triggered this call (state-sync race condition).
            # Return an empty response so Vapi speaks nothing and no DB write happens.
            logfire.info("ghost_turn_skipped", call_id=call_id, vapi_call_id=vapi_call_id, message_count=len(messages))
        else:
            # Time from speech-update(user/stopped) to this LLM call. Includes Vapi's
            # full endpointing silence window + STT finalization + any speculative-call
            # retry overhead. NOT just STT time — label accordingly.
            vapi_pipeline_ms: int | None = None
            entry = _speech_ts.get(call_id, {})
            if "user_stopped_at" in entry:
                vapi_pipeline_ms = int((time.time() - entry["user_stopped_at"]) * 1000)

            # Stash immediately — before any streaming — so the
            # speech-update(assistant/started) handler finds it even if it fires
            # before the streaming loop finishes.
            if vapi_pipeline_ms is not None:
                _speech_ts.setdefault(call_id, {})["vapi_pipeline_ms"] = vapi_pipeline_ms

            with logfire.span(
                    "vapi_llm_request",
                    call_id=call_id,
                    vapi_call_id=vapi_call_id,
                    message_count=len(messages),
                    vapi_pipeline_ms=vapi_pipeline_ms,
                ) as req_span:
                reply_chars = 0
                pipeline = TurnPipeline(engine, call_id, vapi_messages=messages)
                try:
                    ttft_stashed = False
                    async for token in pipeline.stream_tokens():
                        reply_chars += len(token)
                        # Stash llm_ttft_ms + filler_injected on the first token.
                        # pipeline.ttft_ms is set by stream_tokens() before yielding,
                        # so it's available here. Stashing here ensures the values are
                        # in _speech_ts before Vapi fires speech-update(assistant/started),
                        # which can't arrive until after Vapi receives this first token.
                        if not ttft_stashed and vapi_pipeline_ms is not None and pipeline.ttft_ms is not None:
                            live_entry = _speech_ts.setdefault(call_id, {})
                            live_entry["llm_ttft_ms"] = pipeline.ttft_ms
                            live_entry["filler_injected"] = pipeline.filler_injected
                            ttft_stashed = True
                        chunk = {"id": "interviewer", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"role": "assistant", "content": token}, "finish_reason": None}]}
                        yield f"data: {json.dumps(chunk)}\n\n"
                    result = await pipeline.commit()
                    req_span.set_attribute("action", result.action)
                    req_span.set_attribute("reply_chars", reply_chars)
                    if result.ttft_ms is not None:
                        req_span.set_attribute("llm_ttft_ms", result.ttft_ms)
                    if result.should_run_analyst:
                        _fire(_analyst_task(call_id), name=f"analyst-{call_id}")
                except asyncio.CancelledError:
                    stream_cancelled = True
                    filler_already_sent = pipeline.filler_injected
                    logfire.info(
                        "vapi_llm_stream_cancelled",
                        call_id=call_id,
                        vapi_call_id=vapi_call_id,
                        reply_chars=reply_chars,
                        filler_already_sent=filler_already_sent,
                    )
                    if filler_already_sent:
                        # Vapi cancelled after the filler was played — user heard
                        # "Mm-hm," but nothing after. Vapi will retry the full call.
                        logfire.warning(
                            "vapi_filler_orphaned",
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
    _cancel_silence_watch(call_id)
    _speech_ts.pop(call_id, None)

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


def _build_voice_config() -> dict[str, Any]:
    provider = settings.vapi_voice_provider
    if not settings.vapi_voice_id:
        raise ValueError("VAPI_VOICE_ID must be set in .env")
    voice: dict[str, Any] = {"provider": provider, "voiceId": settings.vapi_voice_id}
    if provider != "11labs":
        return voice
    if settings.vapi_voice_model:
        voice["model"] = settings.vapi_voice_model
    for json_key, value in (
        ("stability", settings.vapi_voice_stability),
        ("similarityBoost", settings.vapi_voice_similarity_boost),
        ("style", settings.vapi_voice_style),
        ("speed", settings.vapi_voice_speed),
    ):
        if value is not None:
            voice[json_key] = value
    # chunkPlan buffers streamed assistant text before sending a slice to ElevenLabs.
    # Default without this is ~30 chars — short fillers never flush alone, so TTS waited for LLM text.
    voice["chunkPlan"] = {
        "enabled": True,
        "minCharacters": max(1, settings.vapi_voice_chunk_min_characters),
    }
    return voice


def _build_start_speaking_plan() -> dict[str, Any]:
    return {
        "smartEndpointingPlan": {"provider": "livekit"},
        "transcriptionEndpointingPlan": {
            "onPunctuationSeconds": 0.2,
            "onNoPunctuationSeconds": 0.5,
            "onNumberSeconds": 0.5,
        },
        "waitSeconds": settings.vapi_wait_seconds,
    }


def _build_stop_speaking_plan() -> dict[str, Any]:
    return {
        "numWords": settings.vapi_stop_num_words,
        "backoffSeconds": settings.vapi_stop_backoff_seconds,
        "acknowledgementPhrases": [
            "hmm", "mm-hmm", "yeah", "yes", "okay", "ok",
            "right", "uh-huh", "sure", "got it", "I see",
            "totally", "absolutely", "interesting",
        ],
    }


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
                **({"headers": {"X-Vapi-Secret": settings.llm_secret_token}} if settings.llm_secret_token else {}),
            },
            "server": {
                "url": f"{settings.webhook_url}/vapi/webhook",
                **({"credentialId": settings.vapi_server_credential_id} if settings.vapi_server_credential_id else {}),
            },
            "voice": _build_voice_config(),
            "transcriber": {"provider": "deepgram", "model": "nova-3", "language": "en"},
            "startSpeakingPlan": _build_start_speaking_plan(),
            "stopSpeakingPlan": _build_stop_speaking_plan(),
            "firstMessage": "Hey, thank you for getting on the call! Want to check if you can hear me before we get started.",
            "firstMessageMode": "assistant-speaks-first",
            "maxDurationSeconds": 1800,
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
