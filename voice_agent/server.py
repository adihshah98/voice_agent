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
import secrets
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
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy import update
from sqlmodel import Session, select
from voice_agent import state
from voice_agent.agents.synthesis import SynthesisDeps, run_synthesis_safely as _synthesis_safely
from voice_agent.config import ENABLE_SYNTHESIS_REPORT, VAPI_TIMESTAMP_TOLERANCE_S, settings
from voice_agent.tracing import agent_span, init_tracing
from voice_agent.turn import TurnPipeline

LOGFIRE_BASE_URL = "https://logfire.pydantic.dev"

# Rate limiter — in-memory, keyed by client IP.
# Configure via CALLS_START_RATE_LIMIT env var, e.g. "5/minute". "" = disabled.
limiter = Limiter(key_func=get_remote_address)

engine = state.make_engine(settings.database_url)

_background_tasks: set[asyncio.Task] = set()

# Per-call speech timing for latency instrumentation.
# Keyed by call_id; cleaned up on end-of-call-report.
_speech_ts: dict[str, dict[str, Any]] = {}


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


app = FastAPI(
    title="voice-agent",
    lifespan=lifespan,
    swagger_ui_parameters={"persistAuthorization": True},
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)



def _end_call_vapi_delete(call_id: str, end_reason: str) -> None:
    """Mark a call ended in the DB and fire Vapi DELETE — best-effort, never raises.

    Uses a single conditional UPDATE (Postgres-safe under concurrent callers): only the first
    transition from pending|active → ended returns vapi_call_id for DELETE; duplicates no-op the DB write.
    """
    vapi_call_id: str | None = None
    ended_at = datetime.now(timezone.utc)
    try:
        with state.session_scope(engine) as session:
            stmt = (
                update(state.Call)
                .where(
                    state.Call.id == call_id,
                    state.Call.status.in_(["pending", "active"]),
                )
                .values(status="ended", end_reason=end_reason, ended_at=ended_at)
                .returning(state.Call.vapi_call_id)
            )
            row = session.execute(stmt).fetchone()
            if row is not None:
                vapi_call_id = row[0]
    except Exception:
        logfire.exception("end_call_vapi_delete_db_failed", call_id=call_id, end_reason=end_reason)
        return

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
    digest_ok = len(expected) == len(sig_hex) and hmac.compare_digest(expected, sig_hex)
    if not digest_ok:
        ts_name = settings.vapi_timestamp_header
        logfire.warning(
            "vapi_signature_invalid",
            path=request.url.path,
            expected_prefix=expected[:12],
            received_prefix=(sig_hex[:12] if sig_hex else ""),
            payload_len=len(payload),
            timestamp_header_present=(bool(request.headers.get(ts_name, "")) if ts_name else False),
            hmac_includes_timestamp=bool(ts_name),
        )
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


_bearer_scheme = HTTPBearer(auto_error=False)


async def _require_api_auth(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> None:
    """Require Authorization: Bearer … when API_AUTH_TOKEN is set. No-op when unset (dev).

    Using HTTPBearer as the dependency makes FastAPI declare the BearerAuth security
    scheme in the OpenAPI spec, so Swagger UI shows the Authorize button.
    """
    expected = settings.api_auth_token
    if not expected:
        return
    given = credentials.credentials if credentials else ""
    if not given or len(given) != len(expected) or not secrets.compare_digest(given, expected):
        logfire.warning("api_auth_invalid", path=request.url.path)
        raise HTTPException(status_code=401, detail="Unauthorized")


# --- Webhook ----------------------------------------------------------------


def _call_terminal_for_llm(session: Session, call_id: str) -> bool:
    """True if the custom LLM must not run a turn (missing row, ended, or dial failed)."""
    oc = session.get(state.Call, call_id)
    if oc is None:
        return True
    if oc.status == "ended":
        return True
    if oc.dial_status == state.DIAL_FAILED:
        return True
    return False


def _mark_dial_failed(call_id: str, *, end_reason: str, dial_error: str | None) -> None:
    """Best-effort: pending + queued|dialing → ended + dial_failed; cleanup in-process speech state."""
    ended_at = datetime.now(timezone.utc)
    err = (dial_error or "")[:4000]
    try:
        with state.session_scope(engine) as session:
            session.execute(
                update(state.Call)
                .where(
                    state.Call.id == call_id,
                    state.Call.status == "pending",
                    state.Call.dial_status.in_([state.DIAL_QUEUED, state.DIAL_DIALING]),
                )
                .values(
                    status="ended",
                    dial_status=state.DIAL_FAILED,
                    end_reason=end_reason,
                    dial_error=err,
                    ended_at=ended_at,
                )
            )
    except Exception:
        logfire.exception("mark_dial_failed_db_error", call_id=call_id)
    _speech_ts.pop(call_id, None)


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
                    result = session.execute(
                        update(state.Call)
                        .where(state.Call.id == call_id, state.Call.status == "pending")
                        .values(status="active")
                    )
                    if result.rowcount == 1:
                        logfire.info("call_activated", call_id=call_id, vapi_call_id=vapi_call_id)
            return {}

        if event_type == "end-of-call-report":
            if not call_id:
                logfire.warning("end_of_call_missing_call_id", vapi_call_id=vapi_call_id)
                return {}
            ended_reason = msg.get("endedReason")
            ended_at = datetime.now(timezone.utc)
            with state.session_scope(engine) as session:
                result = session.execute(
                    update(state.Call)
                    .where(
                        state.Call.id == call_id,
                        state.Call.status.in_(["pending", "active"]),
                    )
                    .values(status="ended", end_reason=ended_reason, ended_at=ended_at)
                )
                if result.rowcount != 1:
                    call = session.get(state.Call, call_id)
                    if call is None:
                        logfire.warning("end_of_call_unknown_call_id", call_id=call_id)
                        return {}
                    _speech_ts.pop(call_id, None)
                    return {}
                event_span.set_attribute("ended_reason", ended_reason)
                probes_asked, probes_total = state.probe_utilization(session, call_id)
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

            with state.session_scope(engine) as session:
                token_totals = state.call_llm_token_totals(session, call_id)
                summary_stats = state.call_summary_stats(session, call_id)

            logfire.info(
                "call_ended",
                call_id=call_id,
                vapi_call_id=vapi_call_id,
                ended_reason=ended_reason,
                probes_asked=probes_asked,
                probes_total=probes_total,
                probe_utilization_pct=round(100 * probes_asked / probes_total) if probes_total else None,
                **token_totals,
                **vapi_latency,
            )

            # Post-call aggregate eval — scripted arc completion, probe utilization,
            # barge-in rate, fallback rate. All derived from DB; no LLM call needed.
            if summary_stats:
                interviewer_turns = summary_stats.get("interviewer_turns") or 0
                logfire.info(
                    "call_summary_eval",
                    call_id=call_id,
                    **summary_stats,
                    barge_in_rate_pct=(
                        round(100 * summary_stats["barge_in_count"] / interviewer_turns)
                        if interviewer_turns else None
                    ),
                    fallback_rate_pct=(
                        round(100 * summary_stats["fallback_count"] / interviewer_turns)
                        if interviewer_turns else None
                    ),
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
            tts_duration_ms: int | None = None

            if call_id:
                entry = _speech_ts.setdefault(call_id, {})
                if role == "assistant" and status == "started":
                    llm_ttft_ms = entry.pop("llm_ttft_ms", None)
                    filler_injected = entry.pop("filler_injected", False)
                    logfire.info(
                        "turn_latency",
                        call_id=call_id,
                        llm_ttft_ms=llm_ttft_ms,
                        filler_injected=filler_injected,
                    )
                    entry["assistant_started_at"] = now
                elif role == "assistant" and status == "stopped":
                    assistant_started = entry.get("assistant_started_at")
                    if assistant_started is not None:
                        tts_duration_ms = int((now - assistant_started) * 1000)
                    entry.pop("assistant_started_at", None)

            logfire.info(
                "vapi_speech_update",
                call_id=call_id,
                vapi_call_id=vapi_call_id,
                status=status,
                role=role,
                turn=vapi_turn,
                tts_started=(role == "assistant" and status == "started"),
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


def _load_default_questions(product: str) -> list[str]:
    """Load questions from data/investor_questions.yaml, substituting [product]."""
    import yaml
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "data" / "investor_questions.yaml"
    raw = yaml.safe_load(path.read_text())
    return [
        entry["question"].strip().replace("[product]", product or "the product")
        for entry in raw["questions"]
    ]


class StartCallRequest(BaseModel):
    product: str
    phone_number: str | None = None


@app.post("/calls/start")
@limiter.limit(settings.calls_start_rate_limit or "9999/hour")
async def start_call(request: Request, req: StartCallRequest, _: None = Depends(_require_api_auth)) -> JSONResponse:
    """Dial out via Vapi, loading scripted questions from investor_questions.yaml for the given product."""
    call_id = str(uuid.uuid4())
    scripted_questions = _load_default_questions(req.product)

    wants_dial = bool(req.phone_number and settings.vapi_api_key)
    can_dial = bool(
        wants_dial and settings.vapi_phone_number_id and settings.webhook_url
    )
    sync_abort = bool(wants_dial and not can_dial)
    ended_at = datetime.now(timezone.utc) if sync_abort else None
    dial_status: str | None = None
    status = "pending"
    end_reason: str | None = None
    dial_error: str | None = None
    if can_dial:
        dial_status = state.DIAL_QUEUED
    elif sync_abort:
        dial_status = state.DIAL_FAILED
        status = "ended"
        end_reason = "dial_skipped"
        if not settings.vapi_phone_number_id and not settings.webhook_url:
            dial_error = "VAPI_PHONE_NUMBER_ID and WEBHOOK_URL are not set"
        elif not settings.vapi_phone_number_id:
            dial_error = "VAPI_PHONE_NUMBER_ID is not set"
        else:
            dial_error = "WEBHOOK_URL is not set"

    with state.session_scope(engine) as session:
        existing = session.get(state.Call, call_id)
        if existing:
            logfire.error("call_already_exists", call_id=call_id)
            raise HTTPException(status_code=409, detail=f"Call {call_id} already exists")
        session.add(
            state.Call(
                id=call_id,
                phone_number=req.phone_number,
                scripted_questions=scripted_questions,
                status=status,
                dial_status=dial_status,
                dial_error=dial_error,
                end_reason=end_reason,
                ended_at=ended_at,
            )
        )

    logfire.info(
        "call_created",
        call_id=call_id,
        phone_number=req.phone_number,
        question_count=len(scripted_questions),
        product=req.product,
        dial_status=dial_status,
    )

    body: dict[str, Any] = {"call_id": call_id, "dial_status": dial_status}
    if dial_error:
        body["dial_error"] = dial_error

    if sync_abort:
        return JSONResponse(status_code=200, content=body)

    if can_dial:
        _fire(_dial_vapi(call_id, req.phone_number), name=f"dial-{call_id}")
        return JSONResponse(status_code=202, content=body)

    return JSONResponse(status_code=200, content=body)


@app.get("/calls/{call_id}")
async def get_call(request: Request, call_id: str) -> dict[str, Any]:
    """Poll call lifecycle: status, dial outcome, Vapi id (Phase 4)."""
    request.state.call_id = call_id
    with state.session_scope(engine) as session:
        call = session.get(state.Call, call_id)
        if call is None:
            raise HTTPException(status_code=404, detail="Call not found")
        return {
            "call_id": call.id,
            "status": call.status,
            "dial_status": call.dial_status,
            "vapi_call_id": call.vapi_call_id,
            "end_reason": call.end_reason,
            "dial_error": call.dial_error,
            "phone_number": call.phone_number,
            "started_at": call.started_at.isoformat() if call.started_at else None,
            "ended_at": call.ended_at.isoformat() if call.ended_at else None,
        }


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
            if call.dial_status == state.DIAL_FAILED:
                return JSONResponse(
                    status_code=200,
                    content={
                        "call_id": call_id,
                        "status": "dial_failed",
                        "dial_error": call.dial_error or "",
                        "summary": "",
                        "themes": [],
                        "contradictions": [],
                        "key_quotes": [],
                        "follow_up_questions": [],
                    },
                )
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
            skip_pipeline = False
            if call_id is None:
                logfire.warning(
                    "vapi_llm_missing_call_id_with_user_message",
                    vapi_call_id=vapi_call_id,
                    message_count=len(messages),
                )
                skip_pipeline = True
            else:
                with state.session_scope(engine) as session:
                    skip_pipeline = _call_terminal_for_llm(session, call_id)
                if skip_pipeline:
                    logfire.warning(
                        "vapi_llm_skipped_terminal_call",
                        call_id=call_id,
                        vapi_call_id=vapi_call_id,
                    )

            if not skip_pipeline:
                with logfire.span(
                        "vapi_llm_request",
                        call_id=call_id,
                        vapi_call_id=vapi_call_id,
                        message_count=len(messages),
                    ) as req_span:
                    reply_chars = 0
                    pipeline = TurnPipeline(engine, call_id, vapi_messages=messages)
                    try:
                        ttft_stashed = False
                        async for token in pipeline.stream_tokens():
                            reply_chars += len(token)
                            # Stash llm_ttft_ms + filler_injected on the first token so the
                            # speech-update(assistant/started) handler finds them even if it
                            # fires before the streaming loop finishes.
                            if not ttft_stashed and pipeline.ttft_ms is not None:
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
async def delete_call(
    request: Request, call_id: str, _: None = Depends(_require_api_auth)
) -> dict[str, str]:
    """Cancel an in-flight call. Fires DELETE to Vapi if the call is still active."""
    request.state.call_id = call_id

    ended_at = datetime.now(timezone.utc)
    vapi_call_id: str | None = None
    with state.session_scope(engine) as session:
        stmt = (
            update(state.Call)
            .where(
                state.Call.id == call_id,
                state.Call.status.in_(["pending", "active"]),
            )
            .values(status="ended", end_reason="deleted", ended_at=ended_at)
            .returning(state.Call.vapi_call_id)
        )
        row = session.execute(stmt).fetchone()
        if row is not None:
            vapi_call_id = row[0]
        else:
            if session.get(state.Call, call_id) is None:
                raise HTTPException(status_code=404, detail="Call not found")
            return {"call_id": call_id, "status": "already_ended"}
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


def _build_speech_timeout_hooks() -> list[dict[str, Any]]:
    """Vapi-side dead-air ladder via `customer.speech.timeout` hooks.

    Escalates on continuous user silence: re-prompt at 1x and 2x the base interval,
    then warmly end the call at 3x. `triggerResetMode: onUserSpeech` clears the whole
    ladder the moment the user speaks. Returns [] (no hooks) when disabled.

    This owns true dead-air. Thinking fillers ("um", "let me think") are real
    transcribed speech and are handled by the interviewer model, not here.
    """
    base = settings.vapi_silence_timeout_seconds
    if base <= 0:
        return []
    return [
        {
            "on": "customer.speech.timeout",
            "options": {"timeoutSeconds": base, "triggerMaxCount": 3, "triggerResetMode": "onUserSpeech"},
            "do": [{"type": "say", "exact": "Take your time."}],
        },
        {
            "on": "customer.speech.timeout",
            "options": {"timeoutSeconds": base * 2, "triggerMaxCount": 3, "triggerResetMode": "onUserSpeech"},
            "do": [{"type": "say", "exact": "Still there?"}],
        },
        {
            "on": "customer.speech.timeout",
            "options": {"timeoutSeconds": base * 3, "triggerMaxCount": 1, "triggerResetMode": "onUserSpeech"},
            "do": [
                {
                    "type": "say",
                    "exact": "It sounds like now might not be a great time — thanks so much for your time today. I'll let you go.",
                },
                {"type": "tool", "tool": {"type": "endCall"}},
            ],
        },
    ]


async def _dial_vapi(call_id: str, phone_number: str) -> None:
    """POST to Vapi's outbound call API (background task). Never raises to the HTTP client.

    Preconditions: row has dial_status=queued and VAPI_PHONE_NUMBER_ID + WEBHOOK_URL were set at enqueue.
    """
    if not settings.vapi_phone_number_id or not settings.webhook_url:
        logfire.warning(
            "vapi_dial_skipped",
            reason="VAPI_PHONE_NUMBER_ID or WEBHOOK_URL not set",
            call_id=call_id,
        )
        _mark_dial_failed(
            call_id,
            end_reason="dial_skipped",
            dial_error="VAPI_PHONE_NUMBER_ID or WEBHOOK_URL became unset before dial",
        )
        return

    try:
        with state.session_scope(engine) as session:
            flip = session.execute(
                update(state.Call)
                .where(
                    state.Call.id == call_id,
                    state.Call.dial_status == state.DIAL_QUEUED,
                    state.Call.status == "pending",
                )
                .values(dial_status=state.DIAL_DIALING)
            )
            if flip.rowcount != 1:
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
                "transcriber": {
                    "provider": "deepgram",
                    "model": "flux-general-en",
                    "language": "en",
                    "eotThreshold": 0.7,
                    "eotTimeoutMs": 4500,
                },
                "startSpeakingPlan": _build_start_speaking_plan(),
                "stopSpeakingPlan": _build_stop_speaking_plan(),
                **({"hooks": _hooks} if (_hooks := _build_speech_timeout_hooks()) else {}),
                "firstMessage": (
                    "Hey, thank you for getting on the call! Want to check if you can hear me before we get started."
                ),
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
            _mark_dial_failed(
                call_id,
                end_reason="dial_failed",
                dial_error=f"HTTP {resp.status_code}: {resp.text[:2000]}",
            )
            return

        vapi_call_id = resp.json().get("id")
        if not vapi_call_id:
            _mark_dial_failed(
                call_id,
                end_reason="dial_failed",
                dial_error="Vapi response JSON missing id",
            )
            return

        with state.session_scope(engine) as session:
            persist = session.execute(
                update(state.Call)
                .where(
                    state.Call.id == call_id,
                    state.Call.dial_status == state.DIAL_DIALING,
                    state.Call.status == "pending",
                )
                .values(vapi_call_id=vapi_call_id, dial_status=state.DIAL_DIALED)
            )
            if persist.rowcount != 1:
                logfire.warning(
                    "vapi_dial_id_not_persisted",
                    call_id=call_id,
                    vapi_call_id=vapi_call_id,
                )
                if settings.vapi_api_key:
                    try:
                        async with httpx.AsyncClient() as hc:
                            await hc.delete(
                                f"https://api.vapi.ai/call/{vapi_call_id}",
                                headers={"Authorization": f"Bearer {settings.vapi_api_key}"},
                                timeout=5,
                            )
                    except Exception:
                        logfire.exception("vapi_dial_orphan_delete_failed", vapi_call_id=vapi_call_id)
                return

        logfire.info("vapi_dial_initiated", call_id=call_id, vapi_call_id=vapi_call_id, phone_number=phone_number)
    except Exception as exc:
        logfire.exception("vapi_dial_exception", call_id=call_id)
        _mark_dial_failed(call_id, end_reason="dial_failed", dial_error=str(exc)[:2000])


# --- Dev entrypoint ---------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("voice_agent.server:app", host="0.0.0.0", port=8000, reload=True)
