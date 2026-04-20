"""Logfire configuration + span helpers.

`init_tracing()` is the single entry point — webhook/eval/replay all call it
once at startup. Instrumentation for optional deps (fastapi, pydantic_ai, etc.)
is attempted best-effort so this module is safe to import before those
packages land.

Every span that matters should carry `call_id` + `turn_number` so Logfire can
slice by call or by turn. Use `turn_span(...)` / `agent_span(...)` to get that
for free.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from importlib.util import find_spec
from typing import Any, Iterator, Optional

import logfire


_configured = False


def init_tracing(
    *,
    service_name: str = "voice-agent",
    app: Any | None = None,
    engine: Any | None = None,
    send_to_logfire: Optional[bool] = None,
    console: bool = True,
) -> None:
    """Configure Logfire and attach auto-instrumentation.

    Idempotent — safe to call from multiple entrypoints. Instrumentation for
    optional deps is skipped silently when the dep isn't installed so this
    doesn't break early in the build order.

    `send_to_logfire`:
      - None   → honor LOGFIRE_TOKEN env (default Logfire behaviour).
      - False  → force local-only (useful in CI/tests).
      - True   → require a token; raises if missing.
    """
    global _configured
    if _configured:
        return

    if find_spec("dotenv") is not None:
        from dotenv import load_dotenv
        load_dotenv()

    effective_send = send_to_logfire
    if effective_send is None and not os.getenv("LOGFIRE_TOKEN"):
        effective_send = False

    logfire.configure(
        service_name=service_name,
        send_to_logfire=effective_send,
        console=logfire.ConsoleOptions() if console else False,
    )

    if find_spec("pydantic_ai") is not None:
        logfire.instrument_pydantic_ai()
    if find_spec("httpx") is not None:
        logfire.instrument_httpx()
    if app is not None and find_spec("fastapi") is not None:
        logfire.instrument_fastapi(app)
    if engine is not None:
        logfire.instrument_sqlalchemy(engine=engine)

    _configured = True


# --- Span helpers ----------------------------------------------------------


@contextmanager
def turn_span(call_id: str, turn_number: int, respondent_text: str | None = None) -> Iterator[Any]:
    """One span per Vapi webhook turn. Wrap the interviewer run inside."""
    with logfire.span(
        "turn",
        call_id=call_id,
        turn_number=turn_number,
        respondent_text=respondent_text,
    ) as span:
        yield span


@contextmanager
def agent_span(agent: str, call_id: str, **attrs: Any) -> Iterator[Any]:
    """Generic span for interviewer/analyst/synthesis runs.

    `agent` should be one of: "interviewer", "analyst", "synthesis".
    """
    with logfire.span(
        "{agent}_run",
        _span_name=f"{agent}_run",
        agent=agent,
        call_id=call_id,
        **attrs,
    ) as span:
        yield span


def log_interviewer_decision(
    *,
    call_id: str,
    turn_number: int,
    action: str,
    utterance: str,
    reasoning: str,
    latency_ms: int,
) -> None:
    logfire.info(
        "interviewer_decision",
        call_id=call_id,
        turn_number=turn_number,
        action=action,
        utterance=utterance,
        reasoning=reasoning,
        latency_ms=latency_ms,
    )
