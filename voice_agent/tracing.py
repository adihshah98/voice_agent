"""Logfire configuration + span helpers.

`init_tracing()` is the single entry point — webhook, eval, and replay all call
it. Core Logfire + pydantic/httpx are configured once; `instrument_fastapi` /
`instrument_sqlalchemy` run on the first `init_tracing` that supplies `app` or
`engine` so a prior call without them does not block later server/play wiring.
Optional deps are skipped when not installed.

Every span that matters should carry `call_id` + `turn_number` so Logfire can
slice by call or by turn. Use `agent_span(...)` to get that for free.
"""

from __future__ import annotations

from contextlib import contextmanager
from importlib.util import find_spec
from typing import Any, Iterator, Optional

import logfire
from pydantic_evals.online import (
    configure as configure_online_evals,
    EvaluationResult,
    EvaluatorContext,
    EvaluatorFailure,
)
from typing import Sequence
from voice_agent.config import settings

# Logfire.configure and pydantic/httpx are one-time; FastAPI and SQLAlchemy
# are attached on first `init_tracing` that passes `app` / `engine` so a
# `send_to_logfire=False` test init does not block later `instrument_fastapi`
# when `voice_agent.server` (or `scripts.play`) is imported in the same process.
_logfire_core = False
_fastapi_instrumented = False
_sqlalchemy_instrumented = False


def init_tracing(
    *,
    service_name: str = "voice-agent",
    app: Any | None = None,
    engine: Any | None = None,
    send_to_logfire: Optional[bool] = None,
    console: Optional[bool] = None,
) -> None:
    """Configure Logfire and attach auto-instrumentation.

    Safe to call multiple times: `logfire.configure` and pydantic/httpx are
    applied once. FastAPI and SQLAlchemy are each applied at most once, when
    `app` or `engine` is first provided. Optional deps are skipped if not
    installed.

    `send_to_logfire`:
      - None   → honor LOGFIRE_TOKEN env (default Logfire behaviour).
      - False  → force local-only (useful in CI/tests).
      - True   → require a token; raises if missing.

    `console`:
      - None   → off when exporting to Logfire (token set or send_to_logfire
        True); on when local-only (no export). Pass True/False to override.
    """
    global _logfire_core, _fastapi_instrumented, _sqlalchemy_instrumented

    if not _logfire_core:
        if find_spec("dotenv") is not None:
            from dotenv import load_dotenv
            load_dotenv()

        effective_send = send_to_logfire
        if effective_send is None and not settings.logfire_token:
            effective_send = False

        if send_to_logfire is False:
            exports_to_logfire = False
        elif send_to_logfire is True:
            exports_to_logfire = True
        else:
            exports_to_logfire = bool(settings.logfire_token)

        if console is None:
            console = not exports_to_logfire

        logfire.configure(
            service_name=service_name,
            send_to_logfire=effective_send,
            console=logfire.ConsoleOptions(min_log_level="debug") if console else False,
        )

        if find_spec("pydantic_ai") is not None:
            logfire.instrument_pydantic_ai()
        if find_spec("httpx") is not None:
            logfire.instrument_httpx()

        configure_online_evals(default_sample_rate=1.0, emit_otel_events=True)
        _logfire_core = True

    if not _fastapi_instrumented and app is not None and find_spec("fastapi") is not None:
        logfire.instrument_fastapi(app)
        _fastapi_instrumented = True

    if not _sqlalchemy_instrumented and engine is not None:
        logfire.instrument_sqlalchemy(engine=engine)
        _sqlalchemy_instrumented = True


# --- Span helpers ----------------------------------------------------------


@contextmanager
def agent_span(agent: str, call_id: str, **attrs: Any) -> Iterator[Any]:
    """Generic span for interviewer/analyst/synthesis runs.

    `agent` should be one of: "interviewer", "analyst", "synthesis".
    Callers may set_attribute on the yielded span to attach post-run data
    (e.g. action/utterance/reasoning/latency_ms for the interviewer).
    """
    with logfire.span(
        "{agent}_run",
        _span_name=f"{agent}_run",
        agent=agent,
        call_id=call_id,
        **attrs,
    ) as span:
        yield span


