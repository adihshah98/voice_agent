"""Confirm tracing.py wires up Logfire and emits spans we can inspect."""

from __future__ import annotations

from logfire.testing import CaptureLogfire

from voice_agent.tracing import agent_span, init_tracing


def test_init_tracing_idempotent(capfire: CaptureLogfire) -> None:
    init_tracing(send_to_logfire=False, console=False)
    init_tracing(send_to_logfire=False, console=False)  # second call is a no-op
    # No crash, no duplicate configure — that's the whole assertion.


def test_agent_span_attributes(capfire: CaptureLogfire) -> None:
    init_tracing(send_to_logfire=False, console=False)
    with agent_span("interviewer", call_id="c1", model="claude-sonnet-4-6"):
        pass

    spans = capfire.exporter.exported_spans_as_dict()
    agent = next(s for s in spans if s["name"] == "interviewer_run")
    assert agent["attributes"]["agent"] == "interviewer"
    assert agent["attributes"]["model"] == "claude-sonnet-4-6"


def test_interviewer_span_records_decision(capfire: CaptureLogfire) -> None:
    init_tracing(send_to_logfire=False, console=False)
    with agent_span("interviewer", call_id="c2", turn_number=1, respondent_text="hello") as span:
        span.set_attribute("action", "probe")
        span.set_attribute("utterance", "Why specifically?")
        span.set_attribute("reasoning", "respondent mentioned lost trust")
        span.set_attribute("latency_ms", 842)

    spans = capfire.exporter.exported_spans_as_dict()
    agent = next(s for s in spans if s["name"] == "interviewer_run")
    assert agent["attributes"]["respondent_text"] == "hello"
    assert agent["attributes"]["action"] == "probe"
    assert agent["attributes"]["latency_ms"] == 842
