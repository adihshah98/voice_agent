"""Confirm tracing.py wires up Logfire and emits spans we can inspect."""

from __future__ import annotations

import logfire
from logfire.testing import CaptureLogfire

from tracing import agent_span, init_tracing, log_interviewer_decision, turn_span


def test_init_tracing_idempotent(capfire: CaptureLogfire) -> None:
    init_tracing(send_to_logfire=False, console=False)
    init_tracing(send_to_logfire=False, console=False)  # second call is a no-op
    # No crash, no duplicate configure — that's the whole assertion.


def test_turn_span_captures_attributes(capfire: CaptureLogfire) -> None:
    init_tracing(send_to_logfire=False, console=False)
    with turn_span(call_id="c1", turn_number=3, respondent_text="hello"):
        logfire.info("inside_turn", note="ok")

    spans = capfire.exporter.exported_spans_as_dict()
    turn = next(s for s in spans if s["name"] == "turn")
    assert turn["attributes"]["call_id"] == "c1"
    assert turn["attributes"]["turn_number"] == 3
    assert turn["attributes"]["respondent_text"] == "hello"


def test_agent_span_and_decision_log(capfire: CaptureLogfire) -> None:
    init_tracing(send_to_logfire=False, console=False)
    with agent_span("interviewer", call_id="c2", model="claude-sonnet-4-6"):
        log_interviewer_decision(
            call_id="c2",
            turn_number=1,
            action="probe",
            utterance="Why specifically?",
            reasoning="respondent mentioned lost trust",
            latency_ms=842,
        )

    spans = capfire.exporter.exported_spans_as_dict()
    agent = next(s for s in spans if s["name"] == "interviewer_run")
    assert agent["attributes"]["agent"] == "interviewer"
    assert agent["attributes"]["model"] == "claude-sonnet-4-6"

    decision = next(s for s in spans if s["name"] == "interviewer_decision")
    assert decision["attributes"]["action"] == "probe"
    assert decision["attributes"]["latency_ms"] == 842
