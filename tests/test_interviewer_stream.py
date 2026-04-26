"""Streaming interviewer behavior tests."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

from voice_agent.agents import interviewer as interviewer_module
from voice_agent.models import InterviewerOutput


@pytest.mark.asyncio
async def test_interviewer_stream_tokens_and_output(monkeypatch: pytest.MonkeyPatch) -> None:
    final = InterviewerOutput(
        utterance="What happened next?",
        action="probe",
        reasoning="Follow-up warranted.",
    )

    class FakeStreamed:
        async def stream_output(self, debounce_by=None):
            yield SimpleNamespace(utterance="What ")
            yield SimpleNamespace(utterance="What happened")
            yield SimpleNamespace(utterance="What happened next?")

        async def get_output(self):
            return final

    class FakeRunStreamCtx:
        async def __aenter__(self):
            return FakeStreamed()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class FakeAgent:
        def run_stream(self, prompt_parts, deps=None):
            return FakeRunStreamCtx()

    monkeypatch.setattr(interviewer_module, "interviewer", FakeAgent())

    deps = SimpleNamespace(session=None, call_id="call-1", turn_number=1)
    prepared = interviewer_module.PreparedInterviewerTurn(
        prompt_parts=["prompt"],
        fallback_scripted_question=None,
    )

    stream = interviewer_module.InterviewerStream(deps, prepared)
    tokens = [tok async for tok in stream.tokens()]

    assert tokens == ["What ", "happened", " next?"]
    assert stream.output == final
