"""Streaming interviewer behavior tests."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from pydantic_ai.usage import RunUsage

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("GROQ_API_KEY", "test-key")

from voice_agent.agents import interviewer as interviewer_module
from voice_agent.config import INTERVIEWER_RECOVERY_UTTERANCE
from voice_agent.models import InterviewerLLMMeta


@pytest.mark.asyncio
async def test_interviewer_stream_tokens_and_output(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate a model that correctly uses <utterance> tags
    raw_output = (
        "<utterance>What happened next?</utterance>\n"
        '{"action": "probe", "reasoning": "Follow-up warranted.", "probe_id_used": null}'
    )

    class FakeStreamed:
        async def stream_text(self, delta=False, debounce_by=None):
            # Yield in three chunks to exercise the carry-buffer logic
            yield "<utterance>What happened"
            yield " next?</utterance>"
            yield '\n{"action": "probe", "reasoning": "Follow-up warranted.", "probe_id_used": null}'

        def usage(self) -> RunUsage:
            return RunUsage(
                input_tokens=100,
                output_tokens=20,
                cache_read_tokens=0,
                cache_write_tokens=0,
            )

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

    assert "".join(tokens) == "What happened next?"
    assert stream.output.utterance == "What happened next?"
    assert stream.output.action == "probe"
    assert stream.output.reasoning == "Follow-up warranted."
    assert stream.output.probe_id_used is None
    assert stream.usage is not None
    assert stream.usage.input_tokens == 100
    assert stream.usage.output_tokens == 20


@pytest.mark.asyncio
async def test_interviewer_stream_bare_format(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bare format (no tags) — model omits <utterance> wrapper but still outputs JSON."""

    class FakeStreamed:
        async def stream_text(self, delta=False, debounce_by=None):
            yield "What happened next?"
            yield '\n{"action": "probe", "reasoning": "Follow-up warranted.", "probe_id_used": null}'

        def usage(self) -> RunUsage:
            return RunUsage(input_tokens=50, output_tokens=10, cache_read_tokens=0, cache_write_tokens=0)

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

    assert "".join(tokens) == "What happened next?"
    assert stream.output.utterance == "What happened next?"
    assert stream.output.action == "probe"
    assert stream.output.reasoning == "Follow-up warranted."


def test_parse_streamed_output_validates_with_pydantic() -> None:
    raw = (
        "<utterance>Hello there.</utterance>\n"
        '{"action":"scripted","reasoning":"ok","probe_id_used":null,"noise":1}'
    )
    out = interviewer_module._parse_streamed_output(raw, None)
    assert out.utterance == "Hello there."
    assert out.action == "scripted"
    assert out.reasoning == "ok"
    assert out.probe_id_used is None


def test_parse_streamed_output_invalid_action_falls_back_to_scripted() -> None:
    raw = (
        '<utterance>Hi.</utterance>\n'
        '{"action":"not_a_real_action","reasoning":"x","probe_id_used":null}'
    )
    out = interviewer_module._parse_streamed_output(raw, None)
    assert out.action == "scripted"
    assert "parse failed" in out.reasoning.lower()
    assert out.utterance == "Hi."


def test_parse_streamed_output_clears_probe_id_when_not_probe() -> None:
    raw = (
        "<utterance>Hi.</utterance>\n"
        '{"action":"scripted","reasoning":"x","probe_id_used":42}'
    )
    out = interviewer_module._parse_streamed_output(raw, None)
    assert out.action == "scripted"
    assert out.probe_id_used is None


def test_interviewer_llm_meta_model() -> None:
    m = InterviewerLLMMeta.model_validate({"action": "probe", "reasoning": "r", "probe_id_used": 7})
    assert m.action == "probe"
    assert m.probe_id_used == 7


def test_parse_streamed_output_empty_tagged_utterance_uses_recovery() -> None:
    raw = (
        "<utterance>   </utterance>\n"
        '{"action":"scripted","reasoning":"x","probe_id_used":null}'
    )
    out = interviewer_module._parse_streamed_output(raw, None)
    assert out.utterance == INTERVIEWER_RECOVERY_UTTERANCE
    assert out.reasoning == "empty utterance from model"


def test_parse_streamed_output_json_only_body_uses_recovery() -> None:
    raw = '{"action":"scripted","reasoning":"oops","probe_id_used":null}'
    out = interviewer_module._parse_streamed_output(raw, None)
    assert out.utterance == INTERVIEWER_RECOVERY_UTTERANCE
    assert "spoken utterance block" in out.reasoning


def test_parse_streamed_output_metadata_fail_empty_body_uses_recovery() -> None:
    raw = "<utterance></utterance>\n{not valid json"
    out = interviewer_module._parse_streamed_output(raw, None)
    assert out.utterance == INTERVIEWER_RECOVERY_UTTERANCE
    assert "metadata parse failed" in out.reasoning
