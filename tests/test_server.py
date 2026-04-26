"""Unit tests for server streaming edge cases."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from voice_agent import server
from voice_agent.turn import StreamTurnResult


def _call_body(vapi_call_id: str, call_id: str, content: str, stream: bool = True) -> dict:
    return {
        "stream": stream,
        "call": {
            "id": vapi_call_id,
            "assistant": {"metadata": {"call_id": call_id}},
        },
        "messages": [{"role": "user", "content": content}],
    }


@pytest.mark.asyncio
async def test_vapi_stream_cancelled_error_is_handled(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakePipeline:
        def __init__(self, engine, call_id, vapi_messages=None):
            assert call_id == "call-123"
            assert vapi_messages is not None

        async def stream_tokens(self):
            yield "hello"
            raise asyncio.CancelledError()

        async def commit(self):
            raise AssertionError("commit() should not be called after CancelledError")

    monkeypatch.setattr(server, "TurnPipeline", FakePipeline)

    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/vapi/llm/chat/completions",
            json=_call_body("vapi-call-1", "call-123", "hello there"),
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"content": "hello"' in response.text
    assert "[DONE]" not in response.text


@pytest.mark.asyncio
async def test_vapi_stream_completes_with_done_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakePipeline:
        def __init__(self, engine, call_id, vapi_messages=None):
            assert call_id == "call-456"
            assert vapi_messages is not None

        async def stream_tokens(self):
            yield "partial "
            yield "reply"

        async def commit(self):
            return StreamTurnResult(
                action="probe",
                reasoning="ok",
                llm_latency_ms=123,
                ttft_ms=None,
                persist_ms=5,
                should_run_analyst=False,
            )

    monkeypatch.setattr(server, "TurnPipeline", FakePipeline)

    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/vapi/llm/chat/completions",
            json=_call_body("vapi-call-2", "call-456", "hi again"),
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"content": "partial "' in response.text
    assert '"content": "reply"' in response.text
    assert '"finish_reason": "stop"' in response.text
    assert "data: [DONE]" in response.text
