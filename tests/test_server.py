"""Unit tests for server streaming edge cases."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from voice_agent import server


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
    async def fake_run_speech_turn_stream(
        engine,
        call_id: str,
        respondent_text: str,
        vapi_messages: list[dict] | None = None,
    ):
        assert call_id == "call-123"
        assert respondent_text == "hello there"
        assert vapi_messages is not None
        yield "hello"
        raise asyncio.CancelledError()

    monkeypatch.setattr(server, "run_speech_turn_stream", fake_run_speech_turn_stream)

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
    async def fake_run_speech_turn_stream(
        engine,
        call_id: str,
        respondent_text: str,
        vapi_messages: list[dict] | None = None,
    ):
        assert call_id == "call-456"
        assert respondent_text == "hi again"
        assert vapi_messages is not None
        yield "partial "
        yield "reply"
        yield {"action": "probe", "reasoning": "ok", "latency_ms": 123}

    monkeypatch.setattr(server, "run_speech_turn_stream", fake_run_speech_turn_stream)

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
