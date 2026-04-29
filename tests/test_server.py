"""Unit tests for server streaming edge cases."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from voice_agent import server, state
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
    monkeypatch.setattr(server.settings, "vapi_webhook_secret", "")
    monkeypatch.setattr(server.settings, "llm_secret_token", "")

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
    monkeypatch.setattr(server.settings, "vapi_webhook_secret", "")
    monkeypatch.setattr(server.settings, "llm_secret_token", "")

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
                llm_usage=None,
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


def _speech_update_body(call_id: str, vapi_call_id: str, role: str, status: str) -> dict:
    return {
        "message": {
            "type": "speech-update",
            "status": status,
            "role": role,
            "turn": 1,
            "call": {
                "id": vapi_call_id,
                "assistant": {"metadata": {"call_id": call_id}},
            },
        }
    }


@pytest.mark.asyncio
async def test_extended_silence_marks_call_ended(monkeypatch: pytest.MonkeyPatch) -> None:
    eng = state.make_engine("sqlite:///:memory:")
    state.init_db(eng)
    monkeypatch.setattr(server, "engine", eng)
    monkeypatch.setattr(server.settings, "vapi_webhook_secret", "")
    monkeypatch.setattr(server.settings, "vapi_extended_silence_seconds", 0.05)
    monkeypatch.setattr(server.settings, "vapi_api_key", "")

    with state.session_scope(eng) as session:
        session.add(
            state.Call(
                id="c-silence",
                vapi_call_id="vapi-s1",
                scripted_questions=[],
                status="active",
            )
        )

    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/vapi/webhook", json=_speech_update_body("c-silence", "vapi-s1", "assistant", "stopped")
        )
    assert r.status_code == 200
    await asyncio.sleep(0.12)

    with state.session_scope(eng) as session:
        c = session.get(state.Call, "c-silence")
        assert c is not None
        assert c.status == "ended"
        assert c.end_reason == "extended_silence"


@pytest.mark.asyncio
async def test_extended_silence_cancelled_when_user_speaks(monkeypatch: pytest.MonkeyPatch) -> None:
    eng = state.make_engine("sqlite:///:memory:")
    state.init_db(eng)
    monkeypatch.setattr(server, "engine", eng)
    monkeypatch.setattr(server.settings, "vapi_webhook_secret", "")
    monkeypatch.setattr(server.settings, "vapi_extended_silence_seconds", 0.2)
    monkeypatch.setattr(server.settings, "vapi_api_key", "")

    with state.session_scope(eng) as session:
        session.add(
            state.Call(
                id="c-quiet",
                vapi_call_id="vapi-s2",
                scripted_questions=[],
                status="active",
            )
        )

    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post(
            "/vapi/webhook", json=_speech_update_body("c-quiet", "vapi-s2", "assistant", "stopped")
        )
        await asyncio.sleep(0.05)
        await client.post(
            "/vapi/webhook", json=_speech_update_body("c-quiet", "vapi-s2", "user", "started")
        )
        await asyncio.sleep(0.25)

    with state.session_scope(eng) as session:
        c = session.get(state.Call, "c-quiet")
        assert c is not None
        assert c.status == "active"
