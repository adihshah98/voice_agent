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
        filler_injected = False

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


def _status_in_progress_body(call_id: str, vapi_call_id: str) -> dict:
    return {
        "message": {
            "type": "status-update",
            "status": "in-progress",
            "call": {
                "id": vapi_call_id,
                "assistant": {"metadata": {"call_id": call_id}},
            },
        }
    }


def _end_of_call_report_body(call_id: str, vapi_call_id: str, ended_reason: str = "customer-ended-call") -> dict:
    return {
        "message": {
            "type": "end-of-call-report",
            "endedReason": ended_reason,
            "call": {
                "id": vapi_call_id,
                "assistant": {"metadata": {"call_id": call_id}},
            },
            "artifact": {},
        }
    }


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
async def test_status_update_in_progress_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    eng = state.make_engine("sqlite:///:memory:")
    state.init_db(eng)
    monkeypatch.setattr(server, "engine", eng)
    monkeypatch.setattr(server.settings, "vapi_webhook_secret", "")

    with state.session_scope(eng) as session:
        session.add(
            state.Call(
                id="c-status-dup",
                vapi_call_id="vapi-st1",
                scripted_questions=[],
                status="pending",
            )
        )

    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for _ in range(3):
            r = await client.post(
                "/vapi/webhook",
                json=_status_in_progress_body("c-status-dup", "vapi-st1"),
            )
            assert r.status_code == 200

    with state.session_scope(eng) as session:
        c = session.get(state.Call, "c-status-dup")
        assert c is not None
        assert c.status == "active"


@pytest.mark.asyncio
async def test_end_of_call_report_schedules_synthesis_once(monkeypatch: pytest.MonkeyPatch) -> None:
    eng = state.make_engine("sqlite:///:memory:")
    state.init_db(eng)
    monkeypatch.setattr(server, "engine", eng)
    monkeypatch.setattr(server.settings, "vapi_webhook_secret", "")
    monkeypatch.setattr(server, "ENABLE_SYNTHESIS_REPORT", True)

    synthesis_calls: list[str] = []

    async def _track_synthesis(cid: str) -> None:
        synthesis_calls.append(cid)

    monkeypatch.setattr(server, "_synthesis_task", _track_synthesis)

    with state.session_scope(eng) as session:
        session.add(
            state.Call(
                id="c-eoc-dup",
                vapi_call_id="vapi-eoc1",
                scripted_questions=[],
                status="active",
            )
        )

    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for _ in range(2):
            r = await client.post(
                "/vapi/webhook",
                json=_end_of_call_report_body("c-eoc-dup", "vapi-eoc1"),
            )
            assert r.status_code == 200

    await asyncio.sleep(0)
    assert synthesis_calls == ["c-eoc-dup"]
    with state.session_scope(eng) as session:
        c = session.get(state.Call, "c-eoc-dup")
        assert c is not None
        assert c.status == "ended"


@pytest.mark.asyncio
async def test_delete_call_twice_second_is_already_ended(monkeypatch: pytest.MonkeyPatch) -> None:
    eng = state.make_engine("sqlite:///:memory:")
    state.init_db(eng)
    monkeypatch.setattr(server, "engine", eng)
    monkeypatch.setattr(server.settings, "api_auth_token", "tok")
    monkeypatch.setattr(server.settings, "vapi_api_key", "")

    with state.session_scope(eng) as session:
        session.add(
            state.Call(
                id="c-del-dup",
                vapi_call_id="vapi-del1",
                scripted_questions=[],
                status="active",
            )
        )

    transport = httpx.ASGITransport(app=server.app)
    headers = {"Authorization": "Bearer tok"}
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r1 = await client.delete("/calls/c-del-dup", headers=headers)
        assert r1.status_code == 200
        assert r1.json()["status"] == "ended"
        r2 = await client.delete("/calls/c-del-dup", headers=headers)
        assert r2.status_code == 200
        assert r2.json()["status"] == "already_ended"


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


@pytest.mark.asyncio
async def test_calls_start_and_delete_require_bearer_when_api_auth_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eng = state.make_engine("sqlite:///:memory:")
    state.init_db(eng)
    monkeypatch.setattr(server, "engine", eng)
    monkeypatch.setattr(server.settings, "api_auth_token", "test-bearer-secret")
    monkeypatch.setattr(server.settings, "vapi_api_key", "")

    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post("/calls/start", json={"scripted_questions": ["q1"]})
        assert r.status_code == 401

        r = await client.post(
            "/calls/start",
            json={"scripted_questions": ["q1"]},
            headers={"Authorization": "Bearer wrong"},
        )
        assert r.status_code == 401

        r = await client.post(
            "/calls/start",
            json={"scripted_questions": ["q1"]},
            headers={"Authorization": "Bearer test-bearer-secret"},
        )
        assert r.status_code == 200
        call_id = r.json()["call_id"]

        d = await client.delete(f"/calls/{call_id}")
        assert d.status_code == 401

        d = await client.delete(
            f"/calls/{call_id}",
            headers={"Authorization": "Bearer test-bearer-secret"},
        )
        assert d.status_code == 200
        assert d.json()["status"] == "ended"
