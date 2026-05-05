"""Unit tests for server streaming edge cases."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import httpx
import pytest
import pytest_mock

from voice_agent import server, state
from voice_agent.turn import StreamTurnResult


def _vapi_llm_body(vapi_call_id: str, call_id: str, content: str) -> server.VapiLLMBody:
    return server.VapiLLMBody.model_validate({
        "stream": True,
        "call": {
            "id": vapi_call_id,
            "assistant": {"metadata": {"call_id": call_id}},
        },
        "messages": [{"role": "user", "content": content}],
    })


async def _collect_sse(response) -> str:
    """Drain a StreamingResponse body_iterator into a single string."""
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk if isinstance(chunk, str) else chunk.decode())
    return "".join(chunks)


@pytest.mark.asyncio
async def test_vapi_stream_cancelled_error_is_handled(mocker: pytest_mock.MockerFixture) -> None:
    eng = state.make_engine("sqlite:///:memory:")
    state.init_db(eng)
    mocker.patch.object(server, "engine", eng)
    mocker.patch.object(server.settings, "vapi_webhook_secret", "")
    mocker.patch.object(server.settings, "llm_secret_token", "")
    with state.session_scope(eng) as session:
        session.add(state.Call(id="call-123", scripted_questions=[], status="pending"))

    class FakePipeline:
        filler_injected = False
        ttft_ms = None

        def __init__(self, engine, call_id, vapi_messages=None):
            assert call_id == "call-123"
            assert vapi_messages is not None

        async def stream_tokens(self):
            yield "hello"
            raise asyncio.CancelledError()

        async def commit(self):
            raise AssertionError("commit() should not be called after CancelledError")

    mocker.patch.object(server, "TurnPipeline", FakePipeline)

    req = MagicMock()
    req.state = MagicMock()
    response = await server.vapi_llm(req, _vapi_llm_body("vapi-call-1", "call-123", "hello there"), None)
    body = await _collect_sse(response)

    assert response.media_type == "text/event-stream"
    assert '"content": "hello"' in body
    assert "[DONE]" not in body


@pytest.mark.asyncio
async def test_vapi_stream_completes_with_done_chunk(mocker: pytest_mock.MockerFixture) -> None:
    eng = state.make_engine("sqlite:///:memory:")
    state.init_db(eng)
    mocker.patch.object(server, "engine", eng)
    mocker.patch.object(server.settings, "vapi_webhook_secret", "")
    mocker.patch.object(server.settings, "llm_secret_token", "")
    with state.session_scope(eng) as session:
        session.add(state.Call(id="call-456", scripted_questions=[], status="pending"))

    class FakePipeline:
        filler_injected = False
        ttft_ms = None

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

    mocker.patch.object(server, "TurnPipeline", FakePipeline)

    req = MagicMock()
    req.state = MagicMock()
    response = await server.vapi_llm(req, _vapi_llm_body("vapi-call-2", "call-456", "hi again"), None)
    body = await _collect_sse(response)

    assert response.media_type == "text/event-stream"
    assert '"content": "partial "' in body
    assert '"content": "reply"' in body
    assert '"finish_reason": "stop"' in body
    assert "data: [DONE]" in body


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
async def test_status_update_in_progress_is_idempotent(mocker: pytest_mock.MockerFixture) -> None:
    eng = state.make_engine("sqlite:///:memory:")
    state.init_db(eng)
    mocker.patch.object(server, "engine", eng)
    mocker.patch.object(server.settings, "vapi_webhook_secret", "")

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
async def test_end_of_call_report_schedules_synthesis_once(mocker: pytest_mock.MockerFixture) -> None:
    eng = state.make_engine("sqlite:///:memory:")
    state.init_db(eng)
    mocker.patch.object(server, "engine", eng)
    mocker.patch.object(server.settings, "vapi_webhook_secret", "")
    mocker.patch.object(server, "ENABLE_SYNTHESIS_REPORT", True)

    synthesis_calls: list[str] = []

    async def _track_synthesis(cid: str) -> None:
        synthesis_calls.append(cid)

    mocker.patch.object(server, "_synthesis_task", _track_synthesis)

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
async def test_delete_call_twice_second_is_already_ended(mocker: pytest_mock.MockerFixture) -> None:
    eng = state.make_engine("sqlite:///:memory:")
    state.init_db(eng)
    mocker.patch.object(server, "engine", eng)
    mocker.patch.object(server.settings, "api_auth_token", "tok")
    mocker.patch.object(server.settings, "vapi_api_key", "")

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
async def test_extended_silence_marks_call_ended(mocker: pytest_mock.MockerFixture) -> None:
    eng = state.make_engine("sqlite:///:memory:")
    state.init_db(eng)
    mocker.patch.object(server, "engine", eng)
    mocker.patch.object(server.settings, "vapi_webhook_secret", "")
    mocker.patch.object(server.settings, "vapi_extended_silence_seconds", 0.05)
    mocker.patch.object(server.settings, "vapi_api_key", "")

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
async def test_extended_silence_cancelled_when_user_speaks(mocker: pytest_mock.MockerFixture) -> None:
    eng = state.make_engine("sqlite:///:memory:")
    state.init_db(eng)
    mocker.patch.object(server, "engine", eng)
    mocker.patch.object(server.settings, "vapi_webhook_secret", "")
    mocker.patch.object(server.settings, "vapi_extended_silence_seconds", 0.2)
    mocker.patch.object(server.settings, "vapi_api_key", "")

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
    mocker: pytest_mock.MockerFixture,
) -> None:
    eng = state.make_engine("sqlite:///:memory:")
    state.init_db(eng)
    mocker.patch.object(server, "engine", eng)
    mocker.patch.object(server.settings, "api_auth_token", "test-bearer-secret")
    mocker.patch.object(server.settings, "vapi_api_key", "")

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
        assert r.json()["dial_status"] is None
        call_id = r.json()["call_id"]

        g = await client.get(
            f"/calls/{call_id}",
            headers={"Authorization": "Bearer test-bearer-secret"},
        )
        assert g.status_code == 200
        assert g.json()["call_id"] == call_id
        assert g.json()["status"] == "pending"
        assert g.json()["dial_status"] is None

        d = await client.delete(f"/calls/{call_id}")
        assert d.status_code == 401

        d = await client.delete(
            f"/calls/{call_id}",
            headers={"Authorization": "Bearer test-bearer-secret"},
        )
        assert d.status_code == 200
        assert d.json()["status"] == "ended"


@pytest.mark.asyncio
async def test_calls_start_async_dial_returns_202(mocker: pytest_mock.MockerFixture) -> None:
    eng = state.make_engine("sqlite:///:memory:")
    state.init_db(eng)
    mocker.patch.object(server, "engine", eng)
    mocker.patch.object(server.settings, "api_auth_token", "")
    mocker.patch.object(server.settings, "vapi_api_key", "fake-key")
    mocker.patch.object(server.settings, "vapi_phone_number_id", "phone-id")
    mocker.patch.object(server.settings, "webhook_url", "https://example.com")

    async def _noop_dial(_cid: str, _phone: str) -> None:
        return

    mocker.patch.object(server, "_dial_vapi", _noop_dial)

    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/calls/start",
            json={"scripted_questions": ["q1"], "phone_number": "+15551234567"},
        )
    assert r.status_code == 202
    assert r.json()["dial_status"] == "queued"
    with state.session_scope(eng) as session:
        c = session.get(state.Call, r.json()["call_id"])
        assert c is not None
        assert c.dial_status == "queued"


@pytest.mark.asyncio
async def test_calls_start_sync_abort_when_dial_config_incomplete(
    mocker: pytest_mock.MockerFixture,
) -> None:
    eng = state.make_engine("sqlite:///:memory:")
    state.init_db(eng)
    mocker.patch.object(server, "engine", eng)
    mocker.patch.object(server.settings, "api_auth_token", "")
    mocker.patch.object(server.settings, "vapi_api_key", "fake-key")
    mocker.patch.object(server.settings, "vapi_phone_number_id", "")
    mocker.patch.object(server.settings, "webhook_url", "https://example.com")

    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/calls/start",
            json={"scripted_questions": ["q1"], "phone_number": "+15551234567"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["dial_status"] == "dial_failed"
    assert "dial_error" in body
    with state.session_scope(eng) as session:
        c = session.get(state.Call, body["call_id"])
        assert c is not None
        assert c.status == "ended"
        assert c.end_reason == "dial_skipped"
