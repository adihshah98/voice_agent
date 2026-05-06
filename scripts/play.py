#!/usr/bin/env python3
"""play.py — terminal E2E simulation of a voice research interview.

Usage:
    uv run python scripts/play.py                          # generic terminal REPL
    uv run python scripts/play.py "Notion AI"              # substitute [product]
    uv run python scripts/play.py "Notion AI" --phone +14155551234  # live Vapi call

Loads data/investor_questions.yaml.

Local REPL mode drives the interviewer in-process against the same SQLite DB
the server uses — no running server required.

--phone mode hits a running server to dial out via Vapi:
    uv run uvicorn voice_agent.server:app --reload
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

load_dotenv()

from voice_agent.config import ENABLE_SYNTHESIS_REPORT, settings

_level_name = settings.log_level.upper()
_root_level = getattr(logging, _level_name, logging.INFO)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=_root_level,
        format="%(levelname)s %(name)s %(message)s",
    )
play_logger = logging.getLogger("voice_agent.play")

from sqlmodel import Session, select

from voice_agent import state
from voice_agent.agents.synthesis import SynthesisDeps, run_synthesis_safely
from voice_agent.tracing import agent_span, init_tracing
from voice_agent.turn import run_speech_turn

BASE_URL = "http://localhost:8000"


def _investor_questions_path() -> Path:
    here = Path(__file__).resolve().parent
    for root in [here.parent, *here.parents]:
        candidate = root / "data" / "investor_questions.yaml"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "data/investor_questions.yaml not found. "
        f"Searched upward from {here}. Run from the voice_agent repo clone."
    )


QUESTIONS_FILE = _investor_questions_path()


# --- Question loading -------------------------------------------------------


def load_questions(product: str | None) -> list[str]:
    raw = yaml.safe_load(QUESTIONS_FILE.read_text())
    questions = []
    for entry in raw["questions"]:
        q = entry["question"].strip()
        q = q.replace("[product]", product if product else "the product")
        questions.append(q)
    return questions


# --- In-process call ops ----------------------------------------------------


def seed_call(engine, questions: list[str]) -> str:
    call_id = str(uuid.uuid4())
    with state.session_scope(engine) as session:
        session.add(
            state.Call(
                id=call_id,
                scripted_questions=questions,
                status="active",
            )
        )
    return call_id


def mark_ended(engine, call_id: str) -> None:
    with state.session_scope(engine) as session:
        call = session.get(state.Call, call_id)
        if call and call.status != "ended":
            call.status = "ended"
            call.end_reason = "local-simulation"
            call.ended_at = datetime.now(timezone.utc)
            session.add(call)


async def run_synthesis_inproc(engine, call_id: str) -> None:
    session = Session(engine)
    try:
        with agent_span("synthesis", call_id):
            deps = SynthesisDeps(call_id=call_id, session=session)
            await run_synthesis_safely(deps)
    finally:
        session.close()


def load_report(engine, call_id: str) -> dict | None:
    with state.session_scope(engine) as session:
        report = session.exec(
            select(state.SynthesisReport).where(state.SynthesisReport.call_id == call_id)
        ).first()
        if report is None:
            return None
        return {
            "summary": report.summary,
            "themes": report.themes,
            "contradictions": report.contradictions,
            "key_quotes": report.key_quotes,
            "follow_up_questions": report.follow_up_questions,
        }


# --- Report display ---------------------------------------------------------


def print_report(report: dict) -> None:
    sep = "─" * 60
    print(f"\n{sep}")
    print("SYNTHESIS REPORT")
    print(sep)

    print(f"\nSUMMARY:\n{report.get('summary', '')}")

    themes = report.get("themes", [])
    if themes:
        print(f"\nTHEMES ({len(themes)}):")
        for t in themes:
            if isinstance(t, dict):
                print(f"  • {t.get('theme', t)}")
                for q in t.get("quotes", []):
                    print(f'      "{q}"')
            else:
                print(f"  • {t}")

    contradictions = report.get("contradictions", [])
    if contradictions:
        print("\nCONTRADICTIONS:")
        for c in contradictions:
            print(f"  • {c}")

    key_quotes = report.get("key_quotes", [])
    if key_quotes:
        print("\nKEY QUOTES:")
        for q in key_quotes:
            print(f'  "{q}"')

    follow_ups = report.get("follow_up_questions", [])
    if follow_ups:
        print("\nFOLLOW-UP QUESTIONS:")
        for q in follow_ups:
            print(f"  • {q}")

    print(sep)


# --- REPL -------------------------------------------------------------------



async def run_local_repl(questions: list[str]) -> None:
    engine = state.make_engine(settings.database_url)
    state.init_db(engine)
    init_tracing(engine=engine)

    call_id = seed_call(engine, questions)
    print(f"Call created: {call_id}")

    print('\nType respondent answers below. Enter "quit" or Ctrl-D to stop.')
    print("─" * 60)

    # messages mirrors what Vapi would send as body["messages"] — built locally
    messages: list[dict] = []

    opening = await run_speech_turn(engine, call_id, vapi_messages=messages)
    messages.append({"role": "assistant", "content": opening["message"]})
    print(f"\ninterviewer> {opening['message']}")
    print(f"  [ACTION: {opening['action']}]")
    print(f"  [WHY: {opening['reasoning']}]")

    loop = asyncio.get_running_loop()
    while True:
        try:
            user_input = (await loop.run_in_executor(None, input, "\nrespondent> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        result = await run_speech_turn(engine, call_id, vapi_messages=messages)
        messages.append({"role": "assistant", "content": result["message"]})

        print(f"\ninterviewer> {result['message']}")
        print(f"  [ACTION: {result['action']}]")
        print(f"  [WHY: {result['reasoning']}]")

        if result["action"] == "wrap_up":
            print("\n[wrap_up — ending call...]")
            break

    print("\nEnding call...")
    mark_ended(engine, call_id)

    if ENABLE_SYNTHESIS_REPORT:
        print("Running synthesis...")
        await run_synthesis_inproc(engine, call_id)
        report = load_report(engine, call_id)
        if report is None:
            print("Synthesis produced no report (see logs).")
        else:
            print_report(report)
    else:
        print("Synthesis disabled (ENABLE_SYNTHESIS_REPORT=false).")


# --- --phone mode: dial via server ------------------------------------------


def dial_via_server(questions: list[str], phone_number: str) -> None:
    import httpx

    call_id = str(uuid.uuid4())
    with httpx.Client() as client:
        try:
            client.get(f"{BASE_URL}/docs", timeout=3)
        except httpx.ConnectError:
            play_logger.error("Cannot connect to server at %s", BASE_URL)
            play_logger.error("Start it with: uv run uvicorn voice_agent.server:app --reload")
            sys.exit(1)

        headers: dict[str, str] = {}
        if settings.api_auth_token:
            headers["Authorization"] = f"Bearer {settings.api_auth_token}"

        resp = client.post(
            f"{BASE_URL}/calls/start",
            json={
                "scripted_questions": questions,
                "call_id": call_id,
                "phone_number": phone_number,
            },
            headers=headers,
            timeout=15,
        )
        if not resp.is_success:
            play_logger.error(
                "POST /calls/start failed status=%s body=%s",
                resp.status_code,
                resp.text,
            )
        resp.raise_for_status()
        body = resp.json()
        if body.get("dial_status") == "dial_failed":
            play_logger.error("Dial did not start: %s", body.get("dial_error", ""))
            sys.exit(1)

    play_logger.info("Call created call_id=%s dial_status=%s", call_id, body.get("dial_status"))
    if resp.status_code == 202:
        play_logger.info(
            "Outbound dial queued — poll GET %s/calls/%s for dial_status / vapi_call_id",
            BASE_URL,
            call_id,
        )
    elif body.get("dial_status") is None:
        play_logger.warning(
            "No outbound dial (set VAPI_API_KEY and dial env) — call_id=%s was created for %s",
            call_id,
            phone_number,
        )
    play_logger.info(
        "After the call, synthesis report: curl %s/calls/%s/report",
        BASE_URL,
        call_id,
    )


# --- Entrypoint -------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Terminal E2E voice interview simulation")
    parser.add_argument("product", nargs="?", help="Product name to substitute in questions (optional)")
    parser.add_argument("--phone", metavar="NUMBER", help="Dial this number via Vapi (e.g. +14155551234)")
    args = parser.parse_args()

    product: str | None = args.product or None
    questions = load_questions(product)
    label = f"'{product}'" if product else "generic (no product)"
    print(f"\nLoaded {len(questions)} scripted questions — {label}")

    if args.phone:
        dial_via_server(questions, args.phone)
        return

    asyncio.run(run_local_repl(questions))


if __name__ == "__main__":
    main()
