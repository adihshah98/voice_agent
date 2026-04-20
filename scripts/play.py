"""Interactive E2E playground — you play the respondent.

Runs the full pipeline each turn: analyst (background) + interviewer (foreground),
then synthesis at the end. Uses the same scripted questions as the trajectory evals.

Usage:
    uv run python play.py
    uv run python play.py --call-id my-test-1

Traces:
    LOGFIRE_TOKEN set  → logfire.pydantic.dev (no console trace tree)
    LOGFIRE_TOKEN unset → Logfire pretty-print on console only
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import uuid
import argparse

from dotenv import load_dotenv

load_dotenv()

from sqlalchemy.pool import StaticPool
from sqlmodel import create_engine

from voice_agent import state
from voice_agent.agents.analyst import run_analyst_safely
from voice_agent.agents.interviewer import run_interviewer_with_timeout
from voice_agent.config import INTERVIEWER_BUDGET_S
from voice_agent.models import AnalystDeps, InterviewerDeps
from voice_agent.agents.synthesis import SynthesisDeps, run_synthesis_safely
from voice_agent.tracing import init_tracing, turn_span, log_interviewer_decision

SCRIPTED_QUESTIONS = [
    "How do you currently use this product?",
    "What do you value most about it?",
    "Has anything frustrated you about it recently?",
    "Would you recommend it to someone else?",
    "What would make you use it more often?",
]


def _make_engine():
    return create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


def _seed(engine, call_id: str) -> None:
    with state.session_scope(engine) as s:
        s.add(
            state.Call(
                id=call_id,
                scripted_questions=SCRIPTED_QUESTIONS,
                status="active",
            )
        )


async def _run(call_id: str) -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set — check .env", file=sys.stderr)
        sys.exit(2)

    engine = _make_engine()
    state.init_db(engine)
    _seed(engine, call_id)

    print(f"\n  call_id : {call_id}")
    print(f"  questions: {len(SCRIPTED_QUESTIONS)} scripted")
    print("  Type your reply and press Enter. Ctrl-D or empty line to end early.\n")
    print("─" * 60)

    # Open with the first scripted question
    with state.session_scope(engine) as s:
        opening = state.next_scripted(s, call_id)
        state.mark_scripted_asked(s, call_id)
        s.add(
            state.Turn(
                call_id=call_id,
                turn_number=1,
                speaker="interviewer",
                text=opening,
                action="scripted",
            )
        )

    print(f"\nInterviewer: {opening}\n")
    turn_number = 2

    while True:
        try:
            line = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break

        # Analyst runs in background (fire-and-forget, same as prod)
        async def _analyst():
            with state.session_scope(engine) as s:
                await run_analyst_safely(AnalystDeps(call_id=call_id, session=s))

        analyst_task = asyncio.create_task(_analyst())

        with state.session_scope(engine) as s:
            s.add(
                state.Turn(
                    call_id=call_id,
                    turn_number=turn_number,
                    speaker="respondent",
                    text=line,
                )
            )

        # Interviewer (foreground, same 1.8 s budget as prod)
        with state.session_scope(engine) as s:
            deps = InterviewerDeps(
                call_id=call_id, session=s, turn_number=turn_number + 1
            )
            with turn_span(call_id, turn_number, respondent_text=line):
                t0 = time.perf_counter()
                out = await run_interviewer_with_timeout(deps, line, budget_s=INTERVIEWER_BUDGET_S)
                latency_ms = int((time.perf_counter() - t0) * 1000)
                log_interviewer_decision(
                    call_id=call_id,
                    turn_number=turn_number + 1,
                    action=out.action,
                    utterance=out.utterance,
                    reasoning=out.reasoning,
                    latency_ms=latency_ms,
                )
            s.add(
                state.Turn(
                    call_id=call_id,
                    turn_number=turn_number + 1,
                    speaker="interviewer",
                    text=out.utterance,
                    action=out.action,
                    reasoning=out.reasoning,
                    latency_ms=latency_ms,
                )
            )

        print(f"\nInterviewer: {out.utterance}")
        print(f"  [{out.action} · {latency_ms} ms · {out.reasoning}]\n")

        turn_number += 2

        if out.action == "wrap_up":
            break

    # Synthesis
    print("─" * 60)
    print("Generating synthesis report…")
    with state.session_scope(engine) as s:
        report = await run_synthesis_safely(SynthesisDeps(call_id=call_id, session=s))

    if report:
        print(f"\nSummary:\n  {report.summary}")
        if report.themes:
            print(f"\nThemes:\n  " + "\n  ".join(f"• {t}" for t in report.themes))
        if report.key_quotes:
            print(f"\nKey quotes:\n  " + "\n  ".join(f'"{q}"' for q in report.key_quotes))
        if report.contradictions:
            print(f"\nContradictions:\n  " + "\n  ".join(f"• {c}" for c in report.contradictions))
        if report.follow_up_questions:
            print(f"\nFollow-ups:\n  " + "\n  ".join(f"• {q}" for q in report.follow_up_questions))
    else:
        print("(No synthesis report generated.)")

    print(f"\n  call_id for traces: {call_id}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive voice-agent playground")
    parser.add_argument("--call-id", default=None, help="Stable call ID (default: random UUID)")
    args = parser.parse_args()

    call_id = args.call_id or f"play-{uuid.uuid4().hex[:8]}"
    init_tracing(service_name="voice-agent-play")
    asyncio.run(_run(call_id))


if __name__ == "__main__":
    main()
