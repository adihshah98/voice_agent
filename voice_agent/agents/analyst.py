"""Analyst agent — structured output, fire-and-forget.

Reads all turns for a call, produces themes/contradictions/surprises/probes,
and persists an AnalystSnapshot + new Probe rows.

Never raises into the live call path — use `run_analyst_safely` from server.py.
"""

from __future__ import annotations

import time

import logfire
from dotenv import load_dotenv
from pydantic_ai import Agent
from sqlmodel import select

load_dotenv()

from voice_agent import state
from voice_agent.config import ANALYST_MODEL
from voice_agent.models import AnalysisUpdate, AnalystDeps


ANALYST_PROMPT = """\
You are a qualitative research analyst reviewing a live interview transcript.

Your job:
1. THEMES — recurring ideas or patterns in the respondent's answers.
2. CONTRADICTIONS — places where the respondent says things that conflict with each other.
3. SURPRISES — answers that were unexpected or reveal something the interviewer didn't anticipate.
4. NEW_PROBES — follow-up questions worth asking to dig deeper.

Probe guidelines:
- Reference specifics from the transcript (direct quotes, named things, moments).
- Use open, neutral phrasing — never leading questions.
- Priority 1 = real contradiction or major surprise worth asking immediately.
- Priority 2 = interesting thread worth exploring if time allows.
- Priority 3 = nice-to-have depth.
- Maximum 3 new probes per pass.
- Never repeat a question already in EXISTING_PROBES.

Be concise — bullet-point style for themes/contradictions/surprises.
"""


def _build_prompt(session, call_id: str) -> tuple[str, int]:
    """Return (prompt_text, last_turn_number)."""
    turns = state.recent_turns(session, call_id, n=200)
    if not turns:
        return "(no turns yet)", 0

    lines = []
    for t in turns:
        lines.append(f"{t.speaker.upper()} [{t.turn_number}]: {t.text}")
    transcript = "\n".join(lines)

    existing_stmt = select(state.Probe).where(state.Probe.call_id == call_id)
    existing = [p.question for p in session.exec(existing_stmt)]

    prompt = f"TRANSCRIPT:\n{transcript}"
    if existing:
        prompt += "\n\nEXISTING_PROBES (do not repeat):\n" + "\n".join(
            f"- {q}" for q in existing
        )

    return prompt, turns[-1].turn_number


analyst = Agent(
    ANALYST_MODEL,
    deps_type=AnalystDeps,
    output_type=AnalysisUpdate,
    system_prompt=ANALYST_PROMPT,
    instrument=True,
)


async def run_analyst(deps: AnalystDeps) -> AnalysisUpdate:
    """Read transcript → one LLM call → persist AnalystSnapshot + Probes."""
    t0 = time.perf_counter()

    prompt, after_turn = _build_prompt(deps.session, deps.call_id)
    result = await analyst.run(prompt, deps=deps)
    update: AnalysisUpdate = result.output

    latency_ms = int((time.perf_counter() - t0) * 1000)

    deps.session.add(
        state.AnalystSnapshot(
            call_id=deps.call_id,
            after_turn=after_turn,
            themes=update.themes,
            contradictions=update.contradictions,
            surprises=update.surprises,
            latency_ms=latency_ms,
        )
    )
    for np in update.new_probes:
        deps.session.add(
            state.Probe(
                call_id=deps.call_id,
                question=np.question,
                priority=np.priority,
                rationale=np.rationale,
                generated_after_turn=after_turn,
            )
        )
    deps.session.commit()

    logfire.info(
        "analyst_update",
        call_id=deps.call_id,
        after_turn=after_turn,
        themes_count=len(update.themes),
        contradictions_count=len(update.contradictions),
        surprises_count=len(update.surprises),
        new_probes_count=len(update.new_probes),
        latency_ms=latency_ms,
    )

    return update


async def run_analyst_safely(deps: AnalystDeps) -> AnalysisUpdate | None:
    """Swallow exceptions so a crashing analyst never affects the live call."""
    try:
        return await run_analyst(deps)
    except Exception:
        logfire.exception("analyst_error", call_id=deps.call_id)
        return None


# --- Smoke test: feed a canned transcript, confirm DB writes ---------------


def _seed_canned_transcript(session, call_id: str) -> None:
    session.add(
        state.Call(
            id=call_id,
            phone_number="+15550100",
            scripted_questions=[
                "How do you currently use the product?",
                "What would you change about it?",
                "Who else might benefit from it?",
            ],
            status="active",
        )
    )
    turns = [
        ("interviewer", 1, "How do you currently use the product?", "scripted"),
        ("respondent", 2, "I use it every morning to plan my commute. It's usually great.", None),
        ("interviewer", 3, "What would you change about it?", "scripted"),
        ("respondent", 4, "Honestly nothing — I think it's perfect as-is. Well, except the alerts are always wrong.", None),
        ("interviewer", 5, "Who else might benefit from it?", "scripted"),
        ("respondent", 6, "My whole team, probably. But actually I stopped recommending it because it gave bad directions last week.", None),
        ("interviewer", 7, "That sounds frustrating — what happened exactly?", "probe"),
        ("respondent", 8, "It routed me through a closed road. Cost me 40 minutes. I was really upset. But I still use it every day, can't live without it.", None),
    ]
    for speaker, turn_number, text, action in turns:
        session.add(
            state.Turn(
                call_id=call_id,
                turn_number=turn_number,
                speaker=speaker,
                text=text,
                action=action,
            )
        )
    session.commit()


async def _smoke_test() -> None:
    import os
    import sys
    from sqlalchemy.pool import StaticPool
    from sqlmodel import create_engine
    from tracing import init_tracing

    init_tracing(send_to_logfire=False)

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set — aborting.", file=sys.stderr)
        sys.exit(2)

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    state.init_db(engine)
    call_id = "smoke-test-call"

    with state.session_scope(engine) as session:
        _seed_canned_transcript(session, call_id)

    print("Seeded canned transcript. Running analyst...\n")

    with state.session_scope(engine) as session:
        deps = AnalystDeps(call_id=call_id, session=session)
        update = await run_analyst(deps)

    print("=== AnalysisUpdate ===")
    print(f"Themes ({len(update.themes)}):")
    for t in update.themes:
        print(f"  - {t}")
    print(f"\nContradictions ({len(update.contradictions)}):")
    for c in update.contradictions:
        print(f"  - {c}")
    print(f"\nSurprises ({len(update.surprises)}):")
    for s in update.surprises:
        print(f"  - {s}")
    print(f"\nNew probes ({len(update.new_probes)}):")
    for p in update.new_probes:
        print(f"  [{p.priority}] {p.question}")
        print(f"       rationale: {p.rationale}")

    # Confirm DB writes — read inside the session so ORM objects aren't detached
    print(f"\n=== DB verification ===")
    with state.session_scope(engine) as session:
        snapshots = list(session.exec(
            select(state.AnalystSnapshot).where(state.AnalystSnapshot.call_id == call_id)
        ))
        probes = list(session.exec(
            select(state.Probe).where(state.Probe.call_id == call_id)
        ))
        print(f"AnalystSnapshot rows: {len(snapshots)}")
        for snap in snapshots:
            print(f"  after_turn={snap.after_turn}, latency_ms={snap.latency_ms}")
        print(f"Probe rows: {len(probes)}")
        for probe in probes:
            print(f"  [{probe.priority}] {probe.question[:80]}")
        n_snapshots = len(snapshots)
        n_probes = len(probes)

    assert n_snapshots == 1, f"Expected 1 snapshot, got {n_snapshots}"
    assert n_probes == len(update.new_probes), (
        f"Expected {len(update.new_probes)} probe rows, got {n_probes}"
    )
    print("\nSmoke test passed.")


if __name__ == "__main__":
    import asyncio
    asyncio.run(_smoke_test())
