"""Analyst agent — structured output, fire-and-forget.

Reads all turns for a call, produces themes/contradictions/surprises/probes,
and persists an AnalystSnapshot + new Probe rows.

Never raises into the live call path — use `run_analyst_safely` from server.py.
"""

from __future__ import annotations

import time

import logfire
from opentelemetry import trace
from dotenv import load_dotenv
from pydantic_ai import Agent
from sqlmodel import select

load_dotenv()

from voice_agent import state
from voice_agent.config import ANALYST_MODEL
from voice_agent.models import AnalysisUpdate, AnalystDeps


ANALYST_PROMPT = """\
You are an investment research analyst reviewing a live B2B SaaS/AI customer interview.
The interviewer is gathering intelligence on behalf of an investor doing due diligence.

You may receive an ESTABLISHED CONTEXT block summarising prior analysis, followed by a
NEW TRANSCRIPT block containing only the turns since that analysis. If present, treat
ESTABLISHED CONTEXT as already confirmed — do not re-flag the same themes, contradictions,
or signals. Build on it: surface what's new, changed, or now contradicted.
If there is no ESTABLISHED CONTEXT, you receive the full TRANSCRIPT from the start.

Your job:
1. THEMES — recurring ideas or patterns. Focus on what's investment-relevant:
   adoption depth, use-case clarity, workflow integration, team dependence.

2. CONTRADICTIONS — places where the respondent contradicts themselves.
   Pay special attention to gaps between stated satisfaction and actual usage behavior.

3. SURPRISES — unexpected answers that reveal novel signal: an unexpected use case,
   a competitor you didn't know was in the market, a pricing structure that's unusual,
   an adoption barrier that's non-obvious.

4. INVESTOR_SIGNALS — the most important output. Tag each signal with its category:
   - [PMF] product-market fit signals: daily habit, word-of-mouth, "can't live without it",
     organic expansion within the org, strong unprompted advocacy, high NPS-equivalent
   - [COMPETITIVE] competitive intelligence: named alternatives, why they were rejected,
     switching triggers, what would make the respondent reconsider
   - [REVENUE] revenue/business signals: who owns the budget, annual vs. monthly contract,
     seat expansion signals, price sensitivity, renewal risk or intent
   - [AI-SIGNAL] AI-specific signals: trust in AI outputs, verification habits, ROI clarity,
     adoption barriers (security, compliance, hallucinations), workflow integration depth
   - [RED-FLAG] churn risk or weak PMF: low seat utilization, "mostly for demos",
     implementation stalled, IT/security pushback, evaluating alternatives, low rating

   Format each signal as a tagged one-liner citing the transcript:
   "[PMF] Uses it before every meeting — daily habit, cited 'can't imagine going back'"
   "[RED-FLAG] Only 3 of 20 seats active after 6 months — rollout stalled"

5. NEW_PROBES — follow-up questions worth asking. Probe guidelines:
   - Reference specifics from the transcript.
   - Use open, neutral phrasing — never leading questions.
   - Priority 1 = contradiction about value/usage OR red flag (churn risk, stalled adoption)
                  OR strong PMF signal that hasn't been probed yet (word-of-mouth source,
                  expansion plans, who else uses it)
   - Priority 2 = competitive signal needing clarification, or revenue/budget detail missing
   - Priority 3 = AI trust/adoption depth, nice-to-have context
   - Maximum 3 new probes per pass.
   - Never repeat a question already in EXISTING_PROBES.

Be concise — bullet-point style for all lists except investor_signals (one tagged line each).
"""


def _build_prompt(session, call_id: str) -> tuple[str, int]:
    """Return (prompt_text, last_turn_number).

    If a prior AnalystSnapshot exists, feeds it as established context and only
    appends turns since that snapshot — keeps the prompt within context limits
    regardless of interview length.
    """
    snapshot = state.latest_snapshot(session, call_id)

    if snapshot:
        new_turns = state.turns_since(session, call_id, after_turn=snapshot.after_turn)
        last_turn = new_turns[-1].turn_number if new_turns else snapshot.after_turn

        lines = [
            "ESTABLISHED CONTEXT (from prior analysis — treat as already confirmed):",
            f"  Themes: {'; '.join(snapshot.themes) if snapshot.themes else 'none'}",
            f"  Contradictions: {'; '.join(snapshot.contradictions) if snapshot.contradictions else 'none'}",
            f"  Surprises: {'; '.join(snapshot.surprises) if snapshot.surprises else 'none'}",
            "  Investor signals:",
        ]
        for sig in snapshot.investor_signals:
            lines.append(f"    {sig}")
        lines.append(f"  (covers turns 1–{snapshot.after_turn})")
        lines.append("")
        lines.append(f"NEW TRANSCRIPT (turns {snapshot.after_turn + 1}–{last_turn}):")
        for t in new_turns:
            lines.append(f"  {t.speaker.upper()} [{t.turn_number}]: {t.text}")
    else:
        new_turns = state.recent_turns(session, call_id, n=200)
        if not new_turns:
            return "(no turns yet)", 0
        last_turn = new_turns[-1].turn_number
        lines = ["TRANSCRIPT:"]
        for t in new_turns:
            lines.append(f"{t.speaker.upper()} [{t.turn_number}]: {t.text}")

    existing_stmt = select(state.Probe).where(state.Probe.call_id == call_id)
    existing = [p.question for p in session.exec(existing_stmt)]

    prompt = "\n".join(lines)
    if existing:
        prompt += "\n\nEXISTING_PROBES (do not repeat):\n" + "\n".join(
            f"- {q}" for q in existing
        )

    return prompt, last_turn


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
            investor_signals=update.investor_signals,
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

    span = trace.get_current_span()
    span.set_attribute("after_turn", after_turn)
    span.set_attribute("latency_ms", latency_ms)
    span.set_attribute("themes_count", len(update.themes))
    span.set_attribute("contradictions_count", len(update.contradictions))
    span.set_attribute("surprises_count", len(update.surprises))
    span.set_attribute("investor_signals_count", len(update.investor_signals))
    span.set_attribute("new_probes_count", len(update.new_probes))

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
                "Walk me through how your team actually uses the product day-to-day.",
                "When you were evaluating options, what else did you look at?",
                "What ultimately made you go with this product over those alternatives?",
                "If you had to rate the product from one to ten, what would you say and why?",
            ],
            status="active",
        )
    )
    turns = [
        ("interviewer", 1, "Walk me through how your team actually uses the product day-to-day.", "scripted"),
        ("respondent", 2, "Our AEs use it for call summaries after every customer meeting. A colleague recommended it — she'd seen it at her previous company.", None),
        ("interviewer", 3, "How did your colleague first come across it at that previous company?", "probe"),
        ("respondent", 4, "It had spread organically there. Apparently one sales rep started using it and within a month the whole team was on it.", None),
        ("interviewer", 5, "When you were evaluating options, what else did you look at?", "scripted"),
        ("respondent", 6, "We looked at Gong and Chorus. They're fine but honestly they felt like overkill — this was much easier to deploy. Though I'll say IT had some questions about where the data goes.", None),
        ("interviewer", 7, "What specifically concerned IT about the data?", "probe"),
        ("respondent", 8, "SOC 2 compliance, mostly. Once we confirmed that it was fine. We went annual — our VP of Sales owns the contract, not IT.", None),
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
    print(f"\nInvestor Signals ({len(update.investor_signals)}):")
    for sig in update.investor_signals:
        print(f"  {sig}")
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
