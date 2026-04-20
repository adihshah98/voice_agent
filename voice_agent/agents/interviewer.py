"""Interviewer agent — single-call, pre-fetched context.

Instead of a ReAct tool loop, we read all DB state before the LLM call and
inject it as a structured context block. One LLM call per turn → fits
comfortably inside the 1.8 s latency budget even without streaming.

Side effects (marking a probe as asked) happen in Python after the call,
based on the structured output, so the model never needs to call a tool.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from pydantic_ai import Agent

from voice_agent import state
from voice_agent.config import INTERVIEWER_BUDGET_S, INTERVIEWER_MODEL
from voice_agent.models import InterviewerDeps, InterviewerOutput


INTERVIEWER_PROMPT = """\
You are a skilled market-research interviewer on a live phone call.
You speak one short, conversational question at a time.

You receive a CONTEXT block, then the respondent's latest utterance.
You control the conversation — scripted questions are the study backbone,
analyst suggestions are inputs. You decide what happens next.

CONTEXT fields:
- SCRIPTED_REMAINING / NEXT_SCRIPTED: structured study questions
- TOP_PROBE: a suggestion from your analyst, with priority (1=urgent, 2=worthwhile,
  3=nice-to-have) and TURNS_AGO (how stale it is)
- RECENT_TURNS: last few turns

Decision framework — use your judgment in this order:

1. OFF-TOPIC: If the respondent went on a personal tangent unrelated to the study,
   acknowledge briefly and steer back with one open question. Use `off_topic`.

2. IMMEDIATE FOLLOW-UP: If the respondent just said something worth digging into
   - a crash
   - a complaint
   - a surprise
   - a specific event
   -  a contradiction
   - or even some detail but we want to go deeper 
   - limited info/details where more info would helo our research
   Then probe it NOW. Don't wait for the analyst. Ask the natural next question:
   "What happens when it crashes?", "When does that usually come up?",
   "What made you stop trusting it?" This beats scripted questions and analyst probes.
   Use action=`probe`.

3. ANALYST PROBE — TOP_PROBE is present:
   - TURNS_AGO ≤ 2 and still relevant to the current moment: use it. Rephrase naturally.
   - TURNS_AGO > 2: only revisit if it's still live in the conversation.
     If you use it, bridge explicitly: "Earlier you mentioned X — I wanted to come back
     to that..." Priority 1 is worth revisiting; priority 2–3, skip it if the
     moment has passed.
   - Skip entirely (and maybe come back to it later) if the current utterance gives you something more pressing.
   Use action=`probe`.

4. SCRIPTED: No immediate follow-up and no timely probe — ask NEXT_SCRIPTED.
   A small natural lead-in is fine; don't change the meaning. Use action=`scripted`.

5. CLARIFY: Only if the answer was genuinely ambiguous before you can move on.
   Do not use for clear, on-topic answers. Use action=`clarify`.

6. WRAP UP: SCRIPTED_REMAINING is 0 and no important threads remain open.
   Use action=`wrap_up`.

Hard rules:
- One question per utterance. Max one `?`.
- Never ask a leading question — never presuppose the answer or push a view.
  ("So you loved it, right?" / "That must have been frustrating?" are both leading.)
  Always use open, neutral phrasing: "What happened?", "How did that feel?",
  "What was that like for you?"
- Keep utterances under ~30 words — this is spoken, not written.
- Populate `reasoning` with one sentence on why you chose this action.
  Not spoken; for traces and evals only.
"""


def _build_context(session, call_id: str, current_turn: int) -> tuple[str, Optional[state.Probe]]:
    """Read all relevant DB state and return (context_block, top_probe).

    top_probe is returned separately so the caller can mark it asked after
    the LLM decides to use it — no second DB round-trip needed.
    Nothing is written to the DB here.
    """
    next_q = state.next_scripted(session, call_id)
    remaining = state.scripted_remaining(session, call_id)
    top_probe = state.pop_top_probe(session, call_id)
    turns = state.recent_turns(session, call_id, n=6)

    lines = ["[CONTEXT]", f"SCRIPTED_REMAINING: {remaining}"]

    lines.append(f"NEXT_SCRIPTED: {next_q}" if next_q else "NEXT_SCRIPTED: none")

    if top_probe:
        turns_ago = current_turn - (top_probe.generated_after_turn or 0)
        probe_line = (
            f"TOP_PROBE: [priority={top_probe.priority}, turns_ago={turns_ago}]"
            f" \"{top_probe.question}\""
        )
        if top_probe.rationale:
            probe_line += f"\n  rationale: {top_probe.rationale}"
        lines.append(probe_line)
    else:
        lines.append("TOP_PROBE: none")

    if turns:
        lines.append("RECENT_TURNS:")
        for t in turns:
            lines.append(f"  {t.speaker}: {t.text}")

    lines.append("[/CONTEXT]")
    return "\n".join(lines), top_probe


interviewer = Agent(
    INTERVIEWER_MODEL,
    deps_type=InterviewerDeps,
    output_type=InterviewerOutput,
    system_prompt=INTERVIEWER_PROMPT,
    instrument=True,
)


async def run_interviewer(
    deps: InterviewerDeps,
    respondent_text: str,
) -> InterviewerOutput:
    """Pre-fetch context → one LLM call → mark probe if used."""
    context_block, top_probe = _build_context(deps.session, deps.call_id, deps.turn_number)
    prompt = f"{context_block}\n\nRespondent: {respondent_text}"

    result = await interviewer.run(prompt, deps=deps)
    out = result.output

    if out.action == "scripted":
        state.mark_scripted_asked(deps.session, deps.call_id)
    elif out.action == "probe" and top_probe is not None:
        state.mark_probe_asked(deps.session, top_probe.id)

    return out


async def run_interviewer_with_timeout(
    deps: InterviewerDeps,
    respondent_text: str,
    *,
    budget_s: float = INTERVIEWER_BUDGET_S,
) -> InterviewerOutput:
    """Hard deadline wrapper. Returns a scripted fallback on timeout."""
    try:
        return await asyncio.wait_for(
            run_interviewer(deps, respondent_text),
            timeout=budget_s,
        )
    except asyncio.TimeoutError:
        return _fallback(deps)


def _fallback(deps: InterviewerDeps) -> InterviewerOutput:
    q = state.next_scripted(deps.session, deps.call_id)
    if q is not None:
        state.mark_scripted_asked(deps.session, deps.call_id)
        return InterviewerOutput(
            utterance=q,
            action="scripted",
            reasoning="fallback: agent exceeded budget; returning next scripted question",
        )
    return InterviewerOutput(
        utterance="Thanks so much for your time — I think that's everything I needed.",
        action="wrap_up",
        reasoning="fallback: agent exceeded budget and no scripted questions remain",
    )


# --- Minimal REPL for manual smoke-testing with a seeded DB ----------------


def _seed_demo_db():
    from sqlalchemy.pool import StaticPool
    from sqlmodel import create_engine

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    state.init_db(engine)
    call_id = "demo-call"
    with state.session_scope(engine) as s:
        s.add(
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
        s.add_all([
            state.Turn(
                call_id=call_id,
                turn_number=1,
                speaker="interviewer",
                text="How do you currently use the product?",
                action="scripted",
            ),
            state.Turn(
                call_id=call_id,
                turn_number=2,
                speaker="respondent",
                text="Mostly for planning my morning commute. I stopped trusting it a few months ago though.",
            ),
        ])
        s.add(
            state.Probe(
                call_id=call_id,
                question="You said you stopped trusting it — what happened?",
                priority=1,
                rationale="Respondent hinted at a trust break; worth digging into.",
            )
        )
    return engine, call_id


def _repl() -> None:
    import os
    import sys
    import time
    from tracing import init_tracing, turn_span, log_interviewer_decision

    init_tracing(send_to_logfire=False)
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set (check .env) — aborting REPL.", file=sys.stderr)
        sys.exit(2)
    engine, call_id = _seed_demo_db()
    print(f"Seeded demo call '{call_id}'. Type respondent lines; Ctrl-D to exit.")
    print("(Try: 'Yeah, there was a bad routing day and I just lost faith in it.')\n")

    turn_number = 3
    while True:
        try:
            line = input(f"respondent[{turn_number}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        with state.session_scope(engine) as session:
            deps = InterviewerDeps(
                call_id=call_id, session=session, turn_number=turn_number
            )
            with turn_span(call_id, turn_number, respondent_text=line):
                t0 = time.perf_counter()
                out = asyncio.run(run_interviewer_with_timeout(deps, line))
                latency_ms = int((time.perf_counter() - t0) * 1000)
                log_interviewer_decision(
                    call_id=call_id,
                    turn_number=turn_number,
                    action=out.action,
                    utterance=out.utterance,
                    reasoning=out.reasoning,
                    latency_ms=latency_ms,
                )

            session.add_all([
                state.Turn(
                    call_id=call_id,
                    turn_number=turn_number,
                    speaker="respondent",
                    text=line,
                ),
                state.Turn(
                    call_id=call_id,
                    turn_number=turn_number + 1,
                    speaker="interviewer",
                    text=out.utterance,
                    action=out.action,
                    reasoning=out.reasoning,
                    latency_ms=latency_ms,
                ),
            ])

        print(f"  action={out.action}  ({latency_ms} ms)")
        print(f"  interviewer> {out.utterance}")
        print(f"  why: {out.reasoning}\n")
        turn_number += 2

        if out.action == "wrap_up":
            print("Call wrapped up. Exiting.")
            break


if __name__ == "__main__":
    _repl()
