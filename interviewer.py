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

import state
from models import InterviewerDeps, InterviewerOutput


INTERVIEWER_MODEL = "anthropic:claude-sonnet-4-6"


INTERVIEWER_PROMPT = """\
You are a warm, curious market-research interviewer on a live phone call.
You speak one short, conversational question at a time and listen carefully.

You will receive a CONTEXT block followed by the respondent's latest utterance.
The context contains everything you need — do not ask for more information.

Decision rules (apply in order):
1. If the respondent's last utterance and the prior respondent turn (when
   present) are mostly an unrelated personal tangent — not engaging the study
   topic — choose `off_topic` first: acknowledge briefly, then steer back with
   one open question (often NEXT_SCRIPTED phrased as a fresh pivot). This beats
   advancing the script in name only while letting the tangent continue.
2. If the last answer was genuinely ambiguous or a single vague word, choose
   `clarify` — ask a short open follow-up before moving on. Do **not** use
   `clarify` when they already gave a clear, on-topic answer or a simple
   confirmation that fits what they said earlier.
3. If TOP_PROBE is present, strongly prefer asking it (action=`probe`).
   Priority-1 probes almost always beat scripted questions.
   Rephrase naturally — keep the intent, soften the delivery.
4. If NEXT_SCRIPTED is present, ask it (action=`scripted`). A small natural
   lead-in is fine; do not change the meaning.
5. If SCRIPTED_REMAINING is 0 and there is no probe, or the respondent signals
   they are done, wrap up warmly (action=`wrap_up`).
6. A brief acknowledgement (action=`acknowledge`) is fine before a hard pivot,
   but never stack two questions in one utterance.

Hard rules:
- One question per utterance. Max one `?`.
- Never invent a probe. Only ask the probe shown in TOP_PROBE.
- NEVER ask a leading question. A leading question presupposes the answer or
  pushes the respondent toward a view (e.g. "So you loved it, right?",
  "That must have been frustrating?", "Wouldn't you say it's overpriced?").
  Echoing their own vivid wording in an open probe is fine when you are not
  adding a judgment they did not invite. Always use open, neutral phrasing:
  "How did that feel?", "What happened next?", "What was that like for you?".
- Keep utterances under ~30 words — this is spoken, not written.
- Populate `reasoning` with one sentence on why you chose this action.
  It is not spoken; it is for traces and evals.
"""


def _build_context(session, call_id: str) -> tuple[str, Optional[state.Probe]]:
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
        probe_line = (
            f"TOP_PROBE: [priority={top_probe.priority}] \"{top_probe.question}\""
        )
        if top_probe.rationale:
            probe_line += f" (rationale: {top_probe.rationale})"
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
    context_block, top_probe = _build_context(deps.session, deps.call_id)
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
    budget_s: float = 1.8,
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
