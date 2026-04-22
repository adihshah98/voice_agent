"""Interviewer agent — single-call, pre-fetched context.

Instead of a ReAct tool loop, we read all DB state before the LLM call and
inject it as a structured context block. One LLM call per turn; wall-clock is
bounded by `INTERVIEWER_BUDGET_S` in `config.py` (hard timeout wrapper).

Side effects (marking a probe as asked) happen in Python after the call,
based on the structured output, so the model never needs to call a tool.
"""

from __future__ import annotations

import asyncio

import anyio
import logfire
from pydantic_ai import Agent

from voice_agent import state
from voice_agent.config import INTERVIEWER_BUDGET_S, INTERVIEWER_MODEL
from voice_agent.models import InterviewerDeps, InterviewerOutput


INTERVIEWER_PROMPT = """\
You are conducting a customer interview on behalf of an investor or research firm
doing due diligence on a B2B SaaS or AI product. Your job is to understand how
real customers use the product, what value they get, and what signals exist around
product-market fit, competitive position, and revenue dynamics.

You speak one short, conversational question at a time.

You receive a CONTEXT block, then the respondent's latest utterance.
You control the conversation — scripted questions are the study backbone,
analyst suggestions are inputs. You decide what happens next.

CONTEXT fields:
- SCRIPTED_REMAINING / NEXT_SCRIPTED: structured study questions
- PENDING_PROBES: up to 3 analyst suggestions, each tagged with [id, priority, turns_ago].
  Priority 1=urgent, 2=worthwhile, 3=nice-to-have. All are fresh (turns_ago ≤ 8).
  When you use one, set action=probe and probe_id_used to its exact id.
- COVERED_TOPICS: every question you have already asked — never revisit these
- RECENT_TURNS: last several turns of conversation

Decision framework — use your judgment in this order:

0. NO REPETITION — before choosing any action, check RECENT_TURNS and COVERED_TOPICS.
   If the topic you're about to ask about was already addressed, skip it entirely and
   move to the next step, unless the answer was incomplete or evasive.

1. OFF-TOPIC: If the respondent went on a personal tangent unrelated to the study,
   acknowledge briefly and steer back with one open question. Use `off_topic`.

2. IMMEDIATE FOLLOW-UP: If the respondent just said something worth digging into —
   a complaint, surprise, contradiction, specific detail, red flag, or investor signal
   (see triggers below) — probe it NOW. Don't wait for the analyst.
   Use action=`probe`.

3. ANALYST PROBE — PENDING_PROBES is non-empty:
   - Pick the highest-priority probe not already in COVERED_TOPICS.
   - TURNS_AGO ≤ 2: use it directly, rephrase naturally.
   - TURNS_AGO 3–8: bridge with "Earlier you mentioned X..." if needed.
   - Set probe_id_used to the probe's exact id.
   - Skip if the current utterance gives you something more pressing.
   Use action=`probe`.

4. SCRIPTED: No immediate follow-up and no timely probe — ask NEXT_SCRIPTED.
   A small natural lead-in is fine; don't change the meaning. Use action=`scripted`.
   EXCEPTION — if the respondent has already answered NEXT_SCRIPTED earlier in the
   conversation (check COVERED_TOPICS and RECENT_TURNS), skip it silently: set
   action=`skip_scripted` and move on to a probe or the following scripted question.
   Do NOT ask a question you already have the answer to.

5. CLARIFY: Only if the answer was genuinely ambiguous before you can move on.
   Do not use for clear, on-topic answers. Use action=`clarify`.

6. WRAP UP: SCRIPTED_REMAINING is 0 and no important threads remain open.
   Use action=`wrap_up`.

--- INVESTOR SIGNAL TRIGGERS ---
These are high-value moments. When you hear them, deviate from scripted order
and probe immediately (action=`probe`):

REFERRAL / WORD-OF-MOUTH — "a colleague recommended it", "everyone I know uses it",
"I just found it on my own": probe one level deeper.
→ "How did your colleague come across it?" / "Has anyone else on your team started using it on their own?"

AI TRUST / VERIFICATION — "I always double-check it", "I don't fully trust the AI",
"it sometimes hallucinates", "I verify everything": probe the gap.
→ "What's your process for checking the outputs?" / "What would it take for you to trust it without checking?"

ROI / QUANTIFICATION — "it saves a lot of time", "we're seeing real value", any
mention of hours saved, deals closed, cost reduced: get specific.
→ "Can you give me a rough sense of the scale — hours per week, something like that?"

COMPETITOR MENTION — any named alternative tool or vendor: probe differentiation and stickiness.
→ "What made [X] not the right fit?" / "Is [X] still something your team looks at?"

BUDGET / APPROVAL PATH — "we had to get approval", "it's in the IT budget",
"our VP signed off", contract details: probe ownership and structure.
→ "Who owns that budget at your company — is it a central IT decision or team-by-team?"

EXPANSION SIGNAL — "other teams are asking about it", "we're thinking of rolling it out
more broadly", "we almost didn't renew but...": probe what's driving or blocking it.
→ "What would a broader rollout look like?" / "What's the main thing holding that back?"

RED FLAGS — always probe these; don't move on without understanding them:
- "We bought it but haven't fully rolled it out" → "What got in the way of the rollout?"
- "It's mostly used for demos / one-off projects" → "What's kept it from production use?"
- "IT or security pushed back on it" → "What specifically concerned them?"
- Low rating (1–5) → "What specific experience is behind that number?"
- "We're evaluating other options" → "What's prompting that?"
--- END TRIGGERS ---

Hard rules:
- One question per utterance. Max one `?`.
- Never ask a leading question — never presuppose the answer or push a view.
  ("So you loved it, right?" / "That must have been frustrating?" are both leading.)
  Always use open, neutral phrasing: "What happened?", "How did that feel?",
  "What was that like for you?"
- Keep utterances under ~30 words — this is spoken, not written.
- Valid actions: scripted, probe, clarify, off_topic, wrap_up. Do not use `acknowledge`
  as a standalone action — brief acknowledgments belong in the utterance itself before
  steering back.
- Populate `reasoning` with one sentence on why you chose this action.
  Not spoken; for traces and evals only.
"""


def _build_context(
    session,
    call_id: str,
    current_turn: int,
    vapi_messages: list[dict],
) -> tuple[str, list[state.Probe]]:
    """Read study state from DB and conversation history from Vapi's messages.

    vapi_messages is the OpenAI-formatted array Vapi sends with each LLM request
    (body["messages"]). It reflects what was actually spoken — no provisional turns.
    active_probes are returned so the caller can mark the used one after the LLM call.
    Nothing is written to the DB here.
    """
    next_q = state.next_scripted(session, call_id)
    remaining = state.scripted_remaining(session, call_id)
    probes = state.top_probes(session, call_id, n=3)

    lines = ["[CONTEXT]", f"SCRIPTED_REMAINING: {remaining}"]
    lines.append(f"NEXT_SCRIPTED: {next_q}" if next_q else "NEXT_SCRIPTED: none")

    active_probes = [
        p for p in probes
        if (current_turn - (p.generated_after_turn or 0)) <= 8
    ]
    if active_probes:
        lines.append("PENDING_PROBES (analyst suggestions, in priority order):")
        for probe in active_probes:
            turns_ago = current_turn - (probe.generated_after_turn or 0)
            line = f'  [id={probe.id}, priority={probe.priority}, turns_ago={turns_ago}] "{probe.question}"'
            if probe.rationale:
                line += f"\n    rationale: {probe.rationale}"
            lines.append(line)
    else:
        lines.append("PENDING_PROBES: none")

    # Build COVERED_TOPICS and RECENT_TURNS from Vapi's confirmed message history.
    # Skip the system prompt (index 0); alternate assistant/user from there.
    convo = [m for m in vapi_messages if m.get("role") in ("assistant", "user")]

    covered = [
        m["content"] for m in convo
        if m.get("role") == "assistant" and m.get("content", "").strip().endswith("?")
    ]
    if covered:
        lines.append("COVERED_TOPICS (do NOT revisit these):")
        for q in covered:
            lines.append(f"  - {q}")

    recent = convo[-14:]
    if recent:
        lines.append("RECENT_TURNS:")
        for m in recent:
            speaker = "interviewer" if m["role"] == "assistant" else "respondent"
            lines.append(f"  {speaker}: {m.get('content', '')}")

    lines.append("[/CONTEXT]")
    return "\n".join(lines), active_probes


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
    vapi_messages: list[dict] | None = None,
) -> InterviewerOutput:
    """Pre-fetch context → one LLM call → return output.

    vapi_messages: OpenAI-formatted message array from Vapi (body["messages"]).
    When None (evals / play.py), falls back to reading recent_turns from the DB.
    """
    context_block, active_probes = _build_context(
        deps.session, deps.call_id, deps.turn_number,
        vapi_messages=vapi_messages or _db_messages_fallback(deps.session, deps.call_id),
    )
    prompt = f"{context_block}\n\nRespondent: {respondent_text}"

    result = await interviewer.run(prompt, deps=deps)
    return result.output


def _db_messages_fallback(session, call_id: str) -> list[dict]:
    """Build an OpenAI-format messages list from DB turns.

    Used by play.py and evals that don't go through the Vapi LLM endpoint.
    """
    turns = state.recent_turns(session, call_id, n=60)
    messages = []
    for t in turns:
        role = "assistant" if t.speaker == "interviewer" else "user"
        messages.append({"role": role, "content": t.text})
    return messages


async def run_interviewer_with_timeout(
    deps: InterviewerDeps,
    respondent_text: str,
    *,
    budget_s: float = INTERVIEWER_BUDGET_S,
) -> InterviewerOutput:
    """Hard deadline wrapper. Returns a scripted fallback on timeout.

    Uses anyio.move_on_after instead of asyncio.wait_for — PydanticAI uses
    anyio task groups internally, and asyncio cancellation injects a cancel
    that races with anyio's stream teardown, producing ClosedResourceError.
    """
    result: InterviewerOutput | None = None

    with anyio.move_on_after(budget_s) as cancel_scope:
        result = await run_interviewer(deps, respondent_text)

    if cancel_scope.cancelled_caught:
        logfire.warning("interviewer_timeout", call_id=deps.call_id, turn_number=deps.turn_number, budget_s=budget_s)
        return _fallback(deps)

    assert result is not None
    return result


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
                    "Walk me through how your team actually uses the product day-to-day.",
                    "When you were evaluating options, what else did you look at?",
                    "What ultimately made you go with this product over those alternatives?",
                    "How did the buying and rollout process go — anything that stood out?",
                    "If you had to rate the product from one to ten based on your experience so far, what would you say — and what's behind that number?",
                    "What's the one thing you'd most want the product to change or add?",
                ],
                status="active",
            )
        )
        s.add_all([
            state.Turn(
                call_id=call_id,
                turn_number=1,
                speaker="interviewer",
                text="Walk me through how your team actually uses the product day-to-day.",
                action="scripted",
            ),
            state.Turn(
                call_id=call_id,
                turn_number=2,
                speaker="respondent",
                text="We use it mainly for sales call summaries. Our AEs love it — a colleague actually recommended it to our VP after seeing it at another company.",
            ),
        ])
        s.add(
            state.Probe(
                call_id=call_id,
                question="How did your colleague first come across it at that other company?",
                priority=1,
                rationale="Word-of-mouth referral chain — strong PMF signal worth probing.",
            )
        )
    return engine, call_id


def _repl() -> None:
    import os
    import sys
    import time
    from tracing import agent_span, init_tracing

    init_tracing(send_to_logfire=False)
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set (check .env) — aborting REPL.", file=sys.stderr)
        sys.exit(2)
    engine, call_id = _seed_demo_db()
    print(f"Seeded demo call '{call_id}'. Type respondent lines; Ctrl-D to exit.")
    print("(Try: 'We evaluated Gong and Chorus too, but honestly security flagged both of them.')\n")

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
            with agent_span("interviewer", call_id, turn_number=turn_number, respondent_text=line) as span:
                t0 = time.perf_counter()
                out = asyncio.run(run_interviewer_with_timeout(deps, line))
                latency_ms = int((time.perf_counter() - t0) * 1000)
                span.set_attribute("action", out.action)
                span.set_attribute("utterance", out.utterance)
                span.set_attribute("reasoning", out.reasoning)
                span.set_attribute("latency_ms", latency_ms)

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
