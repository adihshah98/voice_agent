"""Interviewer agent — single-call, pre-fetched context.

Instead of a ReAct tool loop, we read all DB state before the LLM call and
inject it as a structured context block. One LLM call per turn; wall-clock is
bounded by `INTERVIEWER_BUDGET_S` in `config.py` (hard timeout wrapper).

Side effects (marking a probe as asked) happen in Python after the call,
based on the structured output, so the model never needs to call a tool.
"""

from __future__ import annotations

import asyncio
import functools
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import anyio
import logfire
from pydantic_ai import Agent
from pydantic_ai.messages import CachePoint
from pydantic_ai.models.anthropic import AnthropicModelSettings

from voice_agent import state
from voice_agent.config import INTERVIEWER_BUDGET_S, INTERVIEWER_MODEL, settings
from voice_agent.models import InterviewerDeps, InterviewerOutput


CONTEXT_WINDOW_TURNS = 50  # recent turns injected into the LLM context block
PROBE_STALENESS_TURNS = 15    # probes older than this many turns are dropped from context


@dataclass
class PreparedInterviewerTurn:
    prompt_parts: list[str | CachePoint]
    fallback_scripted_question: str | None


@dataclass
class InterviewerContextReads:
    next_scripted_question: str | None
    scripted_remaining: int
    probes: list[state.Probe]
    snapshot: state.AnalystSnapshot | None


def _build_prompt_parts_from_reads(
    reads: InterviewerContextReads,
    current_turn: int,
    respondent_text: str,
    vapi_messages: list[dict],
) -> list[str | CachePoint]:
    covered_lines = []
    if reads.snapshot and reads.snapshot.covered_subtopics:
        covered_lines.append("COVERED_SUBTOPICS (do NOT revisit these areas):")
        for topic in reads.snapshot.covered_subtopics:
            covered_lines.append(f"  - {topic}")
    else:
        covered_lines.append("COVERED_SUBTOPICS: none")

    recent = [m for m in vapi_messages if m.get("role") in ("assistant", "user")][-CONTEXT_WINDOW_TURNS:]
    dynamic_lines = ["[CONTEXT]", f"SCRIPTED_REMAINING: {reads.scripted_remaining}"]
    dynamic_lines.append(
        f"NEXT_SCRIPTED: {reads.next_scripted_question}"
        if reads.next_scripted_question
        else "NEXT_SCRIPTED: none"
    )

    if reads.probes:
        dynamic_lines.append("PENDING_PROBES (analyst suggestions, in priority order):")
        for probe in reads.probes:
            turns_ago = current_turn - (probe.generated_after_turn or 0)
            line = f'  [id={probe.id}, priority={probe.priority}, turns_ago={turns_ago}] "{probe.question}"'
            if probe.rationale:
                line += f"\n    rationale: {probe.rationale}"
            dynamic_lines.append(line)
    else:
        dynamic_lines.append("PENDING_PROBES: none")

    if recent:
        dynamic_lines.append("RECENT_TURNS:")
        for m in recent:
            speaker = "interviewer" if m["role"] == "assistant" else "respondent"
            dynamic_lines.append(f"  {speaker}: {m.get('content', '')}")
    dynamic_lines.append("[/CONTEXT]")
    dynamic_lines.append("")
    dynamic_lines.append(f"Respondent: {respondent_text}")

    return [
        "\n".join(covered_lines),
        CachePoint(ttl="1h"),
        "\n".join(dynamic_lines),
    ]


def prepare_interviewer_turn(
    session,
    call_id: str,
    current_turn: int,
    respondent_text: str,
    vapi_messages: list[dict] | None,
) -> PreparedInterviewerTurn:
    """Load DB-backed context in one short-lived read session."""
    messages = vapi_messages or _db_messages_fallback(session, call_id)
    reads = InterviewerContextReads(
        next_scripted_question=state.next_scripted(session, call_id),
        scripted_remaining=state.scripted_remaining(session, call_id),
        probes=state.top_probes(session, call_id, n=3, min_turn=current_turn - PROBE_STALENESS_TURNS),
        snapshot=state.latest_snapshot(session, call_id),
    )
    prompt_parts = _build_prompt_parts_from_reads(
        reads,
        current_turn,
        respondent_text=respondent_text,
        vapi_messages=messages,
    )
    return PreparedInterviewerTurn(
        prompt_parts=prompt_parts,
        fallback_scripted_question=reads.next_scripted_question,
    )


async def prepare_interviewer_turn_concurrent(
    engine,
    call_id: str,
    current_turn: int,
    respondent_text: str,
    vapi_messages: list[dict] | None,
) -> PreparedInterviewerTurn:
    """Read context with parallel short sessions (best for pooled Postgres)."""

    async def _read(fn, *args):
        with state.session_scope(engine) as session:
            return await anyio.to_thread.run_sync(functools.partial(fn, session, *args))

    next_q, remaining, probes, snapshot = await asyncio.gather(
        _read(state.next_scripted, call_id),
        _read(state.scripted_remaining, call_id),
        _read(state.top_probes, call_id, 3, current_turn - PROBE_STALENESS_TURNS),
        _read(state.latest_snapshot, call_id),
    )
    messages = vapi_messages or await _read(_db_messages_fallback, call_id)

    reads = InterviewerContextReads(
        next_scripted_question=next_q,
        scripted_remaining=remaining or 0,
        probes=probes or [],
        snapshot=snapshot,
    )
    prompt_parts = _build_prompt_parts_from_reads(
        reads,
        current_turn,
        respondent_text=respondent_text,
        vapi_messages=messages,
    )
    return PreparedInterviewerTurn(
        prompt_parts=prompt_parts,
        fallback_scripted_question=reads.next_scripted_question,
    )


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
- COVERED_SUBTOPICS: specific subtopic labels already covered (explicit or organic) — labels name exact entities and dimensions (e.g. 'Notion vs Google Docs product features'). A covered label does NOT block other entities or dimensions ('Notion vs Quip' or 'Notion vs Google Docs pricing' remain open)
- RECENT_TURNS: last several turns of conversation

Decision framework — use your judgment in this order:

0. NO REPETITION — before choosing any action, check RECENT_TURNS and COVERED_SUBTOPICS.
   If the specific subtopic you're about to ask about was already addressed, skip it and
   move to the next step, unless the answer was incomplete or evasive. A broad topic
   being covered does not block adjacent subtopics (e.g. "competitor product features"
   covered does not block "competitor pricing structure").

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
   Do NOT ask about a subtopic already in COVERED_SUBTOPICS or RECENT_TURNS.

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


interviewer = Agent(
    INTERVIEWER_MODEL,
    deps_type=InterviewerDeps,
    output_type=InterviewerOutput,
    system_prompt=INTERVIEWER_PROMPT,
    instrument=True,
    model_settings=AnthropicModelSettings(
        anthropic_cache_instructions='1h',
    ),
)


async def run_interviewer(
    deps: InterviewerDeps,
    respondent_text: str,
    vapi_messages: list[dict] | None = None,
    prepared: PreparedInterviewerTurn | None = None,
) -> InterviewerOutput:
    """Pre-fetch context → one LLM call → return output.

    vapi_messages: OpenAI-formatted message array from Vapi (body["messages"]).
    When None (evals / play.py), falls back to reading recent_turns from the DB.
    """
    if prepared is None:
        assert deps.session is not None
        prepared = prepare_interviewer_turn(
            deps.session,
            deps.call_id,
            deps.turn_number,
            respondent_text=respondent_text,
            vapi_messages=vapi_messages or _db_messages_fallback(deps.session, deps.call_id),
        )
    result = await interviewer.run(prepared.prompt_parts, deps=deps)
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


class InterviewerStream:
    """Streaming interviewer turn. Consume tokens() fully, then read output.

    Uses anyio.move_on_after instead of asyncio.wait_for — PydanticAI uses
    anyio task groups internally, and asyncio cancellation injects a cancel
    that races with anyio's stream teardown, producing ClosedResourceError.
    """

    def __init__(
        self,
        deps: InterviewerDeps,
        prepared: PreparedInterviewerTurn,
        *,
        budget_s: float = INTERVIEWER_BUDGET_S,
    ) -> None:
        self._deps = deps
        self._prepared = prepared
        self._budget_s = budget_s
        self._output: InterviewerOutput | None = None

    @property
    def output(self) -> InterviewerOutput:
        assert self._output is not None, "tokens() must be fully consumed before reading output"
        return self._output

    async def tokens(self) -> AsyncGenerator[str, None]:
        """Yield text deltas as the utterance streams in."""
        deps = self._deps
        prepared = self._prepared
        final_output: InterviewerOutput | None = None

        with anyio.move_on_after(self._budget_s) as cancel_scope:
            prev = ""
            async with interviewer.run_stream(prepared.prompt_parts, deps=deps) as streamed:
                async for partial in streamed.stream_output(debounce_by=None):
                    current = partial.utterance or ""
                    if len(current) > len(prev):
                        yield current[len(prev):]
                        prev = current
                final_output = await streamed.get_output()

        if cancel_scope.cancelled_caught:
            logfire.warning(
                "interviewer_timeout",
                call_id=deps.call_id,
                turn_number=deps.turn_number,
                budget_s=self._budget_s,
            )
            fb = _fallback(deps, fallback_scripted_question=prepared.fallback_scripted_question)
            self._output = fb
            yield fb.utterance
            return

        self._output = final_output


def _fallback(
    deps: InterviewerDeps,
    *,
    fallback_scripted_question: str | None = None,
) -> InterviewerOutput:
    q = fallback_scripted_question
    if q is None and deps.session is not None:
        q = state.next_scripted(deps.session, deps.call_id)
    if q is not None:
        return InterviewerOutput(
            utterance=q,
            action="scripted",
            reasoning="fallback: agent exceeded budget; returning next scripted question",
            is_fallback=True,
        )
    return InterviewerOutput(
        utterance="Thanks so much for your time — I think that's everything I needed.",
        action="wrap_up",
        reasoning="fallback: agent exceeded budget and no scripted questions remain",
        is_fallback=True,
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
    if not settings.anthropic_api_key:
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
                out = asyncio.run(run_interviewer(deps, line))
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
