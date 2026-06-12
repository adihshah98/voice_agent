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
import json
import re
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import anyio
import logfire
from groq import APIError as GroqAPIError
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelAPIError, UnexpectedModelBehavior
from pydantic_ai.messages import CachePoint
from pydantic_ai.usage import RunUsage
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.cerebras import CerebrasModel
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.cerebras import CerebrasProvider
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.groq import GroqProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic import ValidationError

from voice_agent import state
from voice_agent.config import (
    INTERVIEWER_BUDGET_S,
    INTERVIEWER_CEREBRAS_MODEL,
    INTERVIEWER_GEMINI_MODEL,
    INTERVIEWER_GROQ_MODEL,
    INTERVIEWER_HAIKU_MODEL,
    INTERVIEWER_OPENAI_MODEL,
    INTERVIEWER_RECOVERY_UTTERANCE,
    settings,
)
from voice_agent.interviewer_llm_caching import (
    anthropic_interviewer_settings,
    openai_interviewer_settings,
    user_message_cache_breakpoint,
)
from voice_agent.models import InterviewerDeps, InterviewerLLMMeta, InterviewerOutput


CONTEXT_WINDOW_TURNS = 15  # recent turns injected into the LLM context block; keep under Cerebras 8k limit
PROBE_STALENESS_TURNS = 15    # probes older than this many turns are dropped from context


@dataclass
class PreparedInterviewerTurn:
    prompt_parts: list[str | CachePoint]
    fallback_scripted_question: str | None
    respondent_text: str = ""
    has_pending_probes: bool = False
    # Number of consecutive trailing clarify turns; used to enforce wrap_up in Python
    # when the model misses the counting rule (deterministic override).
    consecutive_clarify_count: int = 0
    # Whether the current respondent utterance is silence/blank (for the override check).
    is_silence: bool = False


@dataclass
class InterviewerContextReads:
    next_scripted_question: str | None
    scripted_remaining: int
    probes: list[state.Probe]
    snapshot: state.AnalystSnapshot | None
    consecutive_clarify_count: int = 0


def _recommend_next(reads: InterviewerContextReads) -> str:
    """Resolve the probe-vs-scripted priority comparison in Python.

    The model used to do this arithmetic in-prompt ("priority 1 beats scripted,
    3 yields…"), which small fallback models get wrong. We pre-resolve it and
    inject the result as RECOMMENDED_NEXT. It is a recommendation, not a mandate:
    clarify (step 4) and open threads (step 6) still come first, and the model
    skips a probe that duplicates COVERED_SUBTOPICS. `top_probes` is already
    ordered by priority asc then age, so probes[0] is the strongest candidate.
    """
    has_scripted = reads.next_scripted_question is not None
    if reads.probes:
        top = reads.probes[0]
        if top.priority <= 2:
            return f"probe id={top.id} — priority {top.priority} outranks the next scripted question"
        if has_scripted:
            return "scripted — pending probes are priority 3 (nice-to-have) and yield to scripted"
        return f"probe id={top.id} — priority 3, but no scripted questions remain"
    if has_scripted:
        return "no analyst probe pending — layer first if the answer left a thread open, otherwise ask the scripted question"
    return "wrap_up — no scripted questions remain and no analyst probe pending"


def _build_prompt_parts_from_reads(
    reads: InterviewerContextReads,
    current_turn: int,
    respondent_text: str,
    vapi_messages: list[dict],
) -> list[str | CachePoint]:
    # Order: COVERED_SUBTOPICS + CALL_CONTEXT, cache breakpoint (Anthropic), then per-turn CONTEXT.
    # COVERED_SUBTOPICS and CALL_CONTEXT sit before the cache breakpoint so they are cached across
    # turns (they change only when the analyst runs, not every turn). See interviewer_llm_caching.
    covered_lines = []
    if reads.snapshot and reads.snapshot.covered_subtopics:
        covered_lines.append("COVERED_SUBTOPICS (do NOT revisit these areas):")
        for topic in reads.snapshot.covered_subtopics:
            covered_lines.append(f"  - {topic}")
    else:
        covered_lines.append("COVERED_SUBTOPICS: none")

    if reads.snapshot and (reads.snapshot.themes or reads.snapshot.contradictions or reads.snapshot.investor_signals):
        covered_lines.append("")
        covered_lines.append("CALL_CONTEXT (analyst summary of the full call so far — use this to stay anchored to earlier discussion):")
        if reads.snapshot.themes:
            covered_lines.append(f"  Themes: {'; '.join(reads.snapshot.themes)}")
        if reads.snapshot.contradictions:
            covered_lines.append(f"  Contradictions: {'; '.join(reads.snapshot.contradictions)}")
        if reads.snapshot.investor_signals:
            covered_lines.append("  Key signals (reference these when probing or bridging):")
            for sig in reads.snapshot.investor_signals:
                covered_lines.append(f"    {sig}")

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

    dynamic_lines.append(
        "RECOMMENDED_NEXT (probe-vs-scripted priority already resolved; clarify and "
        "open threads still take precedence): " + _recommend_next(reads)
    )

    if recent:
        dynamic_lines.append("RECENT_TURNS:")
        for m in recent:
            speaker = "interviewer" if m["role"] == "assistant" else "respondent"
            dynamic_lines.append(f"  {speaker}: {m.get('content', '')}")
    dynamic_lines.append("[/CONTEXT]")
    dynamic_lines.append("")
    # Normalize blank/whitespace-only input so the model reliably detects silence
    display_text = respondent_text.strip() or "[silence]"
    dynamic_lines.append(f"Respondent: {display_text}")

    return [
        "\n".join(covered_lines),
        user_message_cache_breakpoint(),
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
        consecutive_clarify_count=state.consecutive_clarify_count(session, call_id),
    )
    prompt_parts = _build_prompt_parts_from_reads(
        reads,
        current_turn,
        respondent_text=respondent_text,
        vapi_messages=messages,
    )
    is_silence = not respondent_text.strip() or respondent_text.strip().lower() in ("[silence]", "silence")
    return PreparedInterviewerTurn(
        prompt_parts=prompt_parts,
        fallback_scripted_question=reads.next_scripted_question,
        respondent_text=respondent_text,
        has_pending_probes=bool(reads.probes),
        consecutive_clarify_count=reads.consecutive_clarify_count,
        is_silence=is_silence,
    )


async def prepare_interviewer_turn_concurrent(
    engine,
    call_id: str,
    current_turn: int,
    respondent_text: str,
    vapi_messages: list[dict] | None,
) -> PreparedInterviewerTurn:
    """Read context — parallel sessions for Postgres, single session for SQLite.

    StaticPool (used in evals and REPL) serializes all connections to one
    underlying SQLite connection. Running concurrent anyio threads against it
    causes SQLAlchemy cursor collisions (IndexError: tuple index out of range).
    Detect SQLite by pool class and fall back to a single sequential session.
    """
    from sqlalchemy.pool import StaticPool as _StaticPool

    if isinstance(engine.pool, _StaticPool):
        # SQLite / in-memory: single session, no threads
        with state.session_scope(engine) as session:
            return prepare_interviewer_turn(
                session, call_id, current_turn, respondent_text, vapi_messages
            )

    async def _read(fn, *args):
        with state.session_scope(engine) as session:
            return await anyio.to_thread.run_sync(functools.partial(fn, session, *args))

    next_q, remaining, probes, snapshot, clarify_count = await asyncio.gather(
        _read(state.next_scripted, call_id),
        _read(state.scripted_remaining, call_id),
        _read(state.top_probes, call_id, 3, current_turn - PROBE_STALENESS_TURNS),
        _read(state.latest_snapshot, call_id),
        _read(state.consecutive_clarify_count, call_id),
    )
    messages = vapi_messages or await _read(_db_messages_fallback, call_id)

    reads = InterviewerContextReads(
        next_scripted_question=next_q,
        scripted_remaining=remaining or 0,
        probes=probes or [],
        snapshot=snapshot,
        consecutive_clarify_count=clarify_count or 0,
    )
    prompt_parts = _build_prompt_parts_from_reads(
        reads,
        current_turn,
        respondent_text=respondent_text,
        vapi_messages=messages,
    )
    is_silence = not respondent_text.strip() or respondent_text.strip().lower() in ("[silence]", "silence")
    return PreparedInterviewerTurn(
        prompt_parts=prompt_parts,
        fallback_scripted_question=reads.next_scripted_question,
        respondent_text=respondent_text,
        has_pending_probes=bool(reads.probes),
        consecutive_clarify_count=reads.consecutive_clarify_count,
        is_silence=is_silence,
    )


# Bump when INTERVIEWER_PROMPT content changes so Logfire traces can be filtered by version.
INTERVIEWER_PROMPT_VERSION = "2026-06-11.1"

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
- COVERED_SUBTOPICS: specific subtopic labels already covered (explicit or organic).
  Labels name exact entities and dimensions, e.g. "Notion vs Google Docs product features".
  A covered label does NOT block other entities or dimensions — "Notion vs Quip" or
  "Notion vs Google Docs pricing" remain open even if product features is covered.
- CALL_CONTEXT: analyst summary of the entire call — themes, contradictions, and tagged
  investor signals (e.g. "[REVENUE] $50k annual spend", "[COMPETITIVE] churned from Google Docs").
  This covers the full call history, including turns before RECENT_TURNS. Use it to:
  (a) stay anchored to facts established earlier in the call,
  (b) avoid re-asking about signals already captured, and
  (c) bridge naturally ("Earlier you mentioned X...") when a new answer connects to an old thread.
- RECENT_TURNS: last several turns of conversation
- RECOMMENDED_NEXT: the probe-vs-scripted priority comparison, already resolved for you.
  Follow it at nodes 6/7 UNLESS clarify (node 2) or an open thread (node 5) applies first,
  or the recommended probe duplicates COVERED_SUBTOPICS. It is a default, not an override.

DECISION TREE — start at node 1 and read top to bottom. The FIRST node whose condition
matches decides your action: apply it and STOP — do not evaluate any node below it.
Each node depends only on what's above it, so you never look ahead or jump back.
In "reasoning", name the node that fired.

Quick map (condition → action):
  pure thinking-filler ("um", "let me think")     → clarify ("Take your time.")
  vague hedge, no substance                        → clarify
  explicit "I'm done" / "I need to go"             → wrap_up
  personal tangent, 2+ unrelated sentences          → off_topic
  unresolved thread in the answer                   → probe
  RECOMMENDED_NEXT points to a probe                → probe
  RECOMMENDED_NEXT is scripted, already answered    → skip_scripted
  RECOMMENDED_NEXT is scripted, not yet answered    → scripted
  nothing left                                      → wrap_up

NODE 1 — IS THERE SUBSTANTIVE CONTENT TO ACT ON?
   Brief confirmations — "yes", "yeah", "sure", "okay", "mm-hmm", "I can hear you" — are
   NOT silence. They are go-aheads: treat as content and continue to node 2.
   Pure thinking filler standing alone — "um", "uh", "hmm", "give me a second",
   "let me think" → action=`clarify`, say only "Take your time." But if the previous
   interviewer turn was already "Take your time." and they are still stalling, say only
   "Still there?" instead. Nothing else, no question appended.
   (Truly blank / "[silence]" input, the re-engagement escalation, and ending the call
   after repeated silence are handled deterministically BEFORE you are called — never
   your job to count silences or wrap up for silence.)
   Otherwise → continue to node 2.

NODE 2 — IS THE ANSWER TOO VAGUE TO ACT ON?
   Vague = present but content-free; your honest reaction is "what do you mean by that?"
   rather than "tell me more about that". You cannot probe or advance from an answer you
   don't understand, so this is settled before any node below.
   The blatant single-word non-answers ("Mixed.", "Fine.", "Not really.", "I guess.")
   are already caught for you. YOUR job is the harder case: a hedge with no real substance
   buried inside a longer sentence — "I mean, I guess it's fine, kind of", "sort of, I
   don't know, it just kind of works". Those are STILL vague.
   MATCH → action=`clarify`, ask for the meaning ("What do you mean by that?").
   NO MATCH → the answer is clear and substantive; continue to node 3.
   ("I dunno, it's fine I guess." → clarify. "You mentioned it saved you time — roughly
   how much?" → not vague, continue.)

NODE 3 — DOES THE CURRENT UTTERANCE EXPLICITLY SIGNAL THEY ARE DONE?
   "I need to go", "I'm done", "that's about it from me", "I've covered everything",
   "I have nothing else", "thanks, bye". The most explicit exits are also caught
   automatically.
   YES → action=`wrap_up`, warm and immediate.
   NO → continue to node 4. (Vague non-answers like "Not really" or "Fine" are NODE 2,
   not this — they never reach here.)

NODE 4 — ARE THEY ON A PERSONAL TANGENT?
   2+ consecutive sentences clearly unrelated to the study (pets, politics, family,
   unrelated complaints). A single off-topic sentence inside an otherwise on-topic
   answer is normal conversation, NOT this node.
   YES → action=`off_topic`: acknowledge briefly, steer back with one open question.
   NO → continue to node 5.

NODE 5 — DOES THE CLEAR ANSWER LEAVE A THREAD OPEN?
   Stay on the thread before moving on. Scan the ENTIRE utterance, not just the part that
   answered your question — a signal dropped in passing opens a thread even when the
   literal question was already answered. A rider clause is the highest-value probe:
     "About two years, though we almost didn't renew" → probe the near-cancellation; do
       NOT advance to scripted just because "how long" was answered.
     "I use them, but I always double-check the outputs" → probe the trust gap.
     "Training, I guess — it took longer than expected" → probe the unexplained delay.
   Ask: does the answer contain anything concrete still unexplained or unresolved? A
   thread is open when the answer contains:
   - A named thing with no story: a product, person, team, event, or competitor named
     but not elaborated ("we tried Gong", "my manager flagged it", "the security team
     pushed back", "the rollout stalled")
   - A tension or contrast: something working AND something not, or a before/after
     ("it was great at first, then...", "some teams use it, others don't")
   - A cause left implicit: something happened but the why wasn't given
     ("we stopped using that feature", "adoption dropped off", "we almost didn't renew")
   - A concrete detail that changes the picture: a number, a timeline, a role, a
     specific incident mentioned in passing
   - An investor signal trigger — referral, AI trust/verification gap, ROI or
     quantification claim, competitor mention, budget/approval path, expansion signal,
     or a red flag (see INVESTOR SIGNAL TRIGGERS below). These are ALWAYS open: probe
     before advancing, even when RECOMMENDED_NEXT is scripted.
   YES → action=`probe`. ONE layer at a time; never chain two probes on the same detail
   in consecutive turns.
   NO → continue to node 6. Treat the thread as closed when: the answer was complete and
   self-contained; you already probed this exact angle recently (check RECENT_TURNS); or
   the answer directly contradicts something said earlier (let the analyst probe at node 6
   handle the contradiction — don't layer on it mid-thread).

NODE 6 — DOES RECOMMENDED_NEXT POINT TO A PROBE?
   RECOMMENDED_NEXT has already resolved probe-vs-scripted priority for you.
   If it points to a probe, use that probe — UNLESS it duplicates COVERED_SUBTOPICS
   (then take the next candidate, or fall through to node 7).
   TURNS_AGO ≤ 2: use it directly, rephrase naturally.
   TURNS_AGO 3–8: bridge with "Earlier you mentioned X..." if needed.
   Set probe_id_used to the probe's exact id. → action=`probe`.
   Otherwise → continue to node 7.

NODE 7 — SCRIPTED, OR WRAP UP.
   RECOMMENDED_NEXT is scripted (or its probe duplicated covered ground). Before asking
   NEXT_SCRIPTED, check whether it was already answered organically — the respondent
   volunteered the information unprompted in any prior turn. Read RECENT_TURNS directly;
   COVERED_SUBTOPICS can lag (the analyst is async). Judge from the actual text.
   - Already answered organically → action=`skip_scripted` (acknowledge it and move on).
     e.g. NEXT_SCRIPTED "Would you recommend it?" but they said "I've already recommended
     it to three colleagues"; or "What do you value most?" answered by "the time savings —
     it cuts my note-taking in half". Re-asking an answered question is a significant
     quality failure.
   - Not yet answered → action=`scripted`. A small lead-in is fine; don't change the meaning.
   - SCRIPTED_REMAINING is 0 and no thread remains open → action=`wrap_up`.

--- INVESTOR SIGNAL TRIGGERS ---
These are the highest-value instances of node 5 (LAYERING). When you hear them,
the thread is always open — stay on it before advancing:

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
- NO REPETITION (applies at nodes 5, 6, and 7): never ask about a subtopic already
  addressed in RECENT_TURNS or COVERED_SUBTOPICS, unless the prior answer was incomplete
  or evasive. A broad topic being covered does not block adjacent subtopics ("competitor
  product features" covered does not block "competitor pricing structure").
- One question per utterance. Max one `?`. Never ask two questions in one turn, even as
  "and also..." constructions.
- Never ask a leading question — never presuppose the answer or push a view.
  ("So you loved it, right?" / "That must have been frustrating?" are both leading.)
  Always use open, neutral phrasing: "What happened?", "How did that feel?",
  "What was that like for you?"
- Keep utterances under 25 words — this is spoken audio, not text. Short is natural.
- Start every response with a brief spoken acknowledgment before your question such as
  "Right.", "Sure.", "Interesting.", "I see.", "Makes sense.", or a short restatement
  like "So it's mostly ad-hoc use —". Never open with a bare question. Match the
  acknowledgment to the content: for a simple yes/no confirmation ("Yes", "Sure",
  "I can hear you") use "Great." or "Perfect." — not "Hmm." Those sound
  off when there's nothing substantive to acknowledge.
- Speak numbers as words: "about thirty percent" not "30%", "three to five times" not "3–5x".
- If the respondent asks who you are or for your name, deflect warmly and briefly —
  "I'm Sagar/ I work at a market research firm" — then immediately ask your next question.
- Valid actions: scripted, probe, clarify, off_topic, wrap_up, skip_scripted. Do not use `acknowledge`
  as a standalone action — brief acknowledgments belong in the utterance itself before
  steering back.

OUTPUT FORMAT — machine parsing depends on an exact shape. Deviation breaks the interview.

Structure (in this order, nothing else):
1) Opening tag `<utterance>` as the very first characters of your entire output.
2) Spoken text only (no nested tags, no `</utterance>` inside the spoken text).
3) Closing tag `</utterance>`.
4) Exactly one newline.
5) Exactly one JSON object on the next line: minified, one line, double-quoted keys.

The JSON object MUST contain only these three keys (no others, no markdown fences):
- "action": one of scripted, probe, clarify, off_topic, wrap_up, skip_scripted
- "reasoning": one sentence naming the node that fired, e.g. "Node 6: RECOMMENDED_NEXT
  points to probe id=3." or "Node 2: hedge with no substance — clarifying before advancing."
  This field is never spoken; it exists for tracing and debugging only.
- "probe_id_used": either a positive integer matching a PENDING_PROBES id, or null

Examples of WRONG output (do not do this):
- Preamble before `<utterance>`
- Pretty-printed JSON spanning multiple lines
- Wrapping the JSON in ``` fences
- Trailing text, apologies, or a second JSON object after the first
- action "probe" with probe_id_used null when you followed a PENDING_PROBES suggestion (must copy the id)

Examples of CORRECT output (copy this shape; substitute your own strings):

scripted:
<utterance>Got it. Walk me through how your team actually uses the product day-to-day.</utterance>
{"action":"scripted","reasoning":"Node 7: no earlier node fired, advancing to next scripted question.","probe_id_used":null}

clarify (thinking filler):
<utterance>Take your time.</utterance>
{"action":"clarify","reasoning":"Node 1: pure thinking filler, no substantive content to act on.","probe_id_used":null}

clarify (thinking filler repeated — previous turn was already "Take your time."):
<utterance>Still there?</utterance>
{"action":"clarify","reasoning":"Node 1: respondent still stalling after Take your time. — escalating to Still there?","probe_id_used":null}

skip_scripted:
<utterance>Interesting. You mentioned earlier you've already recommended it to colleagues — what made you confident enough to do that?</utterance>
{"action":"skip_scripted","reasoning":"Node 7: NEXT_SCRIPTED topic already covered organically in RECENT_TURNS.","probe_id_used":null}
"""


def _log_and_fallback(exc: Exception) -> bool:
    """Log any provider API or structured-output error and fall back to the next model.

    Chain: OpenAI (optional) → Haiku → Gemini → Groq → Cerebras (when API key and model are configured).
    Returning True tells FallbackModel to try the next model.
    Returning False re-raises (for logic errors we don't want to silently swallow).
    """
    is_fallback_error = isinstance(exc, (GroqAPIError, ModelAPIError, UnexpectedModelBehavior))
    if not is_fallback_error:
        try:
            from openai import APIError as OpenAIAPIError
            is_fallback_error = isinstance(exc, OpenAIAPIError)
        except ImportError:
            pass
    if not is_fallback_error:
        try:
            from google.genai.errors import APIError as GoogleAPIError
            is_fallback_error = isinstance(exc, GoogleAPIError)
        except ImportError:
            pass
    if is_fallback_error:
        logfire.warning(
            "interviewer_model_error",
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        return True
    return False


def _build_interviewer_model() -> FallbackModel:
    """Build model chain: OpenAI (optional) → Haiku → Gemini → Groq → Cerebras (optional).

    Each model is constructed explicitly with its provider so API keys are read
    from settings (pydantic-settings / .env) rather than requiring them in os.environ
    at import time.
    """
    def _strip(model_str: str) -> str:
        """'provider:model-name' → 'model-name' (bare model id still accepted)."""
        return model_str.split(":", 1)[1] if ":" in model_str else model_str

    chain: list = []
    openai_id = _strip(INTERVIEWER_OPENAI_MODEL).strip()
    if settings.openai_api_key and openai_id:
        chain.append(
            OpenAIChatModel(
                openai_id,
                provider=OpenAIProvider(api_key=settings.openai_api_key),
                settings=openai_interviewer_settings(settings),
            )
        )

    temp = settings.interviewer_temperature
    haiku = AnthropicModel(
        _strip(INTERVIEWER_HAIKU_MODEL),
        provider=AnthropicProvider(api_key=settings.anthropic_api_key),
        settings=anthropic_interviewer_settings(temperature=temp),
    )
    # thinking_budget=0 caps unbounded reasoning that can spike to 12s+ on complex turns.
    gemini = GoogleModel(
        _strip(INTERVIEWER_GEMINI_MODEL),
        provider=GoogleProvider(api_key=settings.google_api_key),
        settings=GoogleModelSettings(thinking_config={'thinking_budget': 0}, temperature=temp),
    )
    temp_settings = {"temperature": temp} if temp is not None else None
    groq = GroqModel(
        _strip(INTERVIEWER_GROQ_MODEL),
        provider=GroqProvider(api_key=settings.groq_api_key),
        settings=temp_settings,
    )

    chain.extend([haiku, gemini, groq])
    cerebras_id = _strip(INTERVIEWER_CEREBRAS_MODEL).strip()
    if settings.cerebras_api_key and cerebras_id:
        chain.append(
            CerebrasModel(
                cerebras_id,
                provider=CerebrasProvider(api_key=settings.cerebras_api_key),
                settings=temp_settings,
            )
        )

    return FallbackModel(*chain, fallback_on=_log_and_fallback)


_interviewer: Agent | None = None


def _get_interviewer() -> Agent:
    """Return the interviewer Agent, constructing it on first call.

    Lazy init so that settings (including INTERVIEWER_TEMPERATURE) are read
    after load_dotenv() has run, not at module import time. Tests that patch
    env vars before the first call get the correct model config.
    """
    global _interviewer
    if _interviewer is None:
        _interviewer = Agent(
            _build_interviewer_model(),
            deps_type=InterviewerDeps,
            output_type=str,
            system_prompt=INTERVIEWER_PROMPT,
            instrument=True,
        )
    return _interviewer


async def run_interviewer(
    deps: InterviewerDeps,
    respondent_text: str,
    vapi_messages: list[dict] | None = None,
    prepared: PreparedInterviewerTurn | None = None,
    session=None,
) -> InterviewerOutput:
    """Pre-fetch context → one LLM call → return output.

    vapi_messages: OpenAI-formatted message array from Vapi (body["messages"]).
    When None (evals / play.py), falls back to reading recent_turns from the DB.
    session: only needed when prepared is None (evals / REPL path).
    """
    if prepared is None:
        assert session is not None, "session required when prepared is not provided"
        prepared = prepare_interviewer_turn(
            session,
            deps.call_id,
            deps.turn_number,
            respondent_text=respondent_text,
            vapi_messages=vapi_messages or _db_messages_fallback(session, deps.call_id),
        )
    sc = _silence_short_circuit(prepared)
    if sc is not None:
        return sc
    result = await _get_interviewer().run(prepared.prompt_parts, deps=deps)
    output = _parse_streamed_output(result.output, prepared.fallback_scripted_question)
    return _apply_overrides(output, prepared, prepared.respondent_text)


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


_OPEN_TAG = "<utterance>"
_CLOSE_TAG = "</utterance>"
_META_RE = re.compile(r"</utterance>\s*(\{.*)", re.DOTALL)
_META_BARE_RE = re.compile(r"\n(\{.*)", re.DOTALL)  # tag-free fallback: JSON after first \n{

_WRAP_UP_UTTERANCE = "It sounds like now might not be a great time — thanks so much for your time today. I'll let you go."

# Deterministic re-engagement ladder for empty/[silence] turns, indexed by how many
# consecutive re-engagement attempts (clarify turns) have already been made.
# Index 2 must contain "still there" so the turn immediately before a silence wrap_up
# is a "Still there?" — the trajectory eval keys its wrap-after-silence detection off that.
_SILENCE_LADDER = ("Take your time.", "Still there?", "Still there?")


def _silence_short_circuit(prepared: PreparedInterviewerTurn) -> InterviewerOutput | None:
    """Resolve empty/[silence] turns deterministically, without an LLM call.

    Truly blank input carries nothing the model can act on, so the re-engagement
    ladder and the wrap-up after repeated silence are owned by Python: instant,
    free, and 100% reliable on the highest-frequency degenerate turn. Thinking
    fillers ("um", "uh") are NOT is_silence and still go to the model.

    Returns None for any non-silence turn (the normal LLM path).
    """
    if not prepared.is_silence:
        return None
    n = prepared.consecutive_clarify_count
    if n >= len(_SILENCE_LADDER):
        return InterviewerOutput(
            utterance=_WRAP_UP_UTTERANCE,
            action="wrap_up",
            reasoning=f"Python silence handler: consecutive_clarify_count={n} — wrapping up after repeated silence",
            probe_id_used=None,
        )
    return InterviewerOutput(
        utterance=_SILENCE_LADDER[n],
        action="clarify",
        reasoning=f"Python silence handler: empty/[silence] input, re-engagement attempt {n + 1} of {len(_SILENCE_LADDER)}",
        probe_id_used=None,
    )


# Verbal-exit phrases that override the model's action to wrap_up (step 5 of the decision framework).
# These are explicit verbal signals, not vague non-answers — keep this list tight.
_VERBAL_EXIT_RE = re.compile(
    r"\b("
    r"i (need|have) to go|"
    r"thanks[,.]? ?bye"
    r")\b",
    re.IGNORECASE,
)

# Single-word or near-single-word vague answers that are always clarify triggers (step 4).
# Matches the *complete* stripped utterance — not a substring check.
# Each alternative is a concrete phrase so the pattern cannot match the empty string.
_VAGUE_ANSWER_RE = re.compile(
    r"^("
    r"mixed|fine|maybe|sure|kind of|not really|i guess|i guess so|"
    r"sort of|i don'?t know|i dunno|hard to say|"
    r"i mean[,.]?|i mean[,.]? i guess|"
    r"it'?s? fine|it'?s? fine[,.]? i guess|it'?s? fine[,.]? kind of|"
    r"i guess[,.]? kind of|kind of[,.]? i don'?t know|i guess so[,.]? kind of"
    r")\.?$",
    re.IGNORECASE,
)


def _apply_overrides(
    output: InterviewerOutput,
    prepared: PreparedInterviewerTurn,
    respondent_text: str,
) -> InterviewerOutput:
    """Apply deterministic post-processing guards over the LLM's action choice.

    The prompt instructs the model on both rules below, but small fallback models
    (Groq, Cerebras) miss them under pressure. Python guards make the highest-stakes
    rules hard invariants regardless of model compliance.

    Empty/[silence] turns never reach here — `_silence_short_circuit` resolves the
    re-engagement ladder and silence wrap-up before the model is ever called.

    Guards run in priority order — first match wins:
      1. Verbal exit   — wrap_up on explicit "I'm done / I need to go" signals.
      2. Vague clarify — clarify on single-word / hedge-only answers when model probed or advanced.
    """
    # Guard 1: verbal exit — respondent explicitly signals they are done
    if _VERBAL_EXIT_RE.search(respondent_text) and output.action != "wrap_up":
        logfire.info("override_verbal_exit", model_action=output.action, respondent_snippet=respondent_text[:80])
        return InterviewerOutput(
            utterance="Thanks so much for your time — that's really helpful. I'll let you go.",
            action="wrap_up",
            reasoning="Python override: respondent verbal exit detected — forcing wrap_up",
            probe_id_used=None,
        )

    # Guard 2: vague-answer clarify gate — model probed or advanced from an answer that needs clarifying first
    if (
        _VAGUE_ANSWER_RE.match(respondent_text.strip())
        and output.action in ("probe", "scripted", "skip_scripted", "off_topic")
    ):
        logfire.info("override_vague_clarify", model_action=output.action, respondent_snippet=respondent_text[:80])
        return InterviewerOutput(
            utterance="What do you mean by that?",
            action="clarify",
            reasoning="Python override: vague single-word/hedge answer — must clarify before probing or advancing",
            probe_id_used=None,
        )

    return output


def _extract_parts(text: str) -> tuple[str, str, bool]:
    """Split raw model output into (utterance, meta_json, wire_delimited).

    Three formats, tried in order:
      1. Tagged:   <utterance>…</utterance>  then JSON
      2. Bare:     plain text, then \\n{ JSON
      3. Fallback: entire text is the utterance, no metadata
    Returns wire_delimited=True for formats 1 and 2.
    """
    # Format 1: explicit tags
    tag_m = re.search(r"<utterance>(.*?)</utterance>", text, re.DOTALL)
    if tag_m:
        utterance = tag_m.group(1).strip()
        meta_m = _META_RE.search(text)
        meta_json = meta_m.group(1).strip() if meta_m else ""
        return utterance, meta_json, True

    # Format 2: tag-free — utterance before \n{
    bare_m = _META_BARE_RE.search(text)
    if bare_m:
        utterance = text[:bare_m.start()].strip()
        meta_json = bare_m.group(1).strip()
        return utterance, meta_json, True

    # Format 3: no delimiter at all
    return text.strip(), "", False


def _parse_meta(meta_json: str) -> InterviewerLLMMeta:
    """Parse the JSON metadata block into InterviewerLLMMeta.

    Returns default (action=scripted) on any parse failure.
    """
    raw = json.loads(meta_json) if meta_json.strip() else {}
    if not isinstance(raw, dict):
        raise ValueError("metadata JSON must be an object")
    return InterviewerLLMMeta.model_validate(raw)


def _parse_streamed_output(
    text: str,
    fallback_scripted_question: str | None = None,  # noqa: ARG001
    has_pending_probes: bool = False,
) -> InterviewerOutput:
    """Parse <utterance>...</utterance> + trailing JSON from a plain-text model response."""
    utterance, meta_json, wire_delimited = _extract_parts(text)

    # Guard: whole-body JSON with no spoken utterance — never send to TTS.
    if not wire_delimited and utterance.startswith("{") and '"action"' in utterance:
        try:
            blob = json.loads(utterance)
            if isinstance(blob, dict) and "action" in blob:
                logfire.warning("interviewer_json_only_body", text_snippet=utterance[:200])
                return InterviewerOutput(
                    utterance=INTERVIEWER_RECOVERY_UTTERANCE,
                    action="scripted",
                    reasoning="model returned JSON without a spoken utterance block",
                    probe_id_used=None,
                )
        except (json.JSONDecodeError, TypeError):
            pass

    # Guard: empty utterance — nothing safe to send to TTS.
    spoken = utterance.strip()
    if not spoken:
        logfire.warning("interviewer_empty_utterance", wire_delimited=wire_delimited)
        return InterviewerOutput(
            utterance=INTERVIEWER_RECOVERY_UTTERANCE,
            action="scripted",
            reasoning="empty utterance from model",
            probe_id_used=None,
            probe_source=None,
        )

    # Parse metadata; fall back to default action on failure.
    try:
        meta = _parse_meta(meta_json)
    except (json.JSONDecodeError, ValueError, ValidationError) as exc:
        logfire.warning(
            "interviewer_metadata_parse_failed",
            text_snippet=text[:300],
            error_type=type(exc).__name__,
        )
        return InterviewerOutput(
            utterance=spoken,
            action="scripted",
            reasoning="metadata parse failed",
            probe_id_used=None,
        )

    probe_source = None
    if meta.action == "probe":
        probe_source = "analyst" if meta.probe_id_used is not None else "interviewer"

    return InterviewerOutput(
        utterance=spoken,
        action=meta.action,
        reasoning=meta.reasoning,
        probe_id_used=meta.probe_id_used,
        probe_source=probe_source,
    )


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
        self._usage: RunUsage | None = None

    @property
    def usage(self) -> RunUsage | None:
        """Populated after a successful model run; None on timeout, API error fallback, or no call."""
        return self._usage

    @property
    def output(self) -> InterviewerOutput:
        if self._output is None:
            raise RuntimeError("InterviewerStream.output accessed before tokens() was fully consumed")
        return self._output

    async def tokens(self) -> AsyncGenerator[str, None]:
        """Yield utterance tokens as they stream in.

        Model outputs <utterance>...</utterance> as plain text (no JSON schema,
        so all providers stream token-by-token), followed by a JSON metadata block
        used only for DB commit. We yield tokens inside the tags and stop there;
        the full text is parsed into InterviewerOutput after the stream completes.
        """
        deps = self._deps
        prepared = self._prepared
        _stream_error: Exception | None = None
        full_text = ""
        any_yielded = False

        # Empty/[silence] turns are resolved deterministically without a model call:
        # instant first token (no filler needed), zero cost, 100% reliable escalation.
        sc = _silence_short_circuit(prepared)
        if sc is not None:
            self._output = sc
            self._usage = None
            logfire.info(
                "interviewer_silence_short_circuit",
                call_id=deps.call_id,
                turn_number=deps.turn_number,
                action=sc.action,
                consecutive_clarify_count=prepared.consecutive_clarify_count,
            )
            yield sc.utterance
            return

        with anyio.move_on_after(self._budget_s) as cancel_scope:
            try:
                async with _get_interviewer().run_stream(prepared.prompt_parts, deps=deps) as streamed:
                    in_utterance = False
                    utterance_done = False
                    bare_mode = False  # True when model omits <utterance> tags
                    carry = ""

                    async for delta in streamed.stream_text(delta=True, debounce_by=None):
                        full_text += delta

                        if utterance_done:
                            continue

                        carry += delta

                        if not in_utterance and not bare_mode:
                            idx = carry.find(_OPEN_TAG)
                            if idx >= 0:
                                in_utterance = True
                                carry = carry[idx + len(_OPEN_TAG):]
                                # Fall through to in_utterance block below
                            elif len(carry) > len(_OPEN_TAG):
                                # Opening tag not found after enough chars — model skipped tags.
                                # This is a format regression on the primary model; log it so
                                # Logfire makes the failure visible rather than silently succeeding.
                                bare_mode = True
                                logfire.warning(
                                    "interviewer_bare_mode",
                                    call_id=deps.call_id,
                                    turn_number=deps.turn_number,
                                    carry_snippet=carry[:80],
                                )
                                # Fall through to bare_mode block below
                            else:
                                continue

                        if in_utterance:
                            # Inside tagged utterance — yield safe prefix, hold back enough
                            # chars to detect a split </utterance> close tag
                            idx = carry.find(_CLOSE_TAG)
                            if idx >= 0:
                                utterance_done = True
                                if idx > 0:
                                    yield carry[:idx]
                                    any_yielded = True
                                carry = ""
                            else:
                                safe_end = max(0, len(carry) - (len(_CLOSE_TAG) - 1))
                                if safe_end > 0:
                                    yield carry[:safe_end]
                                    any_yielded = True
                                    carry = carry[safe_end:]

                        elif bare_mode:
                            # Tag-free format: yield until \n{ marks the start of the JSON block
                            idx = carry.find('\n{')
                            if idx >= 0:
                                utterance_done = True
                                if idx > 0:
                                    yield carry[:idx]
                                    any_yielded = True
                                carry = ""
                            else:
                                # Hold back 1 char so \n{ can be detected across delta boundaries
                                safe_end = max(0, len(carry) - 1)
                                if safe_end > 0:
                                    yield carry[:safe_end]
                                    any_yielded = True
                                    carry = carry[safe_end:]

                    self._output = _apply_overrides(
                        _parse_streamed_output(full_text, prepared.fallback_scripted_question),
                        prepared,
                        prepared.respondent_text,
                    )
                    self._usage = streamed.usage()
                    # Safety: if the model skipped both tag and bare formats entirely, yield the
                    # parsed utterance as a single chunk so TTS always gets something.
                    if not any_yielded and self._output:
                        yield self._output.utterance
                        any_yielded = True
            except (GroqAPIError, ModelAPIError) as exc:
                _stream_error = exc
            except Exception as exc:  # noqa: BLE001
                # Catches provider errors not in the known set (e.g. google.genai.errors.ServerError)
                # that escape _log_and_fallback and the FallbackModel chain.
                # RuntimeError from pydantic-graph's anyio TaskGroup is the one exception:
                # if the call actually succeeded (self._output is set), don't treat it as an error.
                if isinstance(exc, RuntimeError) and (
                    "Attempted to exit cancel scope in a different task" in str(exc)
                    and self._output is not None
                ):
                    pass
                else:
                    _stream_error = exc

        if cancel_scope.cancelled_caught:
            self._usage = None
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

        if _stream_error is not None:
            self._usage = None
            logfire.warning(
                "interviewer_model_error_fallback",
                call_id=deps.call_id,
                turn_number=deps.turn_number,
                error=str(_stream_error),
            )
            fb = _fallback(deps, fallback_scripted_question=prepared.fallback_scripted_question)
            self._output = fb
            if not any_yielded:
                yield fb.utterance
            return


def _fallback(
    deps: InterviewerDeps,
    *,
    fallback_scripted_question: str | None = None,
) -> InterviewerOutput:
    q = fallback_scripted_question
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
            deps = InterviewerDeps(call_id=call_id, turn_number=turn_number)
            with agent_span("interviewer", call_id, turn_number=turn_number, respondent_text=line) as span:
                t0 = time.perf_counter()
                out = asyncio.run(run_interviewer(deps, line, session=session))
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
