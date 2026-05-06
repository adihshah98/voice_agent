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
    has_pending_probes: bool = False


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
    # Order: COVERED_SUBTOPICS, cache breakpoint (Anthropic), then per-turn CONTEXT. See interviewer_llm_caching.
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
        has_pending_probes=bool(reads.probes),
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
        has_pending_probes=bool(reads.probes),
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

-1. SILENCE / THINKING PAUSE:
    - SHORT AFFIRMATIONS are NOT silence. If the utterance is "yes", "yeah", "sure",
      "okay", "mm-hmm", "I can hear you", or any other brief confirmation — treat it as
      a go-ahead and proceed directly to step 4 (SCRIPTED). Do not say "Still there?".
    - LOOP GUARD: check RECENT_TURNS. If the interviewer has already said "Still there?"
      or "Take your time." in the last 2 turns, skip this rule and go to step 4
      (SCRIPTED) — repeating the same clarify is never helpful.
    - If the utterance is truly empty or silence (blank, "[silence]", no transcribed
      words at all) — respond with only "Still there?" and use action=`clarify`.
    - If the utterance is a pure thinking filler with no other words — "um", "uh",
      "let me think", "give me a second", "hmm" standing alone — respond with only
      "Take your time." and use action=`clarify`.
    In true silence/filler cases: no question appended, nothing else added.

0. NO REPETITION — before choosing any action, check RECENT_TURNS and COVERED_SUBTOPICS.
   If the specific subtopic you're about to ask about was already addressed, skip it and
   move to the next step, unless the answer was incomplete or evasive. A broad topic
   being covered does not block adjacent subtopics (e.g. "competitor product features"
   covered does not block "competitor pricing structure").

1. OFF-TOPIC: If the respondent went on a personal tangent unrelated to the study,
   acknowledge briefly and steer back with one open question. Use `off_topic`.

2. IMMEDIATE FOLLOW-UP: If the respondent just said something that matches an investor
   signal trigger below (referral, AI trust, ROI, competitor, budget, expansion, red flag)
   OR stated a direct contradiction with what they said earlier — probe it NOW.
   Use action=`probe`.
   SKIP this step if: the answer is too vague to understand (go to step 5 CLARIFY), or
   if it's a closing/dismissive statement that wraps up the topic you just asked about
   ("it's fine now", "not really", "I guess so", "never mind").

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

5. CLARIFY: If the answer is too vague to understand — use action=`clarify`.
   Triggers: single words ("Mixed.", "Fine.", "Maybe."), hedges without substance
   ("it's fine I guess, kind of", "sort of", "I don't know"), or any answer where
   you'd need to ask "what do you mean by that?" before you could usefully probe.
   KEY DISTINCTION — clarify asks for the *meaning* of a vague answer; probe digs
   deeper into a clear one. "What do you mean by mixed?" → clarify.
   "You mentioned it saved you time — roughly how much?" → probe.
   Do NOT use `probe` when the answer itself is unclear.

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
- "reasoning": one short sentence (not spoken)
- "probe_id_used": either a positive integer matching a PENDING_PROBES id, or null

Examples of WRONG output (do not do this):
- Preamble before `<utterance>`
- Pretty-printed JSON spanning multiple lines
- Wrapping the JSON in ``` fences
- Trailing text, apologies, or a second JSON object after the first
- action "probe" with probe_id_used null when you followed a PENDING_PROBES suggestion (must copy the id)

Example of CORRECT output (copy this shape; substitute your own strings):
<utterance>Got it. Walk me through how your team actually uses the product day-to-day.</utterance>
{"action":"scripted","reasoning":"Moving to the first scripted question.","probe_id_used":null}
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

    haiku = AnthropicModel(
        _strip(INTERVIEWER_HAIKU_MODEL),
        provider=AnthropicProvider(api_key=settings.anthropic_api_key),
        settings=anthropic_interviewer_settings(),
    )
    # thinking_budget=0 caps unbounded reasoning that can spike to 12s+ on complex turns.
    gemini = GoogleModel(
        _strip(INTERVIEWER_GEMINI_MODEL),
        provider=GoogleProvider(api_key=settings.google_api_key),
        settings=GoogleModelSettings(thinking_config={'thinking_budget': 0}),
    )
    groq = GroqModel(
        _strip(INTERVIEWER_GROQ_MODEL),
        provider=GroqProvider(api_key=settings.groq_api_key),
    )

    chain.extend([haiku, gemini, groq])
    cerebras_id = _strip(INTERVIEWER_CEREBRAS_MODEL).strip()
    if settings.cerebras_api_key and cerebras_id:
        chain.append(
            CerebrasModel(
                cerebras_id,
                provider=CerebrasProvider(api_key=settings.cerebras_api_key),
            )
        )

    return FallbackModel(*chain, fallback_on=_log_and_fallback)


interviewer = Agent(
    _build_interviewer_model(),
    deps_type=InterviewerDeps,
    output_type=str,
    system_prompt=INTERVIEWER_PROMPT,
    instrument=True,
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
    return _parse_streamed_output(result.output, prepared.fallback_scripted_question)


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


def _parse_streamed_output(
    text: str,
    fallback_scripted_question: str | None = None,  # noqa: ARG001
    has_pending_probes: bool = False,
) -> InterviewerOutput:
    """Parse <utterance>...</utterance> + trailing JSON from a plain-text model response.

    Falls back to bare format (utterance text then newline + JSON) when tags are absent.
    """
    tag_m = re.search(r"<utterance>(.*?)</utterance>", text, re.DOTALL)
    wire_delimited = False
    if tag_m:
        utterance = tag_m.group(1).strip()
        meta_m = _META_RE.search(text)
        meta_json = meta_m.group(1).strip() if meta_m else ""
        wire_delimited = True
    else:
        bare_m = _META_BARE_RE.search(text)
        if bare_m:
            utterance = text[:bare_m.start()].strip()
            meta_json = bare_m.group(1).strip()
            wire_delimited = True
        else:
            utterance = text.strip()
            meta_json = ""

    # Whole-body JSON (no tags and no bare newline-before-{ split) — never send to TTS as speech.
    if not wire_delimited:
        u = utterance.strip()
        if u.startswith("{") and '"action"' in u:
            try:
                blob = json.loads(u)
                if isinstance(blob, dict) and "action" in blob:
                    logfire.warning(
                        "interviewer_json_only_body",
                        text_snippet=u[:200],
                    )
                    return InterviewerOutput(
                        utterance=INTERVIEWER_RECOVERY_UTTERANCE,
                        action="scripted",
                        reasoning="model returned JSON without a spoken utterance block",
                        probe_id_used=None,
                    )
            except (json.JSONDecodeError, TypeError):
                pass

    try:
        raw = json.loads(meta_json) if meta_json.strip() else {}
        if not isinstance(raw, dict):
            raise ValueError("metadata JSON must be an object")
        meta = InterviewerLLMMeta.model_validate(raw)
        if meta.action == "probe":
            probe_source = "analyst" if meta.probe_id_used is not None else "interviewer"
        else:
            probe_source = None
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
        return InterviewerOutput(
            utterance=spoken,
            action=meta.action,
            reasoning=meta.reasoning,
            probe_id_used=meta.probe_id_used,
            probe_source=probe_source,
        )
    except (json.JSONDecodeError, ValueError, ValidationError) as exc:
        logfire.warning(
            "interviewer_metadata_parse_failed",
            text_snippet=text[:300],
            error_type=type(exc).__name__,
        )
        spoken = (utterance or "").strip()
        if spoken:
            return InterviewerOutput(
                utterance=spoken,
                action="scripted",
                reasoning="metadata parse failed",
                probe_id_used=None,
            )
        return InterviewerOutput(
            utterance=INTERVIEWER_RECOVERY_UTTERANCE,
            action="scripted",
            reasoning="metadata parse failed",
            probe_id_used=None,
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
        assert self._output is not None, "tokens() must be fully consumed before reading output"
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

        with anyio.move_on_after(self._budget_s) as cancel_scope:
            try:
                async with interviewer.run_stream(prepared.prompt_parts, deps=deps) as streamed:
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
                                # Opening tag not found after enough chars — model is tag-free
                                bare_mode = True
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

                    self._output = _parse_streamed_output(full_text, prepared.fallback_scripted_question)
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
