# Architecture: Voice Research Interviewer

An AI-powered outbound interviewer that conducts structured investor/user research calls over the phone using [Vapi](https://vapi.ai) for telephony and [PydanticAI](https://ai.pydantic.dev) for agent orchestration.

---

## System Overview

```
  Respondent (phone)        Vapi (telephony)           Our Server (FastAPI)       Background tasks
  ──────────────────        ────────────────           ────────────────────       ────────────────

  ── 1. SETUP ───────────────────────────────────────────────────────────────────────────────────
                                               POST /calls/start
                                          ◄────────────────────────  create Call row in DB
                            Vapi API ◄─── _dial_vapi()               store scripted questions
  ◄── phone rings ──────────             (returns vapi_call_id)
                            │
                            ├─ webhook: status-update ──────────────► call.status = "active"
                            │   (ringing → in-progress)

  ── 2. PER TURN (repeats) ──────────────────────────────────────────────────────────────────────
  ── speaks ───────────────►│
                            │  Deepgram transcribes
                            │
                            ├─ POST /vapi/llm/chat/completions ─────► TurnPipeline (SSE)
                            │   {messages: [...], call: {metadata:    ├─ next_turn_number (COUNT+1)
                            │    {call_id: "..."}}}                   ├─ fetch context: next_q,
                            │                                         │   probes, snapshot (DB)
                            │                                         └─ interviewer LLM call
                            │                                              (Haiku 4.5, 5s budget)
                            │◄── SSE chunks (OpenAI format) ──────────
                            │    delta: "Tell me more about..."
  ◄── TTS audio ────────────│    ...
                            │    finish_reason: "stop"
                            │    data: [DONE]
                            │
                            │    after stream: commit() persists       ──► Turn rows (respondent + interviewer)
                            │              Turn rows to DB               ──► analyst LLM if should_run_analyst()
                            │                                                                    (Sonnet 4.6)
                            │                                          scripted cursor advanced      write AnalystSnapshot
                            │                                          OR ≥10 interviewer turns      write Probe rows
                            │                                          since last snapshot           ◄── new probes in DB
                            │
                            ├─ webhook: conversation-update ─────────► debug log only (no DB writes here)

  ── 3. END OF CALL ─────────────────────────────────────────────────────────────────────────────
  ── hangs up ─────────────►│
                            ├─ webhook: end-of-call-report ─────────► call.status = "ended"
                                                                                                ──► synthesis LLM
                                                                       (if ENABLE_SYNTHESIS_       (Sonnet 4.6)
                                                                        REPORT = True)          write SynthesisReport

  ── ALTERNATIVE: local REPL (no server, no Vapi) ───────────────────────────────────────────────
  scripts/play.py ──────────────────────────────────────────────────► run_speech_turn() → same TurnPipeline.commit()
  (terminal I/O)                                                       (persists Turn rows to DB)
```

**Core invariant:** Agents never call each other. All coordination happens through the DB. The interviewer reads from `calls`, `probes`, `analyst_snapshots`; the analyst writes to `analyst_snapshots` and `probes`; synthesis writes to `synthesis_reports`.

---

## Vapi Server Events

**1.** `speech-update` 

The `speech-update` event fires when speaking starts or stops for either the user or the assistant **[1](https://docs.vapi.ai/server-url/events)**:

```

```


|                                      |
| ------------------------------------ |
| {                                    |
| "message": {                         |
| "type": "speech-update",             |
| "status": "started", // or "stopped" |
| "role": "assistant", // or "user"    |
| "turn": 2                            |
| }                                    |
| }                                    |


**2.** `conversation-update` **—** 

Basically when the speech is detected & written down to Vapi msg history. The `conversation-update` event is sent when an update is committed to the conversation history. This is fired pretty much per speech-update (let's say user pauses in middle, and restarts, this will be two speech updates & 2 conversation updates too) :

```

```


|                                                                |
| -------------------------------------------------------------- |
| {                                                              |
| "message": {                                                   |
| "type": "conversation-update",                                 |
| "messages": [ /* current conversation messages */ ],           |
| "messagesOpenAIFormatted": [ /* openai-formatted messages */ ] |
| }                                                              |
| }                                                              |


The key nuance: it fires when *any* update is committed to the history — this includes user messages, assistant messages, tool call results, and system messages. It's not strictly one event per user turn and one per AI turn; it reflects the full conversation state at the time of the commit.


|                                                        |
| ------------------------------------------------------ |
| User speaks                                            |
| → speech-update (role: "user", status: "started")      |
| → speech-update (role: "user", status: "stopped")      |
| → [Endpointing evaluates completion]                   |
| → LLM called                                           |
| → conversation-update (user message committed)         |
| → speech-update (role: "assistant", status: "started") |
| → conversation-update (assistant message committed)    |
| → speech-update (role: "assistant", status: "stopped") |


The ordering of `conversation-update` relative to `speech-update` can vary slightly depending on when Vapi commits each message to history.

**3. LLM called after endpointing** 

The LLM is called immediately after the endpointing system determines the user has finished speaking. The pipeline is **[2](https://docs.vapi.ai/customization/voice-pipeline-configuration)**:

```

```


|                                                                                                                                               |
| --------------------------------------------------------------------------------------------------------------------------------------------- |
| User stops speaking → VAD detects utterance-stop → Endpointing decision → LLM request sent immediately → TTS → waitSeconds → Assistant speaks |


The endpointing decision itself can be driven by smart endpointing (AI-based), transcription-based heuristics, or custom rules — but yes, the LLM is only invoked once Vapi is confident the turn is complete.

Vapi sends `messagesOpenAIFormatted` to the LLM endpoint, which enforces OpenAI's strict alternating user/assistant requirement — so Vapi concatenates consecutive same-role segments into one. The `conversation-update` webhook sends the raw native format, preserving every VAD pause boundary as a separate entry.

## The Three Agents

All three are PydanticAI agents with Anthropic backends. Models and flags live in `[voice_agent/config.py](../voice_agent/config.py)` — nothing else needs to change when swapping a model.

### 1. Interviewer — `voice_agent/agents/interviewer.py`

The only real-time agent. Runs synchronously from the caller's perspective (Vapi waits for a response).


| Property    | Value                                                             |
| ----------- | ----------------------------------------------------------------- |
| Model       | `claude-haiku-4-5-20251001`                                       |
| Deadline    | 5 s (`anyio.move_on_after`)                                       |
| Output type | `InterviewerOutput` (utterance, action, reasoning, probe_id_used) |
| Fallback    | Next scripted question (or `wrap_up` if none remain)              |


**Not a tool loop.** All DB state is pre-fetched into a text `[CONTEXT]` block before the LLM call. The model returns a single structured output; Python applies side effects after. This was a deliberate deviation from the original PLAN.md design (which described a ReAct tool loop) — it eliminates the latency risk of multi-round tool calls within a 5 s budget.

**Context pre-fetch (concurrent):** `prepare_interviewer_turn_concurrent()` opens 5 parallel short-lived DB sessions using `anyio` task groups to read: next scripted question, scripted remaining count, top 3 probes, latest analyst snapshot, and Vapi message history. Anyio task groups are used (not `asyncio.gather`) because PydanticAI's internals also use anyio — mixing the two can cause `ClosedResourceError`.

**Prompt structure (3 parts, with cache point):**

```
Part 1: [COVERED_SUBTOPICS] block  ← static per-call, cached at 1h TTL
        CachePoint(ttl="1h")
Part 2: [CONTEXT] block            ← dynamic per-turn (never cached)
        SCRIPTED_REMAINING
        NEXT_SCRIPTED
        PENDING_PROBES (staleness ≤ 8 turns)
        RECENT_TURNS (last 30 from Vapi messages or DB)
        "Respondent: {text}"
```

The cache point lets Anthropic's prompt cache reuse the covered-subtopics block across turns in the same call, saving latency and cost.

**Action taxonomy:** `scripted | probe | clarify | off_topic | wrap_up | skip_scripted`

- `scripted` / `skip_scripted` — advance `scripted_cursor` on the `Call` row.
- `probe` — if `probe_id_used` is set, marks that `Probe` row as asked. If no ID, the model generated its own probe inline (not from the analyst queue).
- `wrap_up` — signals end of interview; no DB side effects.

**Decision priority (from system prompt):**

1. Off-topic redirect
2. Immediate probe on detected signal (referral, AI trust, ROI, competitor, budget, expansion, red flag)
3. Analyst-queued probe from `PENDING_PROBES`
4. Next scripted question (or `skip_scripted` if already answered)
5. Clarification
6. Wrap-up

### 2. Analyst — `voice_agent/agents/analyst.py`

Async, fire-and-forget. Never blocks a turn response.


| Property    | Value                                           |
| ----------- | ----------------------------------------------- |
| Model       | `claude-sonnet-4-6`                             |
| Deadline    | None (runs as background `asyncio.create_task`) |
| Output type | `AnalysisUpdate`                                |
| Trigger     | `should_run_analyst()` — see below              |


**Trigger condition** (`state.should_run_analyst()`): True if either:

- `scripted_cursor > snapshot.after_scripted_cursor` (a new scripted question was asked since last analysis), OR
- At least `ANALYST_TURN_INTERVAL` (10) **interviewer** turns have elapsed since the last snapshot (`after_turn`) — counts `Turn` rows with `speaker == "interviewer"` after `after_turn`, not a `turn_count` column on `Call`.

**Where it runs:** Evaluated inside `TurnPipeline.commit()` after turns are inserted. If true, `server.py` schedules `run_analyst_safely` from the `/vapi/llm/chat/completions` handler after streaming completes — **not** from the `conversation-update` webhook.

**Incremental context:** If a prior `AnalystSnapshot` exists, the prompt includes only `NEW TRANSCRIPT` turns since `snapshot.after_turn`. This keeps the prompt size bounded regardless of call length. Also appends `EXISTING_PROBES` to prevent duplicating questions already in the queue.

**Output:** `AnalysisUpdate` — themes, contradictions, surprises, investor_signals (tagged `[PMF]`/`[COMPETITIVE]`/`[REVENUE]`/`[AI-SIGNAL]`/`[RED-FLAG]`), up to 3 new probes with priority (1=urgent, 2=worthwhile, 3=nice-to-have), covered_subtopics.

**Persistence:** Writes one `AnalystSnapshot` row + new `Probe` rows per run. Exceptions are caught by `run_analyst_safely` — a crashing analyst never affects the live call.

### 3. Synthesis — `voice_agent/agents/synthesis.py`

Post-call only. Gated by `ENABLE_SYNTHESIS_REPORT` flag (currently `False`).


| Property    | Value                                    |
| ----------- | ---------------------------------------- |
| Model       | `claude-sonnet-4-6`                      |
| Deadline    | None (post-call)                         |
| Output type | `ReportOutput`                           |
| Gate        | `ENABLE_SYNTHESIS_REPORT` in `config.py` |


Takes the full transcript (up to 200 turns) + latest `AnalystSnapshot` and produces a structured investment research report: summary, themes with quotes, contradictions, key quotes, follow-up questions, PMF score (1–5), competitive/revenue/AI-adoption signals, red flags, investment thesis bullets.

Triggered from `server.py` on `end-of-call-report` webhook event. The Tier 3 eval always runs it regardless of the flag.

---

## Turn Pipeline — `voice_agent/turn.py`

Production uses `TurnPipeline` from `/vapi/llm/chat/completions` (streaming SSE). The local REPL and evals call `run_speech_turn()`, which wraps the same pipeline and awaits `commit()` — same persistence path.

```
Phase 1a  Turn number for this utterance
          next_turn_number(session, call_id) → COUNT(existing turns for call) + 1
          → short session, then closed before LLM work

Phase 1b  Parallel context reads (no DB write lock held)
          5 anyio tasks: next_q, remaining, probes, snapshot, messages
          → PreparedInterviewerTurn built with prompt_parts

Phase 1c  LLM call (no open DB session during this phase)
          anyio.move_on_after(5.0 s):
            run_interviewer() → InterviewerOutput
          on timeout: _fallback() → next scripted or wrap_up

Phase 2   commit() — DB writes (brief session)
          respondent Turn + interviewer Turn rows (when Vapi messages present)
          scripted/skip_scripted → mark_scripted_asked() (cursor += 1)
          probe_id_used → mark_probe_asked() + log analyst_lag_turns
          should_run_analyst(session, call_id) for downstream scheduling in server
```

**Turn rows are written in `commit()`** as soon as the LLM stream finishes — before TTS necessarily completes and before the user necessarily hears the full reply. On barge-in, the DB can still hold the full generated text even if playback stops early; a future reconciliation pass may trim analyst-visible text (see project hardening plan).

**Streaming path** (`TurnPipeline.stream_tokens()`) yields `str` tokens as they arrive from the LLM. After the generator is fully consumed, `commit()` reads `stream.output` and applies DB side effects (including `Turn` inserts).

**Filler + `<flush />` tag:** When the first LLM token hasn't arrived within `FILLER_THRESHOLD_S`, `stream_tokens()` yields `filler + " <flush /> "` before the real tokens. The `<flush />` is a Vapi-specific directive: it tells Vapi to dispatch the accumulated text to TTS *immediately*, without buffering for more tokens. Without it, TTS providers (especially ElevenLabs) hold a small chunk until they have enough context to synthesize natural-sounding audio, defeating the purpose of sending it early. Note: ElevenLabs may still buffer internally despite the flush — see `docs/Latency Measurement.md` for the nuance.

**Streaming with structured output — one LLM call, two-phase read:** `InterviewerOutput` is a Pydantic model with both `utterance` (the voice response text) and decision fields (`action`, `reasoning`, `probe_id_used`). Structured JSON output can't be streamed token-by-token — it isn't valid until complete. PydanticAI's streaming API solves this: `InterviewerStream.tokens()` yields the `utterance` field's tokens as they arrive (for immediate TTS delivery) while PydanticAI assembles and validates the full output internally. The validated `InterviewerOutput` is only available via `stream.output` after the stream is exhausted. Result: Vapi/TTS gets tokens at LLM speed (low latency), and routing decisions (action dispatch, probe DB writes) happen in `commit()` after streaming ends — all within a single LLM call, no two-pass approach needed.

---

## Vapi Integration — `voice_agent/server.py`

Vapi treats this server as an OpenAI-compatible custom LLM provider. Two endpoints:

### `POST /vapi/llm/chat/completions`

Per-turn inference. Called by Vapi at each utterance boundary.

**Request** (OpenAI-shaped from Vapi):

```json
{
  "stream": true,
  "call": {"id": "<vapi_call_id>", "assistant": {"metadata": {"call_id": "<our_id>"}}},
  "messages": [{"role": "user", "content": "..."}]
}
```

Our `call_id` is recovered from `call.assistant.metadata.call_id` — this is how we correlate Vapi's call to our DB row. Vapi messages are passed through to the interviewer as conversation history (source of truth for what was spoken).

**Streaming (SSE):** Returns `StreamingResponse` immediately. The generator yields OpenAI chunk format for each `str` token, then a final `finish_reason: "stop"` chunk + `data: [DONE]`. On `asyncio.CancelledError` (Vapi barge-in / interruption), logs and returns without `[DONE]`.

### `POST /vapi/webhook`

Lifecycle events only, no LLM calls.


| Event                         | Action                                                                                           |
| ----------------------------- | ------------------------------------------------------------------------------------------------ |
| `status-update` (in-progress) | `Call.status` → `"active"`                                                                       |
| `conversation-update`         | Debug log only (`messages` available for future reconciliation; **no** transcript writes here)    |
| `end-of-call-report`          | `Call.status` → `"ended"`; schedule synthesis if enabled                                         |
| `speech-update`               | Latency / extended-silence instrumentation (`_speech_ts`), not turn persistence                  |

**Turn persistence and analyst scheduling** happen in `POST /vapi/llm/chat/completions`: after `TurnPipeline.commit()` inserts rows, the handler may fire `run_analyst_safely` when `should_run_analyst()` returned true during that commit.

### Call Lifecycle Endpoints


| Endpoint                 | Purpose                                                       |
| ------------------------ | ------------------------------------------------------------- |
| `POST /calls/start`      | Creates `Call` row; optionally dials via Vapi (`_dial_vapi`)  |
| `GET /calls/{id}/report` | Returns synthesis report (or status stub if disabled/pending) |
| `GET /calls/{id}/trace`  | Returns Logfire trace URL for the call                        |
| `DELETE /calls/{id}`     | Ends call locally + cancels in-flight Vapi call               |


**Outbound dialing (`_dial_vapi`):** POSTs to `https://api.vapi.ai/call/phone`. Key fields:

- `assistant.model`: `{provider: "custom-llm", url: "{WEBHOOK_URL}/vapi/llm"}`
- `assistant.serverUrl`: `{WEBHOOK_URL}/vapi/webhook`  
- `assistant.metadata`: `{"call_id": call_id}` — the recovery key used by both endpoints
- `assistant.transcriber`: Deepgram nova-2
- `assistant.voice`: ElevenLabs by default (voice ID `21m00Tcm4TlvDq8ikWAM`), or Vapi's built-in Elliot if `VAPI_VOICE_PROVIDER=vapi`

**Known race condition:** `vapi_call_id` is written to DB after `POST /call/phone` returns, but Vapi can fire `status-update: ringing` before that write commits. Works in practice (ringing fires hundreds of ms later) but is not atomic. Tracked in `docs/TODO.md`.

---

## Data Model — `voice_agent/state.py`

All SQLModel tables. Schema is SQLite in dev but Postgres-compatible.

```
calls
├── id (PK, our UUID)
├── vapi_call_id (Vapi's ID after dial; unique when set — multiple NULLs allowed pre-dial)
├── phone_number
├── scripted_questions (JSON list)
├── scripted_cursor (int, advances on scripted/skip_scripted)
├── status (pending | active | ended)
├── end_reason, started_at, ended_at
└── → turns, probes, analyst_snapshots, synthesis_report (cascade delete)

turns
├── call_id (FK)
├── turn_number (unique per call — enforced with DB UniqueConstraint)
├── speaker (interviewer | respondent)
├── text
├── action, reasoning (interviewer turns only)
└── latency_ms

probes
├── call_id (FK)
├── question, priority (1=urgent, 2=worthwhile, 3=nice-to-have)
├── rationale, generated_after_turn
└── asked (bool), asked_at

analyst_snapshots
├── call_id (FK)
├── after_turn, after_scripted_cursor
├── themes, contradictions, surprises, investor_signals, covered_subtopics (JSON lists)
└── latency_ms

synthesis_reports
├── call_id (unique FK)
├── summary, pmf_score (1-5), pmf_score_rationale
└── themes, contradictions, key_quotes, follow_up_questions,
    competitive_signals, revenue_signals, ai_adoption_signals,
    red_flags, investment_thesis_bullets (JSON lists)
```

**Key DB helpers** (all in `state.py`):

- `next_turn_number()` — `COUNT(turns for call) + 1` for the next utterance index (short read transaction at turn start).
- `next_scripted()` / `scripted_remaining()` — cursor-based iteration over `scripted_questions` JSON array.
- `top_probes(n=3)` — `WHERE asked=False ORDER BY priority ASC, created_at ASC LIMIT n`.
- `should_run_analyst()` — scripted cursor advanced since last snapshot OR ≥ `ANALYST_TURN_INTERVAL` (10) interviewer turns since `after_turn`.
- `turns_since(after_turn)` — incremental analyst context reads.
- `session_scope(engine)` — context manager: commit on success, rollback on exception.
- `make_engine(url)` — `:memory:` → `StaticPool` (tests); file SQLite → `NullPool`; Postgres → default pool.

---

## Observability — `voice_agent/tracing.py`

Pydantic Logfire (OpenTelemetry-based). `init_tracing()` is idempotent and called from every entrypoint.

**Auto-instrumented:** PydanticAI (LLM calls + token counts), HTTPX (outbound Vapi API calls), FastAPI (request spans), SQLAlchemy (query spans).

**Key custom spans/events:**


| Span/Event                     | Where            | Key attributes                                                   |
| ------------------------------ | ---------------- | ---------------------------------------------------------------- |
| `turn_started`                 | turn.py          | call_id, turn_number                                             |
| `interviewer_run`              | turn.py          | action, utterance, reasoning, latency_ms, fallback, probe_source |
| `interviewer_first_token`      | turn.py (stream) | ttft_ms                                                          |
| `turn_persist`                 | turn.py          | action, probe_id_used                                            |
| `probe_used`                   | turn.py          | probe_id, priority, analyst_lag_turns                            |
| `vapi_event`                   | server.py        | event type, call_id                                              |
| `vapi_llm_request`             | server.py        | respondent_chars, stream, action, elapsed_ms                     |
| `vapi_llm_stream_cancelled`    | server.py        | call_id (barge-in)                                               |
| `analyst_run`                  | server.py        | call_id, after_turn, latency_ms                                  |
| `synthesis_run`                | server.py        | call_id                                                          |
| `call_created/activated/ended` | server.py        | call_id                                                          |


All spans carry `call_id` — Logfire can filter an entire call's trace by that attribute. Trace URL: `GET /calls/{id}/trace`.

When `LOGFIRE_TOKEN` is unset, output is console-only. Evals always pass `send_to_logfire=False`.

---

## Eval Suite — `evals/`

Three tiers using `pydantic_evals`. All run with `uv run pytest`.

### Tier 1 — Single-turn interviewer decisions (`test_interviewer.py`)

18 labeled cases in `datasets/interviewer_turns.yaml`. Each case specifies prior turns, pending probes, and expected action. Seeds a fresh in-memory SQLite per case.

Thresholds: ActionMatches ≥90% | SingleQuestion 100% | warmth ≥4.0/5 | non-leading ≥90%

### Tier 2 — Analyst probe quality (`test_analyst.py`)

6 transcript cases in `datasets/analyst_probes.yaml` (contradiction-heavy, specificity-focused, clean).

Thresholds: HasProbes 100% | NoDuplicateProbes 100% | probes_specific ≥4.0/5 | non_leading ≥90% | priority_calibrated ≥90%

### Tier 3 — Full conversation trajectories (`test_trajectories.py`, `@pytest.mark.slow`)

5 respondent personas in `datasets/personas.yaml`, driven by `simulator.py` (Sonnet 4.6 respondent simulator). Full loop: simulate turn → run analyst (awaited) → run interviewer → repeat up to 20 turns → run synthesis.

Thresholds: CallCompletes 100% | CoveredAllScripted ≥75% | CaughtContradiction 100% (contradictory persona) | report_quality ≥3.0/5

### Eval datasets


| File                      | Contents                                                                      |
| ------------------------- | ----------------------------------------------------------------------------- |
| `interviewer_turns.yaml`  | 18 single-turn cases (probe×5, scripted×5, clarify×3, wrap_up×3, off_topic×2) |
| `analyst_probes.yaml`     | 6 transcript cases for analyst quality                                        |
| `data/investor_questions.yaml` | 10-question canonical scripted arc with `[product]` token (not an eval fixture — lives in `data/`) |
| `personas.yaml`           | 6 respondent personas for trajectory evals                                    |


LLM judge model: `claude-opus-4-6` (defined in `evals/evaluators.py`).

---

## Local Development

**REPL mode** (`scripts/play.py`): Drives a full call in-process — no server, no Vapi. Loads questions from `data/investor_questions.yaml`, maintains message history, calls `run_speech_turn()` directly; `TurnPipeline.commit()` writes `Turn` rows to the DB like production. Shares the same `voice_agent.db` as the server.

**Phone mode** (`scripts/play.py --phone +1...`): Requires the server running. POSTs to `/calls/start`, then Vapi handles everything.

**Environment variables** (see `.env.example`):


| Var                   | Required        | Purpose                                           |
| --------------------- | --------------- | ------------------------------------------------- |
| `ANTHROPIC_API_KEY`   | Yes             | All LLM calls                                     |
| `VAPI_API_KEY`        | For phone calls | Vapi outbound dialing                             |
| `WEBHOOK_URL`         | For phone calls | Public URL Vapi POSTs to (ngrok in dev)           |
| `DATABASE_URL`        | No              | Defaults to `sqlite:///voice_agent.db`            |
| `LOGFIRE_TOKEN`       | No              | Enables Logfire cloud export                      |
| `LOGFIRE_PROJECT`     | No              | Defaults to `"voice-agent"`                       |
| `VAPI_VOICE_PROVIDER` | No              | `"vapi"` for Elliot voice; defaults to ElevenLabs |


---

