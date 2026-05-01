---
name: Voice agent hardening
overview: Sequenced fix plan for the Vapi race conditions and best-practice gaps surfaced in the audit. Ordered smallest-blast-radius first (docs, schema, auth) then idempotency, dial lifecycle, partial-utterance reconciliation, and finally state/scale hygiene + polish. Stops short of moving turn writes wholesale into `conversation-update` (per "balanced" depth) — but does fix the doc lie and shift analyst triggering so DB-vs-spoken divergence stops poisoning the analyst.
todos:
  - id: p1_docs
    content: "Phase 1a: rewrite docs/architecture.md and CLAUDE.md to match actual turn-write + analyst-trigger flow"
    status: pending
  - id: p1_schema
    content: "Phase 1b: add UNIQUE(call_id,turn_number) on Turn and unique=True on Call.vapi_call_id"
    status: pending
  - id: p2_auth
    content: "Phase 2: API_AUTH_TOKEN bearer auth on /calls/start and DELETE /calls/{id}; better signature-mismatch logging"
    status: pending
  - id: p3_idempotency
    content: "Phase 3: convert webhook status flips to conditional UPDATEs; only fire synthesis/warmup on rowcount==1"
    status: pending
  - id: p4_dial
    content: "Phase 4: dial_status (or status=dialing|dial_failed); single-transaction vapi_call_id write; fire-and-forget dial in /calls/start"
    status: pending
  - id: p5_analyst
    content: "Phase 5: move analyst trigger to conversation-update; reconcile Turn.text against Vapi messagesOpenAIFormatted on barge-in"
    status: pending
  - id: p6_state
    content: "Phase 6: CallStateStore abstraction over _speech_ts/_silence_watch_tasks/_last_filler with periodic janitor"
    status: pending
  - id: p7_polish
    content: "Phase 7: sorted covered_subtopics + probes for cache stability; fix interviewer fallback-chain doc; decide on _warmup_groq"
    status: pending
isProject: false
---

# Voice agent hardening

## Phase 1 — Doc/code reconciliation + schema constraints (low risk)

**Problem:** [`docs/architecture.md`](docs/architecture.md) and [`CLAUDE.md`](CLAUDE.md) describe `_write_confirmed_turns()`, `_write_turns_local()`, a `turn_count` column, and atomic `UPDATE … RETURNING` — none of which exist. AI agents (and humans) reading these docs will produce broken edits. Also, no `UNIQUE` constraints backstop the count-based turn numbering.

**Fixes:**
- Rewrite the "Turn pipeline" + "Vapi Integration" + "Data Model" sections of [`docs/architecture.md`](docs/architecture.md) to match reality: turns written inside [`TurnPipeline.commit()`](voice_agent/turn.py) at LLM-completion time; `next_turn_number = COUNT(*)+1`; analyst fired from inside `vapi_llm`, not from `conversation-update`.
- Update [`CLAUDE.md`](CLAUDE.md) "Turn pipeline" + "Vapi lifecycle" sections the same way.
- Add `UniqueConstraint("call_id", "turn_number")` to [`Turn` in voice_agent/state.py](voice_agent/state.py) and `unique=True` on `Call.vapi_call_id`. Fail loudly on duplicates rather than silently corrupting the transcript.
- Note (don't yet implement) Alembic migration for prod; for now SQLite dev DB regenerates cleanly.

## Phase 2 — API auth + webhook signature debuggability

**Problem:** `/calls/start` and `DELETE /calls/{id}` have no auth — anyone with `WEBHOOK_URL` can drain API keys. Webhook signature failures log only `vapi_signature_invalid` with no diagnostic to debug Vapi-dashboard misconfig.

**Fixes:**
- Add `API_AUTH_TOKEN` to [`Settings`](voice_agent/config.py); require it via `Authorization: Bearer …` on `/calls/start` and `DELETE /calls/{id}` ([`voice_agent/server.py`](voice_agent/server.py)). No-op when unset (dev mode), same pattern as `LLM_SECRET_TOKEN`.
- In [`_require_vapi_signature`](voice_agent/server.py), on mismatch log `expected_prefix=expected[:12]`, `received_prefix=sig_hex[:12]`, `payload_len=len(payload)`, and whether timestamp header was present. Never log secrets.

## Phase 3 — Webhook idempotency (row-conditional UPDATEs)

**Problem:** `end-of-call-report` and `status-update` do read-modify-write under default SQLite isolation. Two concurrent retries can both pass the status guard and double-fire `_synthesis_task` / `_warmup_groq`.

**Fixes in [`voice_agent/server.py`](voice_agent/server.py):**
- Replace status flips with conditional updates and only fire side-effects when the rowcount confirms we won the race:
  ```python
  result = session.execute(
      update(state.Call)
      .where(state.Call.id == call_id, state.Call.status == "pending")
      .values(status="active")
  )
  if result.rowcount == 1:
      _fire(_warmup_groq(), name="groq-warmup")
  ```
- Same pattern for `pending|active → ended` in `end-of-call-report` and `delete_call`. Synthesis and the final `call_ended` log only fire on `rowcount == 1`.
- Cleanup of `_speech_ts` and silence-watch is safe to run on both branches.

## Phase 4 — Dial lifecycle: atomic + non-blocking

**Problem:** `_dial_vapi` writes `vapi_call_id` in a separate transaction after the POST returns; if it fails or the POST raises, the `Call` row stays `pending` forever and Vapi may have an orphan call. `/calls/start` blocks the caller for up to 10s.

**Fixes:**
- Add `dial_status` column on `Call`: `pending|dialing|dialed|dial_failed`. Or repurpose `status`: `pending → dialing → active|dial_failed`.
- In [`/calls/start`](voice_agent/server.py): create row with `status="dialing"`, then `_fire(_dial_vapi(...))` — return `202 {call_id}` immediately. Caller polls `GET /calls/{id}` for state.
- In `_dial_vapi`: wrap in try/except; on success, single transaction sets `vapi_call_id` + leaves status `dialing` (the Vapi `status-update: in-progress` flips it to `active` via Phase 3's conditional update). On failure, single transaction sets `status="dial_failed"`, `end_reason=…`. Logfire-error.
- Update `/calls/{id}/report` and the `delete_call` paths to handle `dial_failed`.

## Phase 5 — Partial-utterance reconciliation (analyst-side only)

**Problem:** `commit()` writes the full LLM utterance to DB before Vapi confirms it was spoken. On barge-in, DB diverges from what the respondent actually heard. The analyst then sees ghost text. We're explicitly *not* moving turn writes into `conversation-update` (out of scope for "balanced") — but we can stop the analyst from being fed lies.

**Fixes:**
- Move analyst triggering out of [`vapi_llm` in voice_agent/server.py](voice_agent/server.py) (where it fires *during* the LLM stream, before TTS even starts) into the `conversation-update` handler. Use the existing [`should_run_analyst()`](voice_agent/state.py) gate; it already de-dupes via `scripted_cursor` + `latest_snapshot`.
- In `conversation-update`, take Vapi's `messagesOpenAIFormatted` and, for the most recent assistant message, if the persisted [`Turn.text`](voice_agent/state.py) is longer than what Vapi reports as spoken, overwrite it (and add a `truncated=True` flag — new column on `Turn`). The next analyst pass then sees what the respondent actually heard.
- This is additive: the LLM hot path doesn't change; only what the analyst consumes.

## Phase 6 — Per-call in-process state hygiene

**Problem:** [`_speech_ts`](voice_agent/server.py), [`_silence_watch_tasks`](voice_agent/server.py), and [`_last_filler`](voice_agent/turn.py) are module dicts. They leak when end-of-call events drop, and they break any future multi-worker deploy.

**Fixes (Redis-ready, but in-memory for now):**
- Introduce a thin `CallStateStore` protocol in a new `voice_agent/runtime_state.py` with `get/set/pop/cancel_task` methods. Default impl wraps a dict + asyncio task registry; production swaps to Redis without touching call sites.
- Refactor `server.py` and `turn.py` to use the store. `_last_filler` gets a `pop_call(call_id)` cleanup hook fired from end-of-call paths.
- Add a periodic janitor task (already running silence-watch infra is similar) that evicts entries older than `maxDurationSeconds + grace`. Cheap insurance against dropped end-of-call.

## Phase 7 — Polish

- Sort `covered_subtopics` (and `pending_probes` by `(priority, id)`) before formatting in [`_build_prompt_parts_from_reads`](voice_agent/agents/interviewer.py) — keeps Anthropic prompt cache stable across turns when the analyst returns the same set in a different order.
- Fix the chain-order doc lie in [`_build_interviewer_model`](voice_agent/agents/interviewer.py): docstring says `Cerebras → Groq → Groq → Haiku`, code does `Haiku → Groq → Groq → Cerebras`. Either correct the docstring (fast) or actually flip the order (cheaper p50 TTFT, more behavior change — defer).
- Decide what to do with [`_warmup_groq`](voice_agent/server.py): with Haiku as primary, Groq is only hit on Haiku failure. Either drop the warmup, or move it to the first observed Haiku error so it warms exactly when needed.
- Add `WHERE (call_id, turn_number)` index check now that the `UNIQUE` constraint from Phase 1 exists — SQLAlchemy/SQLModel does this automatically with `UniqueConstraint`, just verify.

---

## Out of scope (explicit deferrals)

- **Full move of turn writes into `conversation-update`** — would change the turn-numbering hot path and the analyst-context contract. Phase 5 reconciliation buys most of the correctness without the rewrite.
- **Multi-tenant auth + Postgres + Alembic** — already in [`docs/TODO.md`](docs/TODO.md).
- **Backchannels / speculative LLM firing / custom EOT** — Vapi-platform-bounded; flagged in TODO as "Advanced Voice UX."
- **Cost / token-rate alerts** — TODO already tracks.