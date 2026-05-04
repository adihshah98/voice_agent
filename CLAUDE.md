# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Documentation

Always use Context7 MCP when I need library/API documentation, code generation, setup or configuration steps without me having to explicitly ask.

## Commands

Package manager is `uv`; env vars load from `.env` (see `.env.example`). Minimum is `ANTHROPIC_API_KEY`.

```bash
uv sync                                               # install deps
uv run uvicorn voice_agent.server:app --reload        # run FastAPI server (webhooks + custom-LLM endpoint)
uv run python scripts/play.py                         # local terminal REPL — drives a call in-process, no server needed
uv run python scripts/play.py "Notion AI"             # same, substitutes [product] token in investor_questions.yaml
uv run python scripts/play.py "Notion AI" --phone +14155551234  # real outbound Vapi call (needs server running)

uv run pytest                                         # all tests + evals
uv run pytest tests/ -v                               # unit tests only (DB, tracing)
uv run pytest evals/ -v                               # eval tiers (hit Anthropic API)
uv run pytest evals/test_interviewer.py::test_tier1_interviewer_decisions -v  # single eval

uv run python -m voice_agent.agents.analyst          # analyst smoke test against a canned transcript
uv run python -m voice_agent.agents.interviewer      # interviewer REPL with a seeded in-memory DB
```

Pytest config in `pyproject.toml` sets `asyncio_mode = "auto"` and `testpaths = ["tests", "evals"]` — async tests need no marker.

## Architecture

Multi-agent voice research interviewer built on Vapi. **[docs/architecture.md](docs/architecture.md)** is the authoritative design doc; read it when making non-trivial changes. Read-over-skim if you touch `interviewer.py`, `analyst.py`, or the webhook flow. (`docs/PLAN.md` is the original design doc but has stale model names and some implementation details that differ from the actual code — treat `architecture.md` as ground truth.)

### The three agents and their latency contract

Three PydanticAI agents, each with a different latency budget — this is the core design constraint:

1. **Interviewer** ([voice_agent/agents/interviewer.py](voice_agent/agents/interviewer.py)) — one LLM call per turn, hard 5 s deadline via `anyio.move_on_after`. **Not** a ReAct tool loop: all DB state is pre-fetched into a CONTEXT block and the model returns a single structured `InterviewerOutput`. Python then applies side effects (marking probe/scripted as asked) based on the output. If the deadline fires, `_fallback()` returns the next scripted question. Uses a `FallbackModel` chain: **Haiku 4.5 → Groq (llama-3.3-70b) → Groq retry → Cerebras (llama3.1-8b)**. Models are env-overridable via `HAIKU_MODEL`, `GROQ_MODEL`, `CEREBRAS_MODEL` in [voice_agent/config.py](voice_agent/config.py). Set `CEREBRAS_MODEL=""` to skip Cerebras.
2. **Analyst** ([voice_agent/agents/analyst.py](voice_agent/agents/analyst.py)) — Sonnet 4.6, fire-and-forget via `asyncio.create_task`. Triggered after [`TurnPipeline.commit()`](voice_agent/turn.py) from the `/vapi/llm/chat/completions` handler ([`voice_agent/server.py`](voice_agent/server.py)) when `should_run_analyst()` returned true during that commit: either the scripted cursor advanced since the last snapshot, or ≥`ANALYST_TURN_INTERVAL` (10) interviewer turns since `after_turn`. Reads prior `AnalystSnapshot` as established context + only the new turns since, so prompt size stays bounded on long calls. Wrapped in `run_analyst_safely` — exceptions are swallowed so a crashing analyst never affects the live call.
3. **Synthesis** ([voice_agent/agents/synthesis.py](voice_agent/agents/synthesis.py)) — Sonnet 4.6, post-call. Gated by `ENABLE_SYNTHESIS_REPORT` in [voice_agent/config.py](voice_agent/config.py) (currently **False** — keep that in mind when testing the end-of-call path).

**Invariant: agents never call each other.** All coordination flows through SQLite tables in [voice_agent/state.py](voice_agent/state.py). Model IDs, timeouts, and the synthesis flag all live in [voice_agent/config.py](voice_agent/config.py) — swap a model there, nothing else changes.

**REPL path caveat:** `run_interviewer` and `run_interviewer_with_timeout` in `interviewer.py` are called only by evals. In the production hot path (`TurnPipeline` in `turn.py`), the SQLModel session is closed *before* `interviewer.run()` is awaited, so `deps.session` is `None` during the actual LLM call. The REPL and evals pass a live session. This path is diverging from prod — be careful when testing REPL behavior as a proxy for production.

### Turn pipeline

[voice_agent/turn.py](voice_agent/turn.py) has two paths:

- **`TurnPipeline`** — production streaming path. The Vapi custom-LLM webhook (`/vapi/llm/chat/completions`) instantiates this and streams SSE tokens. Tracks `llm_ttft_ms`, `filler_injected`, and prompt cache token counts per turn.
- **`run_speech_turn`** — non-streaming, used only by the local REPL (`scripts/play.py`) and evals. Buffers tokens and returns a complete result.

The custom-LLM endpoint returns OpenAI-shaped responses (SSE chunks) because Vapi expects an OpenAI-compatible shape.

**Filler phrases:** When `FILLER_THRESHOLD_S` > 0, `TurnPipeline` yields a short acknowledgment (e.g. "Mm-hm,") followed by `" <flush /> "` if the first LLM token hasn't arrived within that window. The `<flush />` tag tells Vapi to dispatch the buffered text to TTS immediately — without it, TTS providers hold the filler until more tokens arrive. `FILLER_THRESHOLD_S = 0.0` disables this. The filler is flagged via `filler_injected`.

**Streaming + structured output in one LLM call:** `InterviewerOutput` is a Pydantic model; its JSON can't be validated until complete. PydanticAI's streaming API lets `InterviewerStream.tokens()` yield `utterance` tokens live (for TTS) while building the full output internally. `stream.output` (action, reasoning, probe_id_used) is only readable after the stream is exhausted — consumed in `commit()`. No two-pass LLM call needed. See `docs/architecture.md` → Turn Pipeline for details.

**Turn rows** are inserted in [`TurnPipeline.commit()`](voice_agent/turn.py): respondent + interviewer rows after each completed LLM stream (`run_speech_turn()` uses the same pipeline for REPL/evals). Production triggers this from `/vapi/llm/chat/completions` after SSE finishes. The `conversation-update` webhook does **not** write turns — it is logged only (see [`docs/architecture.md`](docs/architecture.md)).

### Vapi lifecycle

Two endpoints, different roles:

- `/vapi/llm/chat/completions` — per-turn inference (replaces OpenAI from Vapi's perspective).
- `/vapi/webhook` — HMAC-verified (see Security below). Four event types:
  - `status-update` — flips `Call.status` pending→active.
  - `conversation-update` — debug log only (no DB writes; transcript reconciliation may use `messages` later); turn persistence and analyst scheduling happen in the custom-LLM endpoint after `commit()`.
  - `end-of-call-report` — flips status to ended; schedules synthesis if enabled; cancels silence watch.
  - `speech-update` — tracks timing (`_speech_ts`) for latency measurement; manages the extended-silence watch (starts timer on `assistant/stopped`, cancels on `user/started`).

Outbound dialing (`_dial_vapi`) writes the Vapi-assigned `vapi_call_id` back onto the `Call` row after the POST returns. Known fragile: Vapi can fire `status-update: ringing` before that write commits. Works in practice (ringing fires ~hundreds of ms later) but is not atomic. See `docs/TODO.md`.

**Extended silence:** After assistant TTS ends, if the user doesn't speak within `VAPI_EXTENDED_SILENCE_SECONDS`, `_end_call_vapi_delete` fires. Set to 0 (default) to disable. Avoids unbounded silent calls up to `maxDurationSeconds`.

### Security

- **Webhook HMAC:** `/vapi/webhook` uses `_require_vapi_signature` (FastAPI dependency). Vapi signs the payload with `VAPI_WEBHOOK_SECRET` using HMAC-SHA256. Requests outside `VAPI_TIMESTAMP_TOLERANCE_S` (5 min) are rejected. `VAPI_WEBHOOK_SECRET=""` skips verification (dev mode).
- **LLM secret token:** `/vapi/llm/chat/completions` checks the `X-Vapi-Secret` header against `LLM_SECRET_TOKEN`. Empty = skip check (dev mode).

### Database (SQLModel/SQLite)

Single file `voice_agent.db`; schema is Postgres-compatible. Key reads used by the interviewer live alongside the tables in [voice_agent/state.py](voice_agent/state.py): `next_scripted`, `top_probes` (priority asc, then age), `recent_turns`, `latest_snapshot`, `turns_since`. The analyst uses `turns_since(snapshot.after_turn)` to stay context-bounded. `scripted_cursor` on `Call` advances on both `scripted` and `skip_scripted` actions — the model can skip a scripted question that was already answered organically.

### Tracing

`init_tracing()` in [voice_agent/tracing.py](voice_agent/tracing.py) is idempotent and called from every entrypoint (server lifespan, play.py, evals). When `LOGFIRE_TOKEN` is unset it falls back to console-only output — evals explicitly pass `send_to_logfire=False`. Every meaningful span carries `call_id` + `turn_number` via `agent_span` so Logfire can slice by call or turn.

### Evals

Three tiers under `evals/`, all using `pydantic_evals`:

- **Tier 1** ([evals/test_interviewer.py](evals/test_interviewer.py)) — single-turn decision eval, seeds in-memory SQLite per case from `interviewer_turns.yaml`. Thresholds: ActionMatches ≥90%, SingleQuestion 100%, warmth ≥4/5, non-leading ≥90%.
- **Tier 2** ([evals/test_analyst.py](evals/test_analyst.py)) — analyst probe quality against canned transcripts in `analyst_probes.yaml`.
- **Tier 3** ([evals/test_trajectories.py](evals/test_trajectories.py)) — full conversation driven by `evals/simulator.py` respondent personas (`personas.yaml`) against the real interviewer + analyst. Marked `@pytest.mark.slow`; bypasses `ENABLE_SYNTHESIS_REPORT` gate.

`evals/cases.py` loads YAML into typed `Case` inputs; `evals/evaluators.py` holds custom scorers and `LLMJudge` configs. In-memory SQLite needs `poolclass=StaticPool` — without it each session gets a fresh empty DB.

### Scripted question source

`scripts/play.py` loads scripted questions from `evals/datasets/investor_questions.yaml` and substitutes `[product]` with the CLI arg. This is the canonical question list; the server's `/calls/start` takes arbitrary `scripted_questions` from the caller.