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
uv run python scripts/play.py "Notion AI"             # same, substitutes [product] token in data/investor_questions.yaml
uv run python scripts/play.py "Notion AI" --phone +14155551234  # real outbound Vapi call (needs server running)

uv run pytest                                         # all tests + evals
uv run pytest tests/ -v                               # unit tests only (DB, tracing)
uv run pytest evals/ -v                               # eval tiers (hit Anthropic API)
uv run pytest evals/test_interviewer.py::test_tier1_interviewer_decisions -v  # single eval

uv run pytest -m replay                              # fast deterministic replay evals (CI-safe)
uv run pytest -m slow                                # slow live-simulation evals

uv run python -m voice_agent.agents.analyst          # analyst smoke test against a canned transcript
uv run python -m voice_agent.agents.interviewer      # interviewer REPL with a seeded in-memory DB

uv run alembic upgrade head                          # apply DB migrations (Postgres / file SQLite)
uv run alembic revision --autogenerate -m "..."      # generate a migration from model changes

cd frontend && npm install && npm run dev            # Next.js operator UI on :3000 (expects backend on :8000)
```

Pytest config in `pyproject.toml` sets `asyncio_mode = "auto"` and `testpaths = ["tests", "evals"]` — async tests need no marker. Markers: `replay` (deterministic, CI-safe), `slow` (live simulation).

## Architecture

Multi-agent voice research interviewer built on Vapi. **[docs/architecture.md](docs/architecture.md)** is the authoritative design doc; read it when making non-trivial changes. Read-over-skim if you touch `interviewer.py`, `analyst.py`, or the webhook flow. (`docs/PLAN.md` is the original design doc but has stale model names and some implementation details that differ from the actual code — treat `architecture.md` as ground truth.)

### The three agents and their latency contract

Three PydanticAI agents, each with a different latency budget — this is the core design constraint:

1. **Interviewer** ([voice_agent/agents/interviewer.py](voice_agent/agents/interviewer.py)) — one LLM call per turn, hard 5 s deadline via `anyio.move_on_after`. **Not** a ReAct tool loop: all DB state is pre-fetched into a CONTEXT block and the model returns a single structured `InterviewerOutput`. Python then applies side effects (marking probe/scripted as asked) based on the output. If the deadline fires, `_fallback()` returns the next scripted question. Uses a `FallbackModel` chain built in `_build_interviewer_model()`: **OpenAI gpt-4.1-mini (optional) → Haiku 4.5 → Gemini 2.0 Flash → Groq (llama-3.3-70b) → Cerebras llama3.1-8b (optional)**. Models are env-overridable via `OPENAI_MODEL`, `HAIKU_MODEL`, `GEMINI_MODEL`, `GROQ_MODEL`, `CEREBRAS_MODEL` in [voice_agent/config.py](voice_agent/config.py). The OpenAI tier is skipped unless `OPENAI_API_KEY` + `OPENAI_MODEL` are both set; Cerebras is skipped unless `CEREBRAS_API_KEY` + `CEREBRAS_MODEL` are both set (set either model to `""` to drop that tier).
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
  - `end-of-call-report` — flips status to ended; schedules synthesis if enabled.
  - `speech-update` — tracks timing (`_speech_ts`) for latency measurement (`assistant/started` stashes TTFT, `assistant/stopped` computes TTS duration).

Outbound dialing: `POST /calls/start` with `phone_number` + `VAPI_API_KEY` and full dial config returns **202** and sets `Call.dial_status` to `queued`; `_dial_vapi` runs in a background task (`queued`→`dialing`→`dialed` + atomic `UPDATE` of `vapi_call_id`, or terminal `dial_failed` / `dial_skipped`). Poll **`GET /calls/{call_id}`** for `dial_status`, `vapi_call_id`, and `status`. Without a phone (or without a dial), response stays **200** and `dial_status` is null. See `docs/TODO.md` for remaining edge cases.

**Silence / dead-air:** Handled entirely by Vapi via `customer.speech.timeout` assistant hooks set in `_dial_vapi` (built by `_build_speech_timeout_hooks`). The ladder escalates on continuous user silence — re-prompt at `VAPI_SILENCE_TIMEOUT_SECONDS` ("Take your time.") and 2x ("Still there?"), then warmly end the call at 3x (`endCall` tool). `triggerResetMode: onUserSpeech` clears the ladder the moment the user speaks. Set `VAPI_SILENCE_TIMEOUT_SECONDS=0` to disable. Thinking fillers ("um", "let me think") are real transcribed speech and are handled by the interviewer model (NODE 1 → "Take your time."), not by these hooks.

**Outbound dial config** (`_dial_vapi`): POSTs to `https://api.vapi.ai/call`. Transcriber is **Deepgram `flux-general-en`** (not nova-2 — that's stale in older docs); voice defaults to ElevenLabs (`VAPI_VOICE_PROVIDER=11labs`) or Vapi's built-in Elliot. `assistant.metadata.call_id` is the recovery key correlating Vapi's call to our DB row, used by both the webhook and the custom-LLM endpoint.

### Multi-tenant auth ([voice_agent/auth.py](voice_agent/auth.py))

Google OAuth + JWT, org-scoped data. Flow: `GET /auth/google` → Google consent → `GET /auth/google/callback` exchanges the code, upserts a `User`, finds their `OrgMember`ship, issues a JWT, and sets it as an httpOnly `access_token` cookie before redirecting to the frontend. `GET /auth/me` decodes the JWT. `require_auth` is a FastAPI dependency returning an `AuthPrincipal` (`user_id`, `org_id`, `role`, `email`); `require_admin` gates admin-only routes.

**Dev mode:** `JWT_SECRET=""` makes `require_auth` return a fake admin principal in `DEFAULT_ORG_ID` so the app works with no auth configured. All data-scoping queries filter by `principal.org_id` — never query `Call`/`Project` without the org filter.

Org management routes (`/orgs/me`, `/orgs/me/members`) are admin-gated. `ADMIN_BOOTSTRAP_EMAIL` grants admin on first login.

### Projects

A `Project` ([state.py](voice_agent/state.py)) is an org-scoped research config: `product`, `product_description`, `focus_areas`, `deprioritize`, `investor_thesis`, and a default `scripted_questions` list. `/calls/start` resolves defaults from the project (when `project_id` is given) and applies per-call overrides on top. A call's `brain` column stores a serialized `CallBrain` ([models.py](voice_agent/models.py)) — the strategic brief injected into the interviewer + analyst contexts. CRUD lives in `server.py` (`/projects`, `/projects/{id}`, `/projects/{id}/calls`).

### Frontend ([frontend/](frontend/))

Next.js 16 + React 19 operator UI (projects, calls, reports, settings, login). **Read `frontend/AGENTS.md` before editing** — it pins a Next.js version with breaking changes vs. training data; consult `node_modules/next/dist/docs/` first. The API client is [frontend/src/lib/api.ts](frontend/src/lib/api.ts); auth uses the cookie set by the OAuth callback (`credentials: "include"`, redirect to `/login` on 401).

### Security

- **Webhook HMAC:** `/vapi/webhook` uses `_require_vapi_signature` (FastAPI dependency). Vapi signs the payload with `VAPI_WEBHOOK_SECRET` using HMAC-SHA256. Requests outside `VAPI_TIMESTAMP_TOLERANCE_S` (5 min) are rejected. `VAPI_WEBHOOK_SECRET=""` skips verification (dev mode).
- **LLM secret token:** `/vapi/llm/chat/completions` checks the `X-Vapi-Secret` header against `LLM_SECRET_TOKEN`. Empty = skip check (dev mode).
- **JWT auth:** REST API routes (`/projects`, `/calls`, `/orgs`) require a valid JWT (cookie or `Authorization: Bearer`) via `require_auth`. `JWT_SECRET=""` = dev superuser (see above).
- **API bearer token:** `_require_api_auth` enforces `API_AUTH_TOKEN` as a static bearer where applied. Empty = skip (dev).
- **Rate limiting:** `POST /calls/start` is rate-limited per client IP via slowapi (`CALLS_START_RATE_LIMIT`, default `20/minute`; `""` disables).

### Database (SQLModel — SQLite dev / Postgres prod)

Dev uses a single SQLite file `voice_agent.db`; **production runs Postgres** via `DATABASE_URL` (bare `postgres://` URLs are normalized to `postgresql+psycopg2://` in `make_engine`). **Schema is managed by Alembic** ([alembic/versions/](alembic/versions/)) — `init_db()` only runs `create_all()` for in-memory SQLite (tests/evals); for file/Postgres DBs run `uv run alembic upgrade head` (the Docker image does this on boot). Generate migrations with `uv run alembic revision --autogenerate -m "..."`.

Tables: `organizations`, `users`, `org_members`, `projects`, `calls`, `turns`, `probes`, `analyst_snapshots`, `synthesis_reports`. Key reads used by the interviewer live alongside the tables in [voice_agent/state.py](voice_agent/state.py): `next_scripted`, `top_probes` (priority asc, then age), `recent_turns`, `latest_snapshot`, `turns_since`. The analyst uses `turns_since(snapshot.after_turn)` to stay context-bounded. `scripted_cursor` on `Call` advances on both `scripted` and `skip_scripted` actions — the model can skip a scripted question that was already answered organically.

### Tracing

`init_tracing()` in [voice_agent/tracing.py](voice_agent/tracing.py) is idempotent and called from every entrypoint (server lifespan, play.py, evals). When `LOGFIRE_TOKEN` is unset it falls back to console-only output — evals explicitly pass `send_to_logfire=False`. Every meaningful span carries `call_id` + `turn_number` via `agent_span` so Logfire can slice by call or turn.

### Evals

Under `evals/`, all using `pydantic_evals` (see [docs/architecture.md](docs/architecture.md) → Eval Suite for thresholds):

- **Tier 1** ([evals/test_interviewer.py](evals/test_interviewer.py)) — single-turn decision eval, seeds in-memory SQLite per case from `interviewer_turns.yaml`. Thresholds: ActionMatches ≥90%, SingleQuestion 100%, warmth ≥4/5, non-leading ≥90%.
- **Tier 2** ([evals/test_analyst.py](evals/test_analyst.py)) — analyst probe quality against canned transcripts in `analyst_probes.yaml`.
- **Tier 3a — replay** ([evals/test_replay.py](evals/test_replay.py), `@pytest.mark.replay`) — canned full transcripts; respondent lines fixed, interviewer runs live. Deterministic + CI-safe. Tests multi-turn state (cursor advancement, probe staleness, loop guards).
- **Tier 3b — simulation** ([evals/test_trajectories.py](evals/test_trajectories.py), `@pytest.mark.slow`) — full conversation driven by `evals/simulator.py` respondent personas (`personas.yaml`) against the real interviewer + analyst; bypasses `ENABLE_SYNTHESIS_REPORT` gate.
- **Synthesis** ([evals/test_synthesis.py](evals/test_synthesis.py)) — post-call report quality against the replay transcripts.

`evals/cases.py` loads YAML into typed `Case` inputs; `evals/evaluators.py` holds custom scorers and `LLMJudge` configs. In-memory SQLite needs `poolclass=StaticPool` — without it each session gets a fresh empty DB.

### Scripted question source

`scripts/play.py` loads scripted questions from `data/investor_questions.yaml` and substitutes `[product]` with the CLI arg. This is the canonical question list; the server's `/calls/start` takes arbitrary `scripted_questions` from the caller.