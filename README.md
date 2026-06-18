# Voice Research Interviewer

An AI-powered outbound voice agent that conducts structured investor / user-research
interviews over the phone. It dials a respondent, runs a scripted question arc while
adaptively probing on interesting signals, analyzes the conversation in the background,
and (optionally) produces a post-call research report.

Built on **[Vapi](https://vapi.ai)** for telephony and **[PydanticAI](https://ai.pydantic.dev)**
for agent orchestration, with a **FastAPI** backend, **Vite + React Router** frontend, and
**Pydantic Logfire** for observability.

### 🚀 Live demo

**Operator UI → [voice-agent-1-mz3u.onrender.com](https://voice-agent-1-mz3u.onrender.com/)**

Deployed on **Render** (Dockerized FastAPI API + Vite static frontend, see [`render.yaml`](render.yaml))
with a **Supabase** Postgres database, Google OAuth login, and **Logfire** tracing. Create a
project, configure the research brief, and dial out — the agent runs the interview live.

> **Authoritative design doc:** [docs/architecture.md](docs/architecture.md). Read it before
> making non-trivial changes. This README is the orientation / quick-start.

---

## What it does

```
  Operator (web UI)            Backend (FastAPI)             Respondent (phone)
  ────────────────             ─────────────────             ──────────────────
  create Project   ──────────► POST /projects
  start Call       ──────────► POST /calls/start  ──► Vapi ──► ☎ phone rings
                                                                │
                                       per turn:                ├─ speaks
                                  /vapi/llm/chat/completions ◄──┤
                                  (interviewer LLM, 5s budget)  │
                                       ──► TTS audio ───────────┤
                                                                │
                               background: analyst LLM          │
                               post-call: synthesis report ◄────┘ hangs up
```

Three LLM agents coordinate **only through the database** (they never call each other):

| Agent | Model | When | Role |
| ----- | ----- | ---- | ---- |
| **Interviewer** | Haiku 4.5 (+ fallback chain) | Every turn, **5 s budget** | Decides the next spoken line: scripted question, adaptive probe, clarify, redirect, or wrap-up |
| **Analyst** | Sonnet 4.6 | Background, fire-and-forget | Reads the transcript, surfaces themes/contradictions/signals, queues probes for the interviewer |
| **Synthesis** | Sonnet 4.6 | Post-call (currently disabled) | Produces a structured research report with a PMF score |

The core design constraint is the **interviewer latency budget** — see
[Architecture → latency contract](docs/architecture.md#the-three-agents).

---

## Services & integrations

The system stitches together several external services. Here is what each one does and why it is here:

### Vapi — telephony orchestration

[Vapi](https://vapi.ai) is the voice infrastructure layer. It handles:

- **Outbound dialing** — `POST /calls/start` calls `https://api.vapi.ai/call`, which rings the respondent's phone.
- **STT (speech-to-text)** — transcription is done by **Deepgram `flux-general-en`** (configured inside the Vapi dial payload; the backend never touches raw audio).
- **Turn management** — Vapi calls our `/vapi/llm/chat/completions` endpoint after each respondent utterance, expecting an OpenAI-compatible streaming response. We own the LLM; Vapi owns the call state.
- **Webhooks** — Vapi POSTs lifecycle events to `/vapi/webhook`: `status-update` (call connected), `conversation-update` (transcript snapshot), `end-of-call-report` (call ended), and `speech-update` (TTS start/stop timing for latency measurement).
- **Silence handling** — Vapi's `customer.speech.timeout` hook ladder is configured per-call: re-prompt at `VAPI_SILENCE_TIMEOUT_SECONDS`, escalate at 2×, hang up at 3×. `triggerResetMode: onUserSpeech` resets the ladder the moment the user speaks.

Required env vars: `VAPI_API_KEY`, `VAPI_PHONE_NUMBER_ID`, `WEBHOOK_URL` (public base URL, e.g. ngrok in dev), `VAPI_WEBHOOK_SECRET` (HMAC key), `VAPI_SERVER_CREDENTIAL_ID`.

> The outbound phone number is provisioned through **Twilio** and imported into Vapi. Vapi owns the call orchestration; Twilio supplies the number. The backend only interacts with Vapi — `VAPI_PHONE_NUMBER_ID` is the Vapi ID of the imported Twilio number.

### ElevenLabs — text-to-speech

ElevenLabs provides the voice for the AI interviewer. It is invoked **through Vapi** (not directly by the backend): Vapi receives our streaming text tokens and forwards them to ElevenLabs for TTS, then plays the resulting audio to the respondent.

Key knobs (all optional; see `.env.example`):

| Var | Effect |
| --- | ------ |
| `VAPI_VOICE_ID` | Which ElevenLabs voice to use |
| `VAPI_VOICE_MODEL` | `eleven_flash_v2_5` ≈ 75 ms latency; `eleven_multilingual_v2` = max quality |
| `VAPI_VOICE_CHUNK_MIN_CHARACTERS` | Streaming flush threshold — `1` means audio starts on short fillers immediately |
| `VAPI_VOICE_STABILITY` / `_SIMILARITY_BOOST` / `_STYLE` / `_SPEED` | Fine-tune expressiveness vs. steadiness |

Set `VAPI_VOICE_PROVIDER=vapi` to switch to Vapi's built-in "Elliot" voice with no ElevenLabs dependency.

### Deepgram — speech-to-text

Deepgram is configured as the transcription provider inside the Vapi dial payload (`transcriber: { provider: "deepgram", model: "flux-general-en" }`). The backend never calls Deepgram directly — transcripts arrive pre-processed via Vapi's `conversation-update` and custom-LLM messages.

### Anthropic — LLM backbone

All three agents run on Anthropic models via PydanticAI:

- **Interviewer** — primary model is **Haiku 4.5** with a 5 s hard deadline. Falls back through: OpenAI gpt-4.1-mini (optional) → Haiku 4.5 → Gemini 2.0 Flash → Groq llama-3.3-70b → Cerebras llama3.1-8b (optional). Every tier is env-overridable or droppable.
- **Analyst** — **Sonnet 4.6**, fire-and-forget background task.
- **Synthesis** — **Sonnet 4.6**, post-call (currently disabled via `ENABLE_SYNTHESIS_REPORT=False`).

Required: `ANTHROPIC_API_KEY`. Optional resilience keys: `GOOGLE_API_KEY`, `GROQ_API_KEY`, `OPENAI_API_KEY`, `CEREBRAS_API_KEY`.

### Google OAuth — authentication

The operator UI uses Google OAuth for login (`GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET`). After the OAuth callback, the backend issues a JWT (`JWT_SECRET`) stored in `localStorage` and sent as `Authorization: Bearer` on every API request. Omit these vars for dev — the backend returns a fake admin principal automatically.

### Logfire — observability

[Pydantic Logfire](https://logfire.pydantic.dev) is the tracing and observability layer. Every meaningful span carries `call_id` + `turn_number` so you can slice by call or turn. When `LOGFIRE_TOKEN` is unset the backend falls back to console-only structured output — no external dependency in dev. Evals always disable Logfire.

### Supabase / Postgres — production database

Dev uses a local SQLite file (`voice_agent.db`). Production uses a Postgres database — hosted on Supabase in the live deployment, but any Postgres URL works. Set `DATABASE_URL=postgresql+psycopg2://...` and run `uv run alembic upgrade head`. The Docker image does this automatically on boot.

### Render — hosting

The live demo runs on [Render](https://render.com): a Dockerized FastAPI web service for the backend and a static site for the Vite frontend. Configuration is in [`render.yaml`](render.yaml). The API service runs `alembic upgrade head` as a pre-deploy command; the frontend service serves `frontend/dist/` with a catch-all rewrite to `index.html`.

---

## Quick start

Prerequisites: **Python ≥ 3.11**, [`uv`](https://docs.astral.sh/uv/), and an `ANTHROPIC_API_KEY`.
Phone calls additionally require a Vapi account; the local REPL needs neither Vapi nor a server.

```bash
uv sync                                  # install backend deps
cp .env.example .env                     # then add ANTHROPIC_API_KEY (minimum)

# Local REPL — drive a full interview in your terminal (no server, no phone):
uv run python scripts/play.py "Notion AI"

# Run the API server (Vapi webhooks + custom-LLM endpoint + REST API):
uv run uvicorn voice_agent.server:app --reload

# Real outbound phone call (needs server running + Vapi env vars set):
uv run python scripts/play.py "Notion AI" --phone +14155551234
```

Frontend (operator UI):

```bash
cd frontend
npm install
npm run dev            # http://localhost:5173  (expects backend on :8000)
```

### Environment

All config loads from `.env` (see [`.env.example`](.env.example) for the full annotated list).
Everything beyond `ANTHROPIC_API_KEY` is optional and degrades gracefully — unset secrets
disable the corresponding feature (Vapi dialing, Logfire export, auth, etc.) rather than
erroring. Model IDs, latency budgets, and feature flags all live in
[`voice_agent/config.py`](voice_agent/config.py).

| Var | Required for | Purpose |
| --- | ------------ | ------- |
| `ANTHROPIC_API_KEY` | everything | All Anthropic LLM calls |
| `GOOGLE_API_KEY` / `GROQ_API_KEY` / `CEREBRAS_API_KEY` | resilience | Interviewer fallback chain (optional) |
| `OPENAI_API_KEY` | optional | Adds GPT-4.1-mini as the first interviewer tier |
| `VAPI_API_KEY`, `VAPI_PHONE_NUMBER_ID`, `WEBHOOK_URL` | phone calls | Outbound dialing + webhooks |
| `VAPI_WEBHOOK_SECRET`, `LLM_SECRET_TOKEN`, `API_AUTH_TOKEN` | prod security | Webhook HMAC / endpoint secrets / API bearer |
| `GOOGLE_CLIENT_ID/SECRET`, `JWT_SECRET` | multi-tenant auth | Google OAuth login + JWT sessions (dev mode when unset) |
| `DATABASE_URL` | prod | Defaults to local SQLite; set Postgres URL in prod |
| `LOGFIRE_TOKEN` | observability | Enables Logfire cloud export (console-only when unset) |
| `VITE_API_URL` | frontend (prod) | Backend URL for the Vite SPA (defaults to `http://localhost:8000`) |

---

## Commands

```bash
# Backend
uv run uvicorn voice_agent.server:app --reload     # API server (webhooks + custom-LLM + REST)
uv run python scripts/play.py "Notion AI"          # local terminal REPL (in-process, no server)
uv run python scripts/play.py "Notion AI" --phone +14155551234   # real outbound Vapi call

# Tests + evals
uv run pytest                                      # everything
uv run pytest tests/ -v                            # unit tests only (DB, server, tracing — no API)
uv run pytest evals/ -v                            # eval tiers (hit the Anthropic API)
uv run pytest -m replay                            # fast deterministic replay evals (CI-safe)
uv run pytest -m slow                              # slow live-simulation evals

# Agent smoke tests
uv run python -m voice_agent.agents.analyst        # analyst against a canned transcript
uv run python -m voice_agent.agents.interviewer    # interviewer REPL with a seeded in-memory DB

# Database migrations (Alembic)
uv run alembic upgrade head                        # apply migrations
uv run alembic revision --autogenerate -m "..."    # create a migration from model changes

# Frontend
cd frontend && npm install && npm run dev          # dev server on :5173
cd frontend && npm run build                       # production build → frontend/dist/
```

---

## Project layout

```
voice_agent/            Backend package
├── server.py           FastAPI app: Vapi webhooks, custom-LLM endpoint, REST API, dialing
├── turn.py             TurnPipeline — per-turn streaming inference + persistence
├── agents/
│   ├── interviewer.py  Real-time agent (5 s budget, fallback model chain)
│   ├── analyst.py      Background analysis agent
│   └── synthesis.py    Post-call report agent
├── state.py            SQLModel tables + DB read/write helpers
├── models.py           Pydantic I/O contracts for every agent
├── config.py           Single source of truth: model IDs, budgets, feature flags, env
├── auth.py             Google OAuth + JWT, multi-tenant org/user model
└── tracing.py          Logfire / OpenTelemetry setup

frontend/               Vite + React Router SPA (projects, calls, reports, settings, login)
├── src/
│   ├── App.tsx         createBrowserRouter — all routes defined here
│   ├── main.tsx        entry point
│   ├── pages/          one file per route
│   ├── components/     NavUser, CallRow, ProjectCard, QuestionEditor, StatusBadge
│   └── lib/
│       ├── api.ts      typed fetch wrapper; reads VITE_API_URL
│       └── auth-context.tsx  AuthProvider; reads ?token= from URL on landing
├── index.html
└── vite.config.ts

evals/                  pydantic_evals suite — Tier 1/2/3 + replay + simulation + synthesis
tests/                  Unit tests (DB, server, tracing, streaming, caching)
alembic/                Database migrations
data/                   investor_questions.yaml — canonical scripted arc
docs/                   architecture.md (authoritative) + design/latency/eval notes
```

---

## Evals

The eval suite lives in `evals/` and uses [`pydantic_evals`](https://ai.pydantic.dev/evals/).
All evals run via `uv run pytest`; thresholds are documented in [docs/architecture.md](docs/architecture.md#eval-suite).

| Tier | File | Marker | What it tests |
| ---- | ---- | ------ | ------------- |
| **1 — interviewer decisions** | `evals/test_interviewer.py` | (none) | Single-turn action choice seeded from `interviewer_turns.yaml`. ActionMatches ≥ 90%, SingleQuestion 100%, warmth ≥ 4/5 |
| **2 — analyst probes** | `evals/test_analyst.py` | (none) | Probe quality against canned transcripts in `analyst_probes.yaml` |
| **3a — replay** | `evals/test_replay.py` | `replay` | Full transcripts: respondent lines fixed, interviewer live. Deterministic + CI-safe. Tests cursor advancement, probe staleness, loop guards |
| **3b — simulation** | `evals/test_trajectories.py` | `slow` | Full conversations with LLM respondent personas (`personas.yaml`). Slow — hits the API |
| **Synthesis** | `evals/test_synthesis.py` | (none) | Post-call report quality against replay transcripts |

```bash
uv run pytest -m replay          # fast, deterministic, safe for CI
uv run pytest -m slow            # full simulation — use sparingly
uv run pytest evals/ -v          # all eval tiers
```

`evals/cases.py` loads YAML fixtures into typed `Case` objects; `evals/evaluators.py` has custom scorers and `LLMJudge` wrappers. In-memory SQLite in evals requires `poolclass=StaticPool`.

---

## Observability (Logfire)

The backend instruments every meaningful span via [Pydantic Logfire](https://logfire.pydantic.dev).
`init_tracing()` in [`voice_agent/tracing.py`](voice_agent/tracing.py) is idempotent and called
from every entrypoint (server lifespan, `play.py`, evals).

- **When `LOGFIRE_TOKEN` is unset** — falls back to console-only structured output. Evals always pass `send_to_logfire=False`.
- **Every span** carries `call_id` + `turn_number` via `agent_span`, so you can slice by call or turn in the Logfire UI.
- **What's traced:** interviewer LLM call (with TTFT, cache tokens, filler flag), analyst task scheduling, synthesis, DB reads, webhook events.
- **Latency metrics:** `llm_ttft_ms` (first token from LLM), TTS start/stop timing from `speech-update` webhooks. See [docs/Latency Measurement.md](docs/Latency%20Measurement.md) for the full breakdown.

Set `LOGFIRE_TOKEN` in `.env` or the Render dashboard to enable cloud export.

---

## Documentation map

| Doc | What it covers |
| --- | -------------- |
| [docs/architecture.md](docs/architecture.md) | **Authoritative.** Full system design, agent contracts, turn pipeline, data model, Vapi integration, observability, evals |
| [docs/Latency Measurement.md](docs/Latency%20Measurement.md) | How latency is measured end-to-end, what each metric means, TTS buffering nuances |
| [docs/evals_plan.md](docs/evals_plan.md) | Eval strategy and tier design |
| [docs/voice-ux-plan.md](docs/voice-ux-plan.md) | Voice UX design notes (fillers, endpointing, barge-in) |
| [docs/TODO.md](docs/TODO.md) | Known edge cases and future improvements |
| [CLAUDE.md](CLAUDE.md) | Repo conventions + architecture summary for AI coding assistants |

---

## Deployment

The backend ships as a Docker image (see [`Dockerfile`](Dockerfile)) and deploys to Render
via [`render.yaml`](render.yaml) — a `web` service for the API (runs `alembic upgrade head`
on boot) and a **static site** service for the Vite frontend (`npm run build` → `dist/`,
with a catch-all rewrite to `index.html` for client-side routing). Production uses Postgres
(`DATABASE_URL`); dev uses a local SQLite file.
