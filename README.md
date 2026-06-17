# Voice Research Interviewer

An AI-powered outbound voice agent that conducts structured investor / user-research
interviews over the phone. It dials a respondent, runs a scripted question arc while
adaptively probing on interesting signals, analyzes the conversation in the background,
and (optionally) produces a post-call research report.

Built on **[Vapi](https://vapi.ai)** for telephony and **[PydanticAI](https://ai.pydantic.dev)**
for agent orchestration, with a **FastAPI** backend, **Next.js** frontend, and
**Pydantic Logfire** for observability.

### 🚀 Live demo

**Operator UI → [voice-agent-frontend-gelo.onrender.com](https://voice-agent-frontend-gelo.onrender.com/)**

Deployed on **Render** (Dockerized FastAPI API + Node/Next.js frontend, see [`render.yaml`](render.yaml))
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
npm run dev            # http://localhost:3000  (expects backend on :8000)
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

frontend/               Next.js operator UI (projects, calls, reports, settings)
evals/                  pydantic_evals suite — Tier 1/2/3 + replay + synthesis
tests/                  Unit tests (DB, server, tracing, streaming, caching)
alembic/                Database migrations
data/                   investor_questions.yaml — canonical scripted arc
docs/                   architecture.md (authoritative) + design/latency/eval notes
```

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
on boot) and a Node `web` service for the Next.js frontend. Production uses Postgres
(`DATABASE_URL`); dev uses a local SQLite file.
</content>
