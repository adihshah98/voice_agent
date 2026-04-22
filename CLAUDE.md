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

Multi-agent voice research interviewer built on Vapi. **PLAN.md** is the authoritative design doc; read it when making non-trivial changes. Read-over-skim if you touch `interviewer.py`, `analyst.py`, or the webhook flow.

### The three agents and their latency contract

Three PydanticAI agents, each with a different latency budget — this is the core design constraint:

1. **Interviewer** ([voice_agent/agents/interviewer.py](voice_agent/agents/interviewer.py)) — Haiku 4.5, one LLM call per turn, hard 5 s deadline via `asyncio.wait_for`. **Not** a ReAct tool loop: all DB state is pre-fetched into a CONTEXT block (`_build_context`) and the model returns a single structured `InterviewerOutput`. Python then applies side effects (marking probe/scripted as asked) based on the output. If the deadline fires, `_fallback()` returns the next scripted question.
2. **Analyst** ([voice_agent/agents/analyst.py](voice_agent/agents/analyst.py)) — Sonnet 4.6, fire-and-forget via `asyncio.create_task`. Runs **only after a `scripted` turn** (see [voice_agent/turn.py:67](voice_agent/turn.py#L67)) — not after every turn. Reads prior `AnalystSnapshot` as established context + only the new turns since, so prompt size stays bounded on long calls. Wrapped in `run_analyst_safely` — exceptions are swallowed so a crashing analyst never affects the live call.
3. **Synthesis** ([voice_agent/agents/synthesis.py](voice_agent/agents/synthesis.py)) — Sonnet 4.6, post-call. Gated by `ENABLE_SYNTHESIS_REPORT` in [voice_agent/config.py](voice_agent/config.py) (currently **False** — keep that in mind when testing the end-of-call path).

**Invariant: agents never call each other.** All coordination flows through SQLite tables in [voice_agent/state.py](voice_agent/state.py). Model IDs, timeouts, and the synthesis flag all live in [voice_agent/config.py](voice_agent/config.py) — swap a model there, nothing else changes.

### Turn pipeline

The single entry point for a speech turn is `run_speech_turn` in [voice_agent/turn.py](voice_agent/turn.py). Both the Vapi custom-LLM webhook (`/vapi/llm/chat/completions` in [voice_agent/server.py](voice_agent/server.py)) and the local REPL (`scripts/play.py`) call it — so in-process REPL and production call share one code path. The custom-LLM endpoint returns OpenAI-shaped responses (with optional SSE streaming) because Vapi expects an OpenAI-compatible shape.

### Vapi lifecycle

Two endpoints, different roles:

- `/vapi/llm/chat/completions` — per-turn inference (replaces OpenAI from Vapi's perspective).
- `/vapi/webhook` — lifecycle events only: `status-update` (flips `Call.status` pending→active) and `end-of-call-report` (flips to ended, schedules synthesis).

Outbound dialing (`_dial_vapi`) writes the Vapi-assigned `vapi_call_id` back onto the `Call` row after the POST returns. See PLAN.md §"Correctness Edges" for the race condition this creates — first Vapi event is usually `ringing`, which is long enough to cover the write, but it's fragile.

### Database (SQLModel/SQLite)

Single file `voice_agent.db`; schema is Postgres-compatible. Key reads used by the interviewer live alongside the tables in [voice_agent/state.py](voice_agent/state.py): `next_scripted`, `top_probes` (priority asc, then age), `recent_turns`, `latest_snapshot`, `turns_since`. The analyst uses `turns_since(snapshot.after_turn)` to stay context-bounded. `scripted_cursor` on `Call` advances on both `scripted` and `skip_scripted` actions — the model can skip a scripted question that was already answered organically.

### Tracing

`init_tracing()` in [voice_agent/tracing.py](voice_agent/tracing.py) is idempotent and called from every entrypoint (server lifespan, play.py, evals). When `LOGFIRE_TOKEN` is unset it falls back to console-only output — evals explicitly pass `send_to_logfire=False`. Every meaningful span carries `call_id` + `turn_number` via `turn_span` / `agent_span` so Logfire can slice by call or turn. `log_interviewer_decision` emits the `interviewer_decision` event with action/utterance/reasoning/latency — this is the primary observability hook.

### Evals

Three tiers under `evals/`, all using `pydantic_evals`:

- **Tier 1** ([evals/test_interviewer.py](evals/test_interviewer.py)) — single-turn decision eval, seeds in-memory SQLite per case from `interviewer_turns.yaml`. Thresholds: ActionMatches ≥90%, SingleQuestion 100%, warmth ≥4/5, non-leading ≥90%.
- **Tier 2** ([evals/test_analyst.py](evals/test_analyst.py)) — analyst probe quality against canned transcripts in `analyst_probes.yaml`.
- **Tier 3** ([evals/test_trajectories.py](evals/test_trajectories.py)) — full conversation driven by `evals/simulator.py` respondent personas (`personas.yaml`) against the real interviewer + analyst.

`evals/cases.py` loads YAML into typed `Case` inputs; `evals/evaluators.py` holds custom scorers and `LLMJudge` configs. In-memory SQLite needs `poolclass=StaticPool` — without it each session gets a fresh empty DB.

### Scripted question source

`scripts/play.py` loads scripted questions from `evals/datasets/investor_questions.yaml` and substitutes `[product]` with the CLI arg. This is the canonical question list; the server's `/calls/start` takes arbitrary `scripted_questions` from the caller.