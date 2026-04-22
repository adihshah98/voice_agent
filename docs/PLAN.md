# Plan: Multi-Agent Market Research Call System (v2)

## Context

A voice-based market research interviewer built on Vapi. The core tension: a live call needs low-latency responses (~1.5–2 s), but real theme/contradiction analysis takes 10–15 s. The fix is two cooperating agents with different latency budgets, connected by a shared DB — and a third agent for post-call synthesis.

v2 sharpens three areas the first draft hand-waved:

1. **Agentic loop** — interviewer becomes a real tool-using ReAct agent, not a router.
2. **Traces** — every turn, tool call, token count, and latency is captured in Logfire.
3. **Evals** — a `pydantic_evals` harness covers single-turn decisions, analyst probe quality, and full-conversation trajectories driven by a simulated respondent.

Scope: **prototype** — one real phone call works end-to-end, instrumented, with a runnable eval suite. Not multi-tenant, not production-hardened.

Stack choices (locked in):

- **Voice**: Vapi (webhooks → FastAPI).
- **Agents**: PydanticAI (`Agent` + tool decorators, typed deps + output).
- **LLMs**: Claude Sonnet 4.6 (interviewer, fast), Claude Opus 4.6 (analyst + synthesis).
- **Traces**: Logfire (OTel-based, auto-instruments PydanticAI, FastAPI, httpx, SQLAlchemy).
- **Evals**: `pydantic_evals` — `Dataset`, `Case`, custom `Evaluator`s, `LLMJudge`.
- **DB**: SQLite via SQLAlchemy (single file; Postgres-compatible).

---

## Architecture

```
       ┌──────────────────────────────────────────────────────┐
       │                     Vapi                             │
       │         (STT, TTS, barge-in, turn detection)         │
       └──────────────────────┬───────────────────────────────┘
                              │ webhook per turn
                              ▼
       ┌──────────────────────────────────────────────────────┐
       │                 FastAPI server                       │
       │                 + Logfire spans                      │
       └──────────────────────┬───────────────────────────────┘
                              │
          ┌───────────────────┴────────────────────┐
          │                                        │
          ▼ awaited (≤2 s budget)                  ▼ fire-and-forget
  Interviewer Agent                        Analyst Agent
  (Claude Sonnet 4.6)                      (Claude Opus 4.6)
  PydanticAI tool loop:                    Structured output:
    ask_scripted, ask_probe,                 themes, contradictions,
    acknowledge, clarify,                    surprises, new_probes[]
    flag_off_topic, wrap_up                Writes → DB
  Reads → DB, writes utterance
                              ▲
                              │ on call-ended
                              ▼
                     Synthesis Agent (Opus)
                     Final report: summary, themes,
                     quotes, contradictions, follow-ups
```

**Invariant:** agents never call each other. All coordination through DB. Analyst can crash or lag — interviewer degrades to scripted questions.

---

## File Layout

```
voice_agent/
├── state.py              # SQLAlchemy models, DB helpers
├── models.py             # Pydantic I/O models for all agents
├── tracing.py            # Logfire config, span helpers, correlation IDs
├── interviewer.py        # PydanticAI Agent + tools (ReAct loop)
├── analyst.py            # PydanticAI Agent (structured output)
├── synthesis.py          # PydanticAI Agent (structured output)
├── server.py             # FastAPI webhook + call lifecycle endpoints
├── replay.py             # Replay recorded calls through agents (regression)
├── evals/
│   ├── datasets/
│   │   ├── interviewer_turns.yaml
│   │   ├── analyst_probes.yaml
│   │   └── personas.yaml
│   ├── simulator.py      # Respondent agent (personas) for trajectory eval
│   ├── evaluators.py     # Custom scorers + LLMJudge configs
│   ├── test_interviewer.py
│   ├── test_analyst.py
│   └── test_trajectories.py
├── requirements.txt
├── .env.example
└── README.md
```

---

## Agentic Loop (interviewer.py)

The interviewer is a **real PydanticAI agent** with tools, not a router. Each webhook turn:

```python
# models.py
class InterviewerDeps(BaseModel):
    call_id: str
    db: Session  # or session factory
    turn_number: int

class InterviewerOutput(BaseModel):
    utterance: str                # spoken text back to Vapi
    action: Literal["probe","scripted","clarify","acknowledge","off_topic","wrap_up"]
    reasoning: str                # kept for traces + evals, not spoken

# interviewer.py
interviewer = Agent(
    "anthropic:claude-sonnet-4-6",
    deps_type=InterviewerDeps,
    output_type=InterviewerOutput,
    system_prompt=INTERVIEWER_PROMPT,   # warm, one question at a time, etc.
    instrument=True,                    # → Logfire spans auto
)

@interviewer.tool
def next_scripted_question(ctx: RunContext[InterviewerDeps]) -> str | None:
    """Return the next unanswered scripted question, or None if exhausted."""

@interviewer.tool
def pop_top_probe(ctx: RunContext[InterviewerDeps]) -> Probe | None:
    """Pop highest-priority unasked probe from analyst's queue."""

@interviewer.tool
def recent_turns(ctx: RunContext[InterviewerDeps], n: int = 6) -> list[Turn]:
    """Read last n turns for context on what respondent just said."""

@interviewer.tool
def mark_probe_asked(ctx, probe_id: int) -> None: ...

@interviewer.tool
def scripted_remaining(ctx) -> int: ...
```

System prompt tells the agent:

- One question at a time; acknowledge before pivoting.
- Prefer high-priority probes over scripted questions when one exists.
- If respondent signals ending or has answered everything, call `wrap_up`.
- Clarify only when the previous answer was genuinely ambiguous.
- Never hallucinate a probe — only use `pop_top_probe`.

Latency budget enforced: if the agent loop exceeds 1.8 s, the webhook returns a pre-computed fallback (last scripted question) and logs an `interviewer_slow` event. This keeps the call alive even when the LLM stalls.

---

## Analyst (analyst.py)

```python
class AnalystDeps(BaseModel):
    call_id: str
    db: Session

class NewProbe(BaseModel):
    question: str
    priority: int = Field(ge=1, le=3)
    rationale: str

class AnalysisUpdate(BaseModel):
    themes: list[str]
    contradictions: list[str]
    surprises: list[str]
    new_probes: list[NewProbe]

analyst = Agent(
    "anthropic:claude-opus-4-6",
    deps_type=AnalystDeps,
    output_type=AnalysisUpdate,
    system_prompt=ANALYST_PROMPT,
    instrument=True,
)
```

Called from server with `asyncio.create_task(...)`. Persists `AnalystSnapshot` + new `Probe` rows. Exceptions are caught + logged to Logfire — never crash the live call.

---

## Synthesis (synthesis.py)

```python
class ReportOutput(BaseModel):
    summary: str
    themes: list[ThemeWithQuotes]
    contradictions: list[str]
    key_quotes: list[str]
    follow_up_questions: list[str]

synthesis = Agent(
    "anthropic:claude-opus-4-6",
    output_type=ReportOutput,
    system_prompt=SYNTHESIS_PROMPT,
    instrument=True,
)
```

Triggered on `call-ended`. Writes to `synthesis_reports`. Exposed via `GET /calls/{id}/report`.

---

## Tracing (tracing.py + Logfire)

```python
# tracing.py
import logfire

def init_tracing():
    logfire.configure(service_name="voice-agent")
    logfire.instrument_fastapi(app)
    logfire.instrument_httpx()
    logfire.instrument_sqlalchemy()
    logfire.instrument_pydantic_ai()
```

Per-turn correlation:

```python
with logfire.span(
    "turn",
    call_id=call_id,
    turn_number=n,
    respondent_text=text,
):
    result = await interviewer.run(...)
    logfire.info(
        "interviewer_decision",
        action=result.output.action,
        utterance=result.output.utterance,
        reasoning=result.output.reasoning,
        latency_ms=...,
    )
```

Key attributes on every span: `call_id`, `turn_number`, `agent` (interviewer/analyst/synthesis), `model`, `latency_ms`, `input_tokens`, `output_tokens`. This lets Logfire slice by call, by turn, or by agent.

Vapi webhook payloads + Vapi response are logged as span attributes for replay.

---

## Evals (pydantic_evals)

Three eval tiers. All produce Logfire traces so failures are inspectable.

### Tier 1 — Single-turn decision eval (`test_interviewer.py`)

`Case` = a synthetic DB state + respondent utterance. Expected = the action the interviewer should take.

```yaml
# evals/datasets/interviewer_turns.yaml
- name: "priority_1_probe_beats_scripted"
  inputs:
    scripted_remaining: 4
    top_probe: {question: "You said trust was lost — why specifically?", priority: 1}
    last_respondent: "Yeah, I stopped using them after that."
  expected_output:
    action: "probe"
- name: "no_probe_use_scripted"
  inputs:
    scripted_remaining: 3
    top_probe: null
    last_respondent: "I've used it for about a year."
  expected_output:
    action: "scripted"
- name: "ambiguous_answer_clarify"
  inputs:
    last_respondent: "It's... fine I guess? kind of."
  expected_output:
    action: "clarify"
```

Evaluators:

- `ActionMatches` — exact match on `action` field.
- `UtteranceWarmth` — `LLMJudge` scoring 0–5 on conversational tone.
- `SingleQuestion` — deterministic: utterance contains ≤1 `?`.
- `NoLeadingQuestions` — LLMJudge: does the question bias the answer?

### Tier 2 — Analyst probe quality (`test_analyst.py`)

`Case` = a 5–10 turn transcript. Expected = probe characteristics.

Evaluators:

- `ProbesAreSpecific` — LLMJudge: do probes reference specifics from the transcript (quotes, named things)?
- `ProbesAreNonLeading` — LLMJudge.
- `PriorityIsCalibrated` — LLMJudge: is priority 1 reserved for real contradictions/surprises?
- `NoDuplicateProbes` — deterministic (semantic sim via embedding).
- `StructuralValid` — pydantic validation already gives us this for free.

### Tier 3 — Full-conversation trajectory eval (`test_trajectories.py`)

`simulator.py` defines a **respondent agent** with personas loaded from `personas.yaml`:

```yaml
- name: "chatty_enthusiast"
  system: "You love this product. Give long anecdotes. Occasionally contradict yourself about price."
- name: "terse_skeptic"
  system: "Answer in one sentence. You're not sold on the product. Push back gently."
- name: "off_topic_rambler"
  system: "You drift to unrelated topics after 2 turns. Mention your dog."
- name: "contradictory"
  system: "In turn 1 say you love feature X. By turn 4, say X is broken."
```

Trajectory run = drive the interviewer + analyst against a simulated respondent until `wrap_up` or 20 turns. Evaluators:

- `CallCompletes` — reached `wrap_up` before turn limit.
- `CoveredAllScripted` — every scripted question was asked (or intentionally skipped with reason).
- `CaughtContradiction` — for contradictory persona: analyst produced at least one contradiction probe that was then asked.
- `RedirectedOffTopic` — for off-topic persona: interviewer called `flag_off_topic` within 2 turns.
- `ReportQuality` — LLMJudge on final synthesis report (coverage, non-hallucination vs transcript, actionable follow-ups).

### Running evals

```bash
uv run pytest evals/ -v                  # one-shot
uv run python -m evals.replay <call_id>  # regression against a real call
```

Each `Dataset.evaluate()` streams to Logfire under `service_name=voice-agent-evals`, so eval failures land in the same UI as prod traces. Baseline scores are stored in `evals/baselines.json`; a prompt change that drops any scorer >5% fails CI.

---

## Database Schema (unchanged from v1, one addition)

Add to v1 schema:

```
eval_runs
  id INT PK
  commit_sha TEXT
  dataset TEXT
  case_name TEXT
  scorer TEXT
  score FLOAT
  logfire_trace_id TEXT
  created_at DATETIME
```

Everything else (`calls`, `turns`, `probes`, `analyst_snapshots`, `synthesis_reports`) as in v1.

---

## Server flow (server.py)

```python
@app.post("/vapi/webhook")
async def vapi_webhook(event: VapiEvent):
    with logfire.span("vapi_event", type=event.type, call_id=event.call_id):
        if event.type == "speech-update":
            asyncio.create_task(run_analyst_safely(event.call_id))
            reply = await run_interviewer_with_timeout(event.call_id, event.text, budget_s=1.8)
            return {"message": reply.utterance}
        if event.type == "call-ended":
            mark_complete(event.call_id, event.end_reason)
            asyncio.create_task(run_synthesis_safely(event.call_id))
            return {"ok": True}
```

`run_interviewer_with_timeout` wraps `interviewer.run(...)` in `asyncio.wait_for`; on timeout returns a `scripted` fallback and emits a `timeout` event to Logfire.

Endpoints:

- `POST /calls/start` — seed scripted questions, call Vapi to dial out.
- `GET /calls/{id}/report` — returns synthesis report; 202 if still pending.
- `GET /calls/{id}/trace` — returns Logfire trace URL for quick debugging.

---

## Critical files to create/modify


| File                                                     | Purpose                                           |
| -------------------------------------------------------- | ------------------------------------------------- |
| [state.py](state.py)                                     | SQLAlchemy models + DB helpers                    |
| [models.py](models.py)                                   | Pydantic I/O types for every agent                |
| [tracing.py](tracing.py)                                 | Logfire config, span helpers                      |
| [interviewer.py](interviewer.py)                         | PydanticAI agent + 6 tools + timeout wrapper      |
| [analyst.py](analyst.py)                                 | PydanticAI agent, structured output, safe wrapper |
| [synthesis.py](synthesis.py)                             | PydanticAI agent, final report                    |
| [server.py](server.py)                                   | FastAPI + Vapi webhook + lifecycle endpoints      |
| [evals/simulator.py](evals/simulator.py)                 | Respondent persona agent                          |
| [evals/evaluators.py](evals/evaluators.py)               | Custom + LLMJudge scorers                         |
| [evals/test_interviewer.py](evals/test_interviewer.py)   | Tier 1                                            |
| [evals/test_analyst.py](evals/test_analyst.py)           | Tier 2                                            |
| [evals/test_trajectories.py](evals/test_trajectories.py) | Tier 3                                            |


---

## Build Order

1. **state.py + models.py** — DB + Pydantic types; unit-test happy path.
2. **tracing.py** — Logfire wired; confirm a test span shows in UI.
3. **interviewer.py** — tools + agent; call `agent.run(...)` from a REPL with a seeded DB.
4. **Tier 1 evals** — 15–20 cases; iterate on system prompt until ≥90% `ActionMatches`.
5. **analyst.py** — structured output; feed it a canned transcript; confirm DB writes.
6. **Tier 2 evals** — analyst probe quality.
7. **server.py skeleton** — Vapi webhook, timeout wrapper, local echo test.
8. **evals/simulator.py + Tier 3** — offline trajectory runs before touching real Vapi.
9. **Wire Vapi** — ngrok, test outbound call end-to-end.
10. **synthesis.py** + `/report` endpoint.
11. **replay.py** — replay a real completed call through a new prompt; compare to baseline.

---

## Vapi Flow

## 1. The three concepts

**speech-update** — Vapi's VAD (voice activity detection) telling you about audio-level state changes. Two statuses: `started` (someone is speaking) and `stopped` (they went quiet). Fires on the webhook, no LLM involved. Your server just logs it and returns `{}`.

**conversation-update** — Vapi maintaining its own internal message history. Fires after speech is transcribed and appended, and again after assistant responses are confirmed. Contains the running `messages[]` array. Also webhook-only, no LLM involved.

**utterance boundary** — Not a Vapi event name, it's the internal decision Vapi makes: "this person has finished their thought, time to call the LLM." Vapi combines `speech-update: stopped` + a silence timeout (endpointing threshold) to decide this. Only *then* does it POST to `/vapi/llm/chat/completions`. This is the only path that triggers your code.

Timeline for one turn:

```
speech-update: started     (user starts talking)
speech-update: stopped     (user goes quiet — VAD fires)
[Vapi waits ~300-500ms endpointing window]
→ POST /vapi/llm/chat/completions   ← utterance boundary confirmed
conversation-update                  (Vapi logs the transcript)
speech-update: started     (assistant TTS starts playing)
speech-update: stopped     (assistant TTS done)
conversation-update                  (Vapi logs the assistant turn)

```

---

## 2. Which LLM response wins when Vapi fires multiple?

This is **entirely Vapi's responsibility** — your server doesn't know. Vapi's pipeline works like this:

When a second utterance boundary fires while the first LLM response is still being streamed/played, Vapi treats it as a **barge-in / interruption**. It:

1. Stops the TTS playback of the first response mid-stream
2. Discards the remainder of the first SSE stream
3. Calls your LLM endpoint again with the updated `messages[]` (now including the newer transcript)
4. Plays whatever the second call returns

The **last LLM call to complete wins** from Vapi's perspective. Earlier in-flight calls whose responses arrive after Vapi has moved on get discarded at the TTS layer — Vapi won't play stale audio.

Vapi fires the LLM call **optimistically at the utterance boundary** so that if the user has genuinely finished, the LLM response is already in flight or done by the time TTS needs it. This directly cuts the perceived response latency.

The tradeoff: if the user continues speaking (mid-thought pause), Vapi detects barge-in, kills the in-flight/playing response, and fires a new LLM call with the full updated transcript. You pay for a wasted LLM call but the common case (user actually done) is faster.

---

## 3. Can we demarcate played vs. overridden in current traces?

The signal you want is `speech-update` **with** `role=assistant` **+** `status=started` arriving after an LLM call. That's Vapi telling you "TTS started playing this response." You can correlate by `vapi_call_id` + timing.

The problem right now: your `vapi_speech_update` span logs the fields but they're coming back null in traces, meaning Vapi's webhook payload nests them differently than expected. The fix is two parts:

**First**, log the raw payload shape so we can see exactly what Vapi sends. In [server.py:136-143](vscode-webview://1cs7h1ek6q7ovmprtg9bdot95jcsuqacgmkkvsp9dsk6pmgl8fem/voice_agent/server.py#L136-L143):

```python
if event_type == "speech-update":
    logfire.info(
        "vapi_speech_update",
        vapi_call_id=vapi_call_id,
        status=msg.get("status"),
        role=msg.get("role"),
        raw=msg,   # temporary — remove once confirmed
    )

```

**Second**, once you can see `role=assistant, status=started` events, the cleanest observability approach is: in `vapi_llm_request`, set a `response_id` (e.g. the `X-Trace-ID` already being set as a header). Then if Vapi's `speech-update` payload ever echoes back an ID (some platforms do), you can join them. If not, the timing correlation approach works: last `vapi_llm_request` completed before an `assistant speech-update: started` = the one that was played.

## Verification (end-to-end)

- **Traces:** `logfire` UI shows a timeline for a test call: span per turn, nested spans for interviewer and analyst, with model, tokens, latency.
- **Tier 1 eval passes:** `pytest evals/test_interviewer.py` — ≥90% action match, warmth ≥4/5.
- **Tier 2 eval passes:** analyst probes score ≥4/5 on specificity + non-leading on canned transcripts.
- **Tier 3 eval passes:** all 4 personas reach `wrap_up`; contradictory persona triggers a contradiction probe that gets asked; off-topic persona gets redirected within 2 turns.
- **Graceful degradation:** kill analyst task mid-call (`raise` in a tool) → interviewer still responds from scripted queue; Logfire shows the exception span.
- **Latency budget:** p95 interviewer turn < 2.0 s on the test call; timeouts appear as `interviewer_slow` events and fall back cleanly.
- **Live call:** one real outbound Vapi call to your phone completes; `GET /calls/{id}/report` returns a synthesis report within 30 s of hangup; trace URL opens the full call in Logfire.
- **Regression:** run `replay.py` on the completed call with a modified interviewer prompt — diff of `action` decisions is inspected before merge.

