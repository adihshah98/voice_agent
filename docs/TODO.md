## Things to watch out for/might break

- **Race condition:** `vapi_call_id` is written to DB *after* `POST /call/phone` returns, but Vapi can fire `assistant.started` before that write commits. In practice the first event is usually `status-update` (`ringing`) which is long enough — but it's fragile.
- Probe popping
- Context compression for topics covered
- Context drift: agent forgets or misinterprets earlier info

## My questions/Future improvements

- Latency
  - **No prompt caching** on Anthropic calls — the system prompt is reprocessed every turn. Needs `cache_control` breakpoints.
  - Real streaming
  - 1. DB sessions around LLM call
    [turn.py:41-49](vscode-webview://1cs7h1ek6q7ovmprtg9bdot95jcsuqacgmkkvsp9dsk6pmgl8fem/voice_agent/turn.py#L41-L49): Two separate sessions bracket the LLM call — good. But `_build_context` does 4 DB queries sequentially inside a session that's held open during the LLM call:
    - `next_scripted`, `scripted_remaining`, `top_probes`, `latest_snapshot`
    These could run concurrently (they're all reads), but since it's SQLite that's moot. Not a real bottleneck unless you move to Postgres.
    ### 4. `_resolve_vapi_call_id` on every turn
    [server.py:376](vscode-webview://1cs7h1ek6q7ovmprtg9bdot95jcsuqacgmkkvsp9dsk6pmgl8fem/voice_agent/server.py#L376): Called on every `/vapi/llm` request with up to 5 retries × exponential backoff (3.1s worst case). After turn 1, the call is always in the DB — no need for retries on subsequent turns.
    **Fix:** Add a simple in-memory dict `_vapi_id_cache: dict[str, str]` that maps `vapi_call_id → call_id` so retries only happen once per call lifecycle.
    ### 5. Analyst: two sequential LLM calls when compression fires
    [analyst.py:200-203](vscode-webview://1cs7h1ek6q7ovmprtg9bdot95jcsuqacgmkkvsp9dsk6pmgl8fem/voice_agent/agents/analyst.py#L200-L203): `run_analyst` does `analyst.run(...)` then `_maybe_compress_subtopics(...)` — these are sequential. The compression call is a second Sonnet call every 25 turns.
    **Fix:** Run them in parallel with `asyncio.gather` when compression is needed, since the compression agent doesn't depend on the analyst result.
    ### 6. Compression agent reads full transcript redundantly
    [analyst.py:178-180](vscode-webview://1cs7h1ek6q7ovmprtg9bdot95jcsuqacgmkkvsp9dsk6pmgl8fem/voice_agent/agents/analyst.py#L178-L180): `_maybe_compress_subtopics` re-fetches `recent_turns(..., n=200)` even though `_build_prompt` already fetched the same turns. Minor but wasteful.
    ### 7. `httpx.AsyncClient()` not reused
    [server.py:539](vscode-webview://1cs7h1ek6q7ovmprtg9bdot95jcsuqacgmkkvsp9dsk6pmgl8fem/voice_agent/server.py#L539) and [server.py:458](vscode-webview://1cs7h1ek6q7ovmprtg9bdot95jcsuqacgmkkvsp9dsk6pmgl8fem/voice_agent/server.py#L458): New `httpx.AsyncClient()` created per request. Should be a module-level singleton with connection pooling — matters under concurrent calls.
- Voice Nuances/Vapi Config
  - Barge in etc.
  - Gate LLM calls till final-transcript (more chatlike)
  - Debounce partial updates (Wait 500ms of silence before triggering LLM)
  - Reduce ASR chunk frequency
  - See how to modify Vapi config
  - If the respondent says hmm, or yes - it shouldn't stop
  - If responder is taking time to think, it shoudn't interrupt/know when to interrupt
  - If it asks for name, it should not give it. But make the transition back to the question smoother
- TTS
  - Right now, it's too fake sounding
- Multi-tenant 
  - See, eventual goal is per customer, per call, per project level configurabilityt across many customers. Design keeping that in midn
    - Tell it it is diligencing which product
    - Tell it which direction to go, where not to spend too much time
    - If not customization, uses the default
- Synthesis Report
  - Reinstate synthesis report once this works
- Eval Infra
  - Live Observability
    - No per-call cost tracking (input/output tokens × rate).
    - No alerting on `vapi_unknown_call` or `vapi_dial_error` — they just log.
  - Evals
    - E2E Evals
      - The Evals for Trajectory call - **do E2E evals—but in a *very constrained, layered, and replay-heavy way*.** Not brute-force 1-hour runs.
    - Online Evals
- Minor changes
  - Cache --> Recent turns, remaining scripted in Anthropic prompts
- Infra - Prod Level
  - Webhook correctness (auth/idempotency)
  - See where traces go & have a good observabilty process/dashnoard
  - Vesioned prompts/datasets/eval runs
  - Live DB + Alembic
  - Caching & Latency
    - Add *tts*active to Redis
  - Prompt Caching
  - Model Pinning & Rollback
  - Secrets Management
  - Dependency Mgmt on pyproject.timl and remove requirements.txt
- Infra - Deployment Level
  - SQS Queues
  - Secrets manager 
  - Render Deployment + CI/CD
  - Multi-tenant auth
  - Rate limiting
  - **Feature flags / kill switches**: disable analyst, disable probes, force scripted-only mode during incidents
- Advanced
  - Barge-in / interruption handling beyond Vapi's capabilities - Using Livekit

