## Things to watch out for/might break

- **Race condition:** `vapi_call_id` is written to DB *after* `POST /call/phone` returns, but Vapi can fire `assistant.started` before that write commits. In practice the first event is usually `status-update` (`ringing`) which is long enough — but it's fragile.
- Probe popping
- Context compression for topics covered
- Context drift: agent forgets or misinterprets earlier info

## My questions/Future improvements

- Clean up Tech Debt
  - See Cursor Plan
- Eval Infra
  - Evals
    - E2E Evals
      - **The single most underrated item on your list:** stopping the REPL path from diverging further. It's already happening and it quietly invalidates your eval results — which are the foundation of everything else.
      - The Evals for Trajectory call - **do E2E evals—but in a *very constrained, layered, and replay-heavy way*.** Not brute-force 1-hour runs.
      - Eventually do something about the non-Vapi path (used only for evals) functions like run_speech_run in turn.py, db_messages_fallback, prepare_interviewer_turn
      - REPL Path is already diverging from prod
      - In the hot path (`turn.py`), the session is closed *before* `interviewer.run()` is awaited (hence `Session | None` — the docstring explains this). So `deps.session` is `None` during the actual LLM call in production; it's only non-None in the REPL and evals. run_interviewer & run_interviewer_with_timeout are only called by evals
    - Online Evals
  - Versioned prompts/datasets/eval runs
  - Alerts
    - Alerts created, but no channel added yet
  - Live Observability
    - No per-call cost tracking (input/output tokens × rate).
    - No alerting on `vapi_unknown_call` or `vapi_dial_error` — they just log.
  - Logging
    - Should we be logging more stuff so if it fails, you can see it. Also log instead of trace?
- Conversation Trajectory
  - Make sure it asks everything
  - Make sure it probes at the correct depth
  - Make sure it wraps up with everything covered in a set time
  - Make sure it doesn't ask again/get stuck in loops/rabbitholes
  - Make sure it deals with non-happy path behavior
  - Repeated the tell me more abt day to day - "The respondent has provided some information about their role and team, but more context is needed to understand their daily activities and how Notion is used." In this case, ask it to specify exactly what you want to know more about
  - When handling a probe from earlier, it should say like you said earlier
- Multi-tenant 
  - See, eventual goal is per customer, per call, per project level configurabilityt across many customers, with a frotnend to be able to configure it. Design keeping that in midn
    - Tell it it is diligencing which product
    - Tell it which direction to go, where not to spend too much time
    - If not customization, uses the default
- Infra - Prod Level
  - Webhook correctness (auth/idempotency)
  - Live DB + Alembic
  - Caching & Latency
    - Add *tts*active to Redis
  - Model Pinning & Rollback
  - Secrets Management
  - Dependency Mgmt on pyproject.timl and remove requirements.txt
  - The analyst is triggered by polling `should_run_analyst()` on every `conversation-update`. Fine for one call, but with N concurrent calls you get lock contention and polling overhead. Production systems use a task queue (Celery + Redis, SQS, etc.) — the webhook handler enqueues a job instead of calling `asyncio.create_task` inline.
  - The analyst competes with the real-time interviewer for the event loop. A slow Sonnet call during a burst can delay turn responses. In production you'd want the analyst as a separate worker service — the invariant holds, you just move the writes to a different process.
  - SQS Queues
  - Render Deployment + CI/CD
  - Multi-tenant auth
  - Rate limiting
  - **Feature flags / kill switches**: disable analyst, disable probes, force scripted-only mode during incidents

---

---

- Memory
  - Memory/Improving agents with usage
- Advanced Voice UX
  - Advanced Voice UX features & Voice UX Evals
    - Make the fillers sound more natural
    - Livekit/Pipecat & Deepgram: In production systems, it's not acceptable to wait for 1+ seconds (to decide if user is done talking w/o punctuation), but also not acceptable to interrupt users mid thought - how would it be done irl.
      - Speculative LLM Firing: Calling LLM before VAD done: That's a managed-service tax. If sub-500ms E2E latency is a hard requirement, the honest answer is **Vapi is the constraint** — frameworks like LiveKit Agents or Pipecat running on your own infra are the production best practice for latency-critical voice AI. Fire the LLM on the *interim transcript* — before endpointing confirms. If the user continues speaking, cancel the inflight request and refire with the updated transcript. The wasted token cost is negligible vs. the latency win.
      This requires streaming STT with interim results (Deepgram supports it), and a cancellation mechanism on the LLM side. **This is the technique that cuts perceived latency in half.** It works because most of the time, the user's last ~200ms of audio doesn't change the semantic meaning.
      - Endpointing: Custom EOT Models: RIght now, EOT is taking 1s
        - Deepgram flux: To reduce aggressive cutting on long answers (Not part of deepgram free tier)
      - **No backchannels** — The bot can't say "mm-hmm" mid-answer. This is the single biggest voice UX gap vs. a human interviewer. It's a Vapi architectural constraint — true backchannels require LiveKit/Pipecat on your own infra (the TODO already flags this). Not worth solving now unless latency and reliability are solid.
      - **LiveKit full barge-in control** — only if Vapi's built-in config is insufficient
    - Model routing by intent/tier
    - Change intonation of the fillers
- Not too important
  - Synthesis Report
    - Reinstate synthesis report once this works
    - Maybe use async queues for this (learn to use async queues either way)

