## Things to watch out for/might break

- **Race condition:** `vapi_call_id` is written to DB *after* `POST /call/phone` returns, but Vapi can fire `assistant.started` before that write commits. In practice the first event is usually `status-update` (`ringing`) which is long enough — but it's fragile.
- Probe popping
- Context compression for topics covered
- Context drift: agent forgets or misinterprets earlier info

## My questions/Future improvements

- Evals - We have the base setup for all. We need to go 1 level deeper and se if this is properly targeting what we want
  - For all tiers: Run the evals (See if working & fix what is not), Ask if it's prod grade, See if we are testing appropriate things
    - Tier 1: (Interviewer)
    - Tier 2 (Analyst)
    - Tier 3: Is this covering enough
      - Make sure it asks everything
      - Make sure it probes at the correct depth
      - Make sure it wraps up with everything covered in a set time
      - Make sure it doesn't ask again/get stuck in loops/rabbitholes
      - Make sure it deals with non-happy path behavior
      - Repeated the tell me more abt day to day - "The respondent has provided some information about their role and team, but more context is needed to understand their daily activities and how Notion is used." In this case, ask it to specify exactly what you want to know more about
      - When handling a probe from earlier, it should say like you said earlier
  - Versioned datasets/eval runs: 
    - Where are these saved, how to view & compare nicely, is this best practice? Hosted vs Local? Why print?
  - Online Evals: See if working fine
- Infra - Prod Level
  - Render Deployment 
  - Webhook correctness (auth/idempotency)
  - Live DB + Alembic
  - Rate limiting
- Multi-tenant 
  - See, eventual goal is per customer, per call, per project level configurabilityt across many customers, with a frotnend to be able to configure it. Design keeping that in mind
    - Tell it it is diligencing which product & some knowledge abt it
    - Tell it which direction to go, where not to spend too much time
    - If not customization, uses the default
    - Synthesis
  - Multi-tenant auth

---

---

- Evals
  - Host datasets directly on Logfire (Currently we run it on local and save runs there, now we can host datasets, collab etc. directly on LF)
  - Hosted prompt & prompt versioning on Logfire
- Prod Infra
  - Multi-server deployment - cleaning up data that is in-memory worker dependant
  - CI/CD
  - SQS Queues
  - Secrets Mgmt.
  - Model Pinning & Rollback
  - Caching & Latency
    - Add *tts*active to Redis
  - **Feature flags / kill switches**: disable analyst, disable probes, force scripted-only mode during incidents
  - The analyst is triggered by polling `should_run_analyst()` on every `conversation-update`. Fine for one call, but with N concurrent calls you get lock contention and polling overhead. Production systems use a task queue (Celery + Redis, SQS, etc.) — the webhook handler enqueues a job instead of calling `asyncio.create_task` inline.
  - The analyst competes with the real-time interviewer for the event loop. A slow Sonnet call during a burst can delay turn responses. In production you'd want the analyst as a separate worker service — the invariant holds, you just move the writes to a different process.
- Memory
  - Memory/Improving agents with usage
- Advanced Voice UX
  - Advanced Voice UX features & Voice UX Evals
    - Make the fillers sound more natural
    - Livekit/Pipecat & Deepgram: In production systems, it's not acceptable to wait for 1+ seconds (to decide if user is done talking w/o punctuation), but also not acceptable to interrupt users mid thought - how would it be done irl.
      - Speculative LLM Firing: Calling LLM before VAD done: That's a managed-service tax. If sub-500ms E2E latency is a hard requirement, the honest answer is **Vapi is the constraint** — frameworks like LiveKit Agents or Pipecat running on your own infra are the production best practice for latency-critical voice AI. Fire the LLM on the *interim transcript* — before endpointing confirms. If the user continues speaking, cancel the inflight request and refire with the updated transcript. The wasted token cost is negligible vs. the latency win.
      This requires streaming STT with interim results (Deepgram supports it), and a cancellation mechanism on the LLM side. **This is the technique that cuts perceived latency in half.** It works because most of the time, the user's last ~200ms of audio doesn't change the semantic meaning.
      - **No backchannels** — The bot can't say "mm-hmm" mid-answer. This is the single biggest voice UX gap vs. a human interviewer. It's a Vapi architectural constraint — true backchannels require LiveKit/Pipecat on your own infra (the TODO already flags this). Not worth solving now unless latency and reliability are solid.
      - **LiveKit full barge-in control** — only if Vapi's built-in config is insufficient
    - Model routing by intent/tier
- Not too important
  - Synthesis Report
    - Reinstate synthesis report once this works
    - Maybe use async queues for this (learn to use async queues either way)
  - Fun Stuff
    - Clone my voice on 11labs & use it

