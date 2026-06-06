## Todo

- Evals
  - Tier 3: 
    - See if it is covering enough
      - Make sure it asks everything - Done
      - When handling a probe from earlier, it should say like you said earlier - Done 
      - Silence not working
      - Scripted Skip not working
      - Context compression for topics covered
      - Context drift: agent forgets or misinterprets earlier info
      - Make sure it wraps up with everything covered in a set time but probes at correct depth
      - Make sure it doesn't ask again/get stuck in loops/rabbitholes - ENd if responder is not talking/off-topic. Do'nt wait till all scripted are asked
  - Misc
    - Diff btw replay & trajectory
    - See if we are testing appropriate things
    - Should test an E2E interview but efficiently
    - Ask if it's prod grade
    - Best way to track state transitions etc. of the agents? (Like still there etc.)
    - Run the evals (See if working & fix what is not)
  - Versioned datasets/eval runs: 
    - How to view & compare nicely
  - Online Evals: 
    - Only returning that probes were generated - we should add more
- Infra - Prod Level
  - Render Deployment 
  - Live DB + Alembic (Remove all alter tables)
  - Rate limiting
- Multi-tenant 
  - Eventual goal is per customer, per call, per project level configurabilityt across many customers, with a frotnend to be able to configure it. Design keeping that in mind
    - Tell it it is diligencing which product & some knowledge abt it
    - Tell it which direction to go, where not to spend too much time
    - If not customization, uses the default
    - Synthesis
  - Multi-tenant auth

---

---

## Future improvements

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
- Better Agent Loop
  - State Machine to manage loops/rabbitholes?
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

