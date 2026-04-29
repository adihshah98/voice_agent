# Plan: Voice UX Improvements — Latency, Endpointing, Barge-in, TTS, Conversation Design

## Metrics

**Per-turn** — one `turn_latency` log per turn (emitted when assistant starts speaking):

```
stt_ms        STT transcription time
llm_ttft_ms   LLM first token
tts_ttft_ms   TTS first audio (derived: total - stt - llm)
total_ms      End-to-end from user stopped → first audio heard

```

**Per-call** — Vapi's `vapi_endpointing_ms_avg`, `vapi_stt_ms_avg` etc. on `call_ended` (all 4 stages averaged). You'll now see `vapi_perf_metrics_absent` as a **warning** if Vapi isn't sending that payload, so you'll know immediately if the data path is wrong.

The one component you can't get per-turn from our server is **endpointing latency** — that's the configurable silence timeout in Vapi before VAD fires, and it's not observable to us. Vapi's per-call average is the only place we'll ever see it.

The **only metric that actually matters** is:

```
T_last_audio_byte_received → T_first_audio_byte_sent
```

Everything in between is a pipeline implementation detail. The sub-segment breakdowns only make sense for **debugging where time is being lost**, not for representing user-perceived latency.

---

### Why Sub-Segment Math Breaks Down

In a streaming pipeline:

```
t=0ms    Last audio byte arrives
t=0ms    STT has already been processing for 2000ms (overlapped with speech)
t=80ms   VAD fires (endpointing)
t=120ms  Final transcript ready (STT marginal)
t=130ms  LLM first token arrives (LLM was speculatively started at t=100ms)
t=210ms  First TTS audio chunk sent out
```

If you add up "endpointing + STT + LLM TTFT + TTS TTFT" you'd get ~700ms. But actual E2E was **210ms**. The sub-segments are overlapping — you can't sum them.

---

### What You Actually Measure

**One primary metric:**

python

```python
t0 = timestamp of last audio packet received from user
t1 = timestamp of first audio packet written to output stream

e2e_voice_latency = t1 - t0
```

**Sub-segments only for debugging (wall-clock deltas, not summed):**

python

```python
# Each measured from t0, not from each other
endpointing_from_t0   = t_vad_fired - t0
transcript_from_t0    = t_final_transcript - t0
llm_first_token_from_t0 = t_llm_first_token - t0
first_audio_from_t0   = t1 - t0  # = e2e_latency
```

This gives you a **waterfall**, not a sum. You can see where in the pipeline time is being spent relative to T0.

---

### What Vapi Does

Vapi exposes this in call analytics as `assistant_latency` — defined exactly as: end of user speech → start of assistant audio. That's the number they surface in their dashboard and webhooks.

Internally they also log sub-segments but the contract they expose to customers is the single E2E number, because that's what maps to user experience.

If you're building on Vapi, the `call.ended` webhook payload includes latency metadata. For deeper instrumentation you'd need to run your own timestamps at the WebRTC/audio stream layer since Vapi abstracts the internals.

---

### The Right Mental Model

Think of it like a **Gantt chart**, not a serial chain:

```
t0 (last audio byte)
│
├── VAD processing          ████░░░░░░░░░░░░░░░░░░
├── STT finalization             ████░░░░░░░░░░░░░
├── LLM streaming                    ████████░░░░░
├── TTS first chunk                          █████
│                                                │
t1 (first audio out) ────────────────────────────┘

E2E = t1 - t0  ← this is your metric
Sub-segments = each bar's start relative to t0 ← these are your debug tools
```

The sub-segments tell you **which stage is your bottleneck**, but the only number you report is E2E.

## Context

The voice pipeline from user-stops-speaking to first audio has 5 stages where time accumulates:

```
User stops speaking
  → VAD detects silence        (Vapi/Deepgram, ~100-400ms)
  → STT transcribes            (Deepgram nova-2, ~100-300ms)
  → LLM first token            (Haiku 4.5, ~200-800ms)
  → TTS first audio chunk      (ElevenLabs standard, ~300-400ms → Flash = 75ms)
  → Audio delivered to user    (~50-150ms)
```

Right now: no endpointing config, no barge-in config, no acknowledgement phrases, and the TTS model defaults to ElevenLabs standard (~400ms latency, robotic). The interviewer prompt is also missing spoken-language conventions (acknowledgment tokens, number spelling, short-turn enforcement).

---

## Phase 1 — Vapi Voice Pipeline Config

**Files:** `voice_agent/server.py`, `voice_agent/config.py`, `.env.example`

### 1a. Endpointing (Turn Detection)

Goal: don't fire the LLM when user pauses mid-thought; do fire when user is genuinely done. Research interviews need longer silence tolerance than transactional agents.

Add `startSpeakingPlan` to the assistant payload in `_dial_vapi`:

```json
"startSpeakingPlan": {
  "smartEndpointingPlan": { "provider": "livekit" },
  "transcriptionEndpointingPlan": {
    "onPunctuationSeconds": 0.2,
    "onNoPunctuationSeconds": 2.0,
    "onNumberSeconds": 0.8
  },
  "waitSeconds": 0.3
}
```

- **LiveKit smart endpointing**: AI-based model that understands grammatical completeness, not just silence. Works alongside nova-2 (no transcriber change). Prevents cutting off "I'd say... it's been really valuable for..."
- `**onNoPunctuationSeconds: 2.0`** (vs default 1.5): extra patience for thoughtful pauses in research context.
- `**waitSeconds: 0.3`** (vs default 0.4): slightly faster response once endpointing fires.
- Transcription rules are the fallback when LiveKit confidence is low.

Add config fields to `Settings`:

```python
vapi_wait_seconds: float = 0.3
```

### 1b. Barge-in & Interruption Handling

Goal: "hmm", "yeah", "right" mid-assistant-speech should NOT stop the assistant. A real interruption (3+ words) should.

Add `stopSpeakingPlan` to the assistant payload:

```json
"stopSpeakingPlan": {
  "numWords": 3,
  "backoffSeconds": 1.0,
  "acknowledgementPhrases": [
    "hmm", "mm-hmm", "yeah", "yes", "okay", "ok",
    "right", "uh-huh", "sure", "got it", "I see",
    "totally", "absolutely", "interesting"
  ]
}
```

- `**numWords: 3**`: user must say 3 words before assistant stops. Single-word backchannels ("hmm") won't interrupt.
- `**acknowledgementPhrases**`: even if the user says "yeah" clearly enough to count as speech, Vapi ignores it as an interruption and lets the assistant continue.
- `**backoffSeconds: 1.0**`: after a real interruption, assistant waits 1s before responding (avoids immediately talking over the user again).

Add config fields:

```python
vapi_stop_num_words: int = 3
vapi_stop_backoff_seconds: float = 1.0
```

### 1c. Transcriber upgrade: nova-2 → nova-3

Deepgram nova-3 is faster and more accurate on phone audio (compressed codec, background noise). Simple one-line change in `_dial_vapi`:

```python
"transcriber": {"provider": "deepgram", "model": "nova-3", "language": "en"},
```

### 1d. Dead air handling

If the respondent goes silent mid-call (> 10s), Vapi can send a nudge automatically. Add to assistant payload:

```json
"silenceTimeoutSeconds": 10,
"silenceTimeoutMessage": "Still there?"
```

### 1e. Config consolidation in server.py

Refactor `_vapi_assistant_voice()` and inline payload construction into named helpers:

```python
def _build_voice_config() -> dict:         # was _vapi_assistant_voice()
def _build_start_speaking_plan() -> dict:  # new
def _build_stop_speaking_plan() -> dict:   # new
```

---

## Phase 2 — TTS Improvement

**Files:** `voice_agent/server.py`, `voice_agent/config.py`

### 2a. ElevenLabs model: standard → Flash v2.5

`eleven_flash_v2_5` cuts TTS latency from ~400ms to ~75ms. Still sounds natural.

Add to `Settings`:

```python
vapi_voice_model: str | None = None   # "eleven_flash_v2_5" recommended
```

Add to `_build_voice_config()`:

```python
if settings.vapi_voice_model:
    voice["model"] = settings.vapi_voice_model
```

Set in `.env`:

```
VAPI_VOICE_MODEL=eleven_flash_v2_5
```

### 2b. Voice parameter tuning (`.env` only)

```
VAPI_VOICE_STABILITY=0.45
VAPI_VOICE_SIMILARITY_BOOST=0.75
VAPI_VOICE_STYLE=0.10
VAPI_VOICE_SPEED=0.95
```

---

## Phase 3 — Interviewer Prompt: Spoken-Language Conventions

**File:** `voice_agent/agents/interviewer.py` — `INTERVIEWER_PROMPT`

### 3a. Acknowledgment tokens

Start every response with a brief acknowledgment token before the question. Mandatory — never open with a bare question. Also buys ~200-400ms of effective latency (filler plays while LLM generates the rest).

### 3b. Spoken-number enforcement

Spell out numbers: "about thirty percent" not "about 30%".

### 3c. Name deflection

If respondent asks who you are, deflect briefly and pivot immediately back to the question.

### 3d. Short-turn enforcement

Tighten to 25 words. No multi-part questions.

### 3e. Thinking-pause handling

If utterance is very short ("um", "let me think"), respond with "Take your time." only — do not ask a question.

---

## Phase 4 — Robustness Edge Cases


| Scenario                                           | Fix                                       | Status    |
| -------------------------------------------------- | ----------------------------------------- | --------- |
| Backchannel ("hmm", "yeah") interrupting assistant | `stopSpeakingPlan.acknowledgementPhrases` | Phase 1b  |
| User silent > 10s                                  | `silenceTimeoutSeconds: 10`               | Phase 1d  |
| LLM timeout                                        | `_fallback()` (already implemented)       | No change |
| Off-script respondent                              | OFF-TOPIC action (already in prompt)      | No change |


---

## Out of Scope (future work)

- **Filler audio** ("Give me a moment...") — needs pre-recorded clips + Vapi audio injection
- **LiveKit full barge-in control** — only if Vapi's built-in config is insufficient
- **Deepgram Flux** — better end-of-turn scoring but requires transcriber swap
- Also, looks' loike it is waiting for some time before spakign --> So even though ttft is <600ms the time since I stop spekaing is long

---

## Verification

1. Server starts: `uv run uvicorn voice_agent.server:app --reload`
2. Dial succeeds: POST `/calls/start` with phone — no 4xx from Vapi
3. **Endpointing**: pause 1-2s mid-sentence → wait; 3s after finishing → LLM fires
4. **Backchannel**: say "yeah" while assistant speaks → assistant continues
5. **Interruption**: say full sentence while assistant speaks → assistant stops, responds
6. **TTS**: `eleven_flash_v2_5` — noticeably less robotic, faster first word
7. **Name deflection**: ask "what's your name?" → warm deflect + immediate pivot
8. **Acknowledgment**: every response begins with "Got it.", "Right.", etc.
9. **Dead air**: silent for 12s → Vapi sends "Still there?"

