# Plan: Voice UX Improvements — Latency, Endpointing, Barge-in, TTS, Conversation Design

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

