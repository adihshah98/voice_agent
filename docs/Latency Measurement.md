# Latency Measurement

## Definitions

| Field                 | Meaning                                                                                             | How computed / source                                                                               |
| --------------------- | --------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `llm_ttft_ms`         | Pure LLM internal TTFT, unaffected by filler or DB prep                                             | `time.perf_counter()` inside `stream_tokens()`; stashed in `_speech_ts`, logged in `turn_latency`  |
| `filler_injected`     | Whether a filler phrase was streamed before LLM tokens                                              | stashed from `stream_tokens()`, logged in `turn_latency`                                            |
| `vapi_endpointing_ms` | Last audio byte → VAD fires                                                                         | `end-of-call-report turnLatencies[i]`                                                               |
| `vapi_stt_ms`         | VAD fires → final transcript ready                                                                  | `end-of-call-report turnLatencies[i]`                                                               |
| `vapi_llm_ms`         | LLM request arrival → first token at Vapi's boundary (includes our DB prep + filler token)         | `end-of-call-report turnLatencies[i]`                                                               |
| `vapi_tts_ms`         | First LLM token → first audio out                                                                   | `end-of-call-report turnLatencies[i]`                                                               |
| `vapi_turn_ms`        | True user-perceived E2E: last audio byte → first audio out                                          | `end-of-call-report turnLatencies[i]`                                                               |

---

## Source of Truth by Metric

| What you want            | Use                   | Why                                                                                       |
| ------------------------ | --------------------- | ----------------------------------------------------------------------------------------- |
| True user-perceived E2E  | `vapi_turn_ms`        | Includes endpointing we can't see                                                         |
| Endpointing latency      | `vapi_endpointing_ms` | Only Vapi can measure this                                                                |
| Transcription latency    | `vapi_stt_ms`         | Authoritative; our old `stt_ms` blurred endpointing + STT + dispatch                     |
| Our server processing    | `vapi_llm_ms`         | Measures LLM request → first token at Vapi's network boundary                            |
| LLM TTFT (granular)      | `llm_ttft_ms`         | More precise than `vapi_llm_ms` — excludes DB prep and filler token overhead              |
| TTS TTFT                 | `vapi_tts_ms`         | Direct                                                                                    |
| Filler effect            | `filler_injected`     | Only we know; Vapi doesn't distinguish                                                    |

---

## Why no gap between filler and the rest of speech?

The most likely explanation: **ElevenLabs is buffering**. Vapi sends `"Mm-hm, "` (8 chars) to ElevenLabs immediately, but ElevenLabs' streaming TTS internally waits for enough context before generating audio — `chunkPlan.minCharacters=1` controls Vapi's chunk size, not ElevenLabs' internal synthesis buffer. So:

- ElevenLabs holds `"Mm-hm, "` until more text arrives ~1.5s later
- Then generates audio for the full concatenated response in one pass
- Result: no gap, but `voiceLatency = 1.7s` because TTS started at the filler token but audio didn't come back until LLM was done

**If this is what's happening, the filler isn't actually helping perceptual latency at all** — ElevenLabs is just absorbing it into the full response.

---

## Notes on STT Latency

STT runs in streaming mode during the user's speech, so by the time the user finishes speaking, a large portion of the transcription is already done. `vapi_stt_ms` captures the **residual finalization time** — the delay to produce the final transcript after the user stops — rather than the full transcription duration from scratch.

Endpointing itself operates on top of the transcription output, so it is genuinely sequential and additive after STT.

---

## Dashboard

**Voice Latency** dashboard in Logfire (project: `research-agent`, slug: `voice-latency`), 6h window, 5-minute buckets:

| Panel | Query source | Shows |
| ----- | ------------ | ----- |
| E2E Latency (median & p95) | `turn_latency` spans | Removed — `e2e_ms` retired; use `vapi_turn_ms` from panel 3 |
| Sub-segment Waterfall | `vapi_turn_latency` spans | `vapi_endpointing_ms`, `vapi_stt_ms`, `vapi_llm_ms`, `vapi_tts_ms` per bucket |
| Our E2E vs Vapi (per call) | `call_ended` spans | `vapi_turn_ms_avg`, `vapi_endpointing_ms_avg`, `vapi_llm_ms_avg`, `vapi_tts_ms_avg` |
| Filler Injection Rate & Effect | `turn_latency` spans | Turn count + avg/p95 `e2e_ms` split by `filler_injected` |
