Ready for review
Select text to add comments on the plan
Latency Measurement Plan
Context
The voice agent already has partial latency instrumentation scattered across several logfire spans, but it's incomplete and inconsistent:

turn_latency span logs stt_ms, llm_ttft_ms, tts_ttft_ms, total_ms — but total_ms is not clearly named as the primary E2E metric, and the waterfall sub-segments (all measured from t0) aren't explicit
Vapi's end-of-call data is logged only as per-call averages — per-turn breakdown is discarded
The Turn DB table has latency_ms and action columns that are never written (existing bug)
No single dashboard shows all metrics together
The goal: one clear E2E metric, explicit waterfall sub-segments from t0, per-turn Vapi data, and a Logfire dashboard showing everything together.

Definitions
t0 = when our server receives speech-update(user/stopped) = Vapi's VAD already fired, NOT the user's actual last audio byte.

Field	Meaning	How computed
e2e_ms	Our primary metric: endpointing → first assistant audio	user_stopped_at → speech-update(assistant/started)
stt_ms	Endpointing → transcript at LLM endpoint	user_stopped_at → LLM request arrival
llm_ttft_ms	LLM internal TTFT	time.perf_counter() inside stream_tokens()
tts_ttft_ms	First LLM token → first audio out	e2e_ms - stt_ms - llm_ttft_ms (residual)
filler_injected	Whether an instant filler prefix was streamed before LLM tokens	stashed from stream_tokens()
Vapi endpointingLatency	Last audio byte → VAD fires (pre-t0 offset, per turn)	end-of-call-report turnLatencies[i]
Vapi turnLatency	Vapi's E2E ≈ our e2e_ms (same window, different clock)	end-of-call-report turnLatencies[i]
true_e2e_ms	True user-perceived: last audio byte → first audio out	vapi_endpointing_ms + e2e_ms (computable at call-end)
Important: e2e_ms does NOT include the endpointing decision delay. The time from the user's actual last word to our t0 is Vapi's endpointingLatency. True user-perceived latency = vapi_endpointing_ms + e2e_ms.

Filler effect: When fillers are enabled, speech-update(assistant/started) fires for the filler first, so e2e_ms reflects filler-to-audio latency before the main LLM line. e2e_ms must be read alongside filler_injected to be meaningful.

**Why no gap between filler and the rest of speech?**

The most likely explanation: **ElevenLabs is buffering**. Vapi sends "Mm-hm, " (8 chars) to ElevenLabs immediately, but ElevenLabs' streaming TTS internally waits for enough context before generating audio — `chunkPlan.minCharacters=1` controls Vapi's chunk size, not ElevenLabs' internal synthesis buffer. So:

- ElevenLabs holds "Mm-hm, " until more text arrives ~1.5s later
- Then generates audio for the full concatenated response in one pass
- Result: no gap, but `voiceLatency = 1.7s` because TTS started at the filler token but audio didn't come back until LLM was done

**If this is what's happening, the filler isn't actually helping perceptual latency at all** — ElevenLabs is just absorbing it into the full response.



Currently

- STT begins onlny after VAD
- STT does not stream. It sends to LLM only after fully done
- Metrics would change once we have these 
- We should also have a ts_ttft (including filler, and excluding filler/"Pure")

Changes

1. voice_agent/server.py — stash filler_injected in _speech_ts (~line 639)

After pipeline.commit() returns in sse(), also stash whether a filler was injected so the speech-update handler can attach it to turn_latency. result already carries llm_latency_ms and ttft_ms; we need filler_injected too.

Add to StreamTurnResult in voice_agent/turn.py:

filler_injected: bool = False
Set it in TurnPipeline.commit() from self._filler_injected (already tracked in stream_tokens() as the local filler variable — just store it on self).

Then in sse() after pipeline.commit():

entry["filler_injected"] = result.filler_injected
2. voice_agent/server.py — speech-update(assistant/started) handler (~line 417)
Current turn_latency log: stt_ms, llm_ttft_ms, tts_ttft_ms, total_ms

Change to:

logfire.info(
    "turn_latency",
    call_id=call_id,
    e2e_ms=turnaround_ms,          # renamed from total_ms — primary metric
    stt_ms=stt_ms,
    llm_ttft_ms=llm_ttft_ms,
    tts_ttft_ms=tts_ttft_ms,
    filler_injected=entry.get("filler_injected", False),   # NEW
)
Drop total_ms, rename to e2e_ms. Drop llm_first_token_from_t0_ms (trivially derived as stt_ms + llm_ttft_ms).

1. voice_agent/server.py — end-of-call-report handler (~line 354)

After computing turn_latencies, add a per-turn log loop before the existing call_ended summary:

for i, t in enumerate(turn_latencies):
    logfire.info(
        "vapi_turn_latency",
        call_id=call_id,
        vapi_turn_index=i,
        vapi_endpointing_ms=round(t.get("endpointingLatency", 0) * 1000),
        vapi_stt_ms=round(t.get("transcriberLatency", 0) * 1000),
        vapi_llm_ms=round(t.get("modelLatency", 0) * 1000),
        vapi_tts_ms=round(t.get("voiceLatency", 0) * 1000),
        vapi_turn_ms=round(t.get("turnLatency", 0) * 1000),
    )
Also add true_e2e_ms_avg to the existing call_ended log — computable once we have both numbers:

# inside the existing vapi_latency dict

"true_e2e_ms_avg": _avg_ms("endpointingLatency") + our_e2e_avg,  # endpointing offset + our measurement
(where our_e2e_avg = average of e2e_ms values stashed in _speech_ts across turns, or skip if unavailable)

Keep the existing averaged call_ended log — don't remove it.

1. voice_agent/turn.py — commit() (~line 210)

Fix pre-existing bug: action and latency_ms exist on the Turn model but are never written. Add them to the interviewer row:

session.add(
    state.Turn(
        call_id=call_id,
        turn_number=next_num,
        speaker="interviewer",
        text=reply.utterance,
        action=reply.action,               # ADD
        latency_ms=self._llm_latency_ms,   # ADD
        tokens_input=...,
        ...
    )
)
No schema change needed — columns already exist in state.Turn.

1. Logfire Dashboard (via MCP)

Create a dashboard named "Voice Latency" with 4 panels:

Panel 1 — E2E Latency (primary metric)

Type: TimeSeriesChart
SQL: select e2e_ms from turn_latency logs, over time
Shows: median and p95 E2E per turn
Panel 2 — Sub-segment Waterfall

Type: TimeSeriesChart (stacked or multi-series)
Series: stt_ms, llm_ttft_ms, tts_ttft_ms from turn_latency logs
Shows: where time is going per turn
Panel 3 — Our measurement vs Vapi

Type: TablePanel or stat
SQL: compare per-call averages from call_ended (vapi_turn_ms_avg) vs computed averages from turn_latency (e2e_ms), segmented by filler_injected
Shows: validation / discrepancy check; vapi_turn_ms ≈ e2e_ms since they measure the same window
Panel 4 — Filler injection rate + effect on latency

SQL: count of filler_injected=true vs total turns from turn_latency logs
Second stat: avg e2e_ms where filler_injected=true vs false — shows how much fillers mask real latency
Files Modified
File	Change
voice_agent/server.py	Rename total_ms→e2e_ms, add filler_injected to turn_latency, stash filler_injected in _speech_ts, add per-turn Vapi loop, add true_e2e_ms_avg to call_ended
voice_agent/turn.py	Add filler_injected to StreamTurnResult; store self._filler_injected on pipeline; write action + latency_ms to Turn row in commit()
Logfire dashboard	Created via MCP mcp__logfire__dashboard_create
No changes to state.py — all new fields already exist in the DB schema.

Verification
Run a call via uv run python scripts/play.py "Notion AI" (REPL — no Vapi webhook, so speech-update won't fire; E2E won't log, but DB writes will be testable)
For full pipeline test, run the server and make a test call via Vapi
In Logfire, query: SELECT * FROM logs WHERE message = 'turn_latency' LIMIT 10 — verify e2e_ms, filler_injected appear
Query: SELECT * FROM logs WHERE message = 'vapi_turn_latency' LIMIT 10 — verify per-turn Vapi rows
Open the dashboard and confirm all 4 panels render
Add Comment

**Source of truth by metric**


| What you want           | Use                                                                                                 | Why                                                                                              |
| ----------------------- | --------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| True user-perceived E2E | Vapi `vapi_turn_ms (Subtract the filler time to be more accurate)`                                  | Includes endpointing we can't see                                                                |
| Endpointing latency     | Vapi `vapi_endpointing_ms`                                                                          | Only Vapi can measure this                                                                       |
| Transcription/dispatch  | Vapi `vapi_stt_ms` `(Our stt is measured from speech_end, so it also incldues part of endpointing)` | Their `transcriberLatency` is the authoritative bucket even if measurement quirky                |
| Our server processing   | Vapi `vapi_llm_ms (Includes request prep, db calls, and llm ttft)`                                  | Measures LLM request → first token at Vapi's network boundary (this is usually our filler token) |
| LLM TTFT (granular)     | Our `llm_ttft_ms`                                                                                   | More precise for actual LLM time taken                                                           |
| TTS TTFT                | Vapi `vapi_tts_ms`                                                                                  | Direct                                                                                           |
| Filler effect           | Our `filler_injected`                                                                               | Only we know; Vapi doesn't distinguish                                                           |


STT runs in streaming mode during the user's speech, so by the time the user finishes speaking, a large portion of the transcription is already done . The transcriberLatencyAverage metric likely captures the residual finalization time — the delay to produce the final transcript after the user stops — rather than the full transcription duration from scratch. (Our  custom stt measures total stt - from when user stops to stt done (so includes some of endpointing)

Endpointing itself is text-based (or audio-text fusion) and operates on top of the transcription output, so it is genuinely sequential and additive after STT .

So the Vapi transciber latency is the time to finalize stt transcript once endpoint is detected