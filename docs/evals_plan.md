# Evals Plan: Voice Agent

## Context

The current 3-tier eval system (pydantic_evals, 18 Tier 1 cases, 6 Tier 2, 6 Tier 3 personas) is solid but has four compounding problems:

1. **REPL path divergence silently invalidates eval results.** Evals call `run_interviewer()` with a live session; production calls `TurnPipeline.stream_tokens()` with `deps.session=None`. They exercise different code paths.
2. **No online evals.** Logfire captures rich per-turn telemetry, but no automatic post-call scoring fires on production traffic.
3. **No versioning / score history.** Can't correlate "which dataset version + git SHA produced which scores."
4. **Tier 3 is too slow and expensive to be prod-grade.** Full simulation with a Sonnet respondent per turn is ~40–80 LLM calls per persona, non-deterministic, and runs minutes. Not suitable for CI.

---

## Platform Choice: Logfire Evals + Live Evals

Logfire/pydantic-evals provides exactly the right infrastructure:

- **Offline evals → Logfire Evals UI** (`/evals`): `logfire.configure()` + `dataset.evaluate()` sends every eval run (per-case scores, aggregates, evaluator traces) to the Logfire Evals: Datasets & Experiments page automatically. Datasets can also be hosted in Logfire via `LogfireAPIClient` and fetched at eval time. Use `send_to_logfire='if-token-present'` so local runs are free and CI with `LOGFIRE_TOKEN` set lands in the UI.

- **Online evals → Logfire Live Evals** (`/live-evals`): `pydantic_evals.online.evaluate` decorator (or `OnlineEvaluation` capability on PydanticAI agents) runs evaluators in the background on live traffic and emits `gen_ai.evaluation.result` OTel events. These show up on the Live Monitoring page with sparklines per evaluator per target over time. **This replaces the hand-rolled `online_eval.py` approach** — no new file needed, just decorate the right function.

Both are in the existing `pydantic-evals` + `logfire` stack already installed.

---

## Priority 1: Fix the REPL Path Divergence

**This is the most critical fix — it currently invalidates Tier 1 and Tier 3 results.**

### The Exact Divergence

| | Production | Evals today |
|---|---|---|
| Entry | `TurnPipeline.stream_tokens()` | `run_interviewer()` in `interviewer.py` |
| `deps.session` | Always `None` | Always non-None |
| DB prep | `prepare_interviewer_turn_concurrent()` (async parallel) | `prepare_interviewer_turn()` (sync sequential) |
| vapi_messages | Always from Vapi webhook | `_db_messages_fallback()` if missing |
| Session lifetime | Closed before LLM call | Open during entire LLM call |

`run_interviewer()`, `prepare_interviewer_turn()`, and `_db_messages_fallback()` in `interviewer.py` are never called in production. Evals test a dead code path.

### Fix: Route Evals Through `run_speech_turn()`

`run_speech_turn()` in `turn.py` already wraps `TurnPipeline` exactly as production does. It takes `(engine, call_id, vapi_messages)`, runs `stream_tokens()` + `commit()`, and returns a dict with `action`, `message`, `reasoning`, `should_run_analyst`, etc.

**`evals/test_interviewer.py`** — replace `run_interviewer_on_case()`:

```python
from voice_agent.turn import run_speech_turn

async def run_interviewer_on_case(inputs: InterviewerCaseInputs) -> InterviewerOutput:
    engine, call_id, _ = _seed_engine(inputs)
    vapi_messages = [
        {"role": "assistant" if t.speaker == "interviewer" else "user", "content": t.text}
        for t in inputs.prior_turns
    ]
    vapi_messages.append({"role": "user", "content": inputs.last_respondent})
    result = await run_speech_turn(engine, call_id, vapi_messages=vapi_messages)
    return InterviewerOutput(
        utterance=result["message"],
        action=result["action"],
        reasoning=result["reasoning"],
    )
```

Remove the `run_interviewer` import and the `session_scope` block. `_seed_engine()` is unchanged.

**`evals/test_trajectories.py`** — inner loop "Interviewer turn" block (lines 157–179):

```python
# Remove: with state.session_scope(engine) as s: block wrapping run_interviewer()
# Remove: manual session.add(state.Turn(...)) for the interviewer turn — commit() does it
vapi_messages = [
    {"role": "assistant" if t["speaker"] == "interviewer" else "user", "content": t["text"]}
    for t in history
]
result = await run_speech_turn(engine, call_id, vapi_messages=vapi_messages)
# out.action → result["action"], out.utterance → result["message"]
```

`commit()` now writes the interviewer Turn row — remove the manual `session.add(Turn(...))` for the interviewer inside the loop.

### Post-Fix Cleanup (Not Blocking)

`run_interviewer()`, `prepare_interviewer_turn()`, `_db_messages_fallback()` in `interviewer.py` will have no callers except the REPL. Add `# REPL-only — not used in production or evals` on each. Leave for a future REPL consolidation.

---

## Priority 2: Offline Evals → Logfire Evals UI

### Wire Logfire Evals into the existing test files

Change `init_tracing(send_to_logfire=False)` → `logfire.configure(send_to_logfire='if-token-present')` at the top of each eval test file. When `LOGFIRE_TOKEN` is set (CI), every `dataset.evaluate()` call sends results to the Logfire Evals page automatically. Locally without a token, evals run as before with no changes.

No structural changes to the eval framework — `pydantic_evals.Dataset.evaluate()` already sends traces when Logfire is configured.

### Add Dataset Name to Each Dataset

```python
dataset = Dataset[InterviewerCaseInputs, InterviewerOutput, None](
    name="tier1-interviewer-decisions",  # shows up as dataset name in Logfire UI
    cases=cases,
    evaluators=[ActionMatches(), SingleQuestion(), ...],
)
```

This lets you track score history per named dataset across git SHAs.

### Tier 1 — New YAML Cases (append to `evals/datasets/interviewer_turns.yaml`)

Current 18 cases cover probe (5), scripted (5), clarify (3), wrap_up (3), off_topic (2). Missing:

| Gap | Case name | Expected action |
|---|---|---|
| `[silence]` → clarify | `true_silence_clarify` | clarify |
| Pure filler ("um") → clarify | `thinking_filler_clarify` | clarify |
| ROI quantification → probe | `roi_quantification_probe` | probe |
| Named competitor mention → probe | `competitor_mention_probe` | probe |
| Expansion signal → probe | `expansion_signal_probe` | probe |
| Low rating (4/10) → probe | `low_rating_probe` | probe |
| Scripted topic already organically covered → skip | `skip_scripted_already_covered` | skip_scripted |

**+7 cases → 25 total.** No Python changes; append YAML entries.

### Tier 2 — New Analyst Cases (append to `evals/datasets/analyst_probes.yaml`)

Current 6 cases: contradictions (3), specificity (1), advocacy gap (1), clean (1). Missing:

| Gap | Case name | Expected probe |
|---|---|---|
| Low adoption + IT security block | `red_flag_low_adoption` | priority-1 probe |
| Strong word-of-mouth + expansion | `pmf_expansion_signal` | priority-1 or priority-2 |

**+2 cases → 8 total.** No Python changes; append YAML entries.

### Dataset Versioning via YAML Frontmatter

Add to each `evals/datasets/*.yaml`:
```yaml
version: "2026-05-05.1"
description: "Added 7 Tier 1 cases: silence, investor triggers, skip_scripted"
cases:
  - ...
```

Add to `evals/cases.py`:
```python
def get_dataset_version(path: str | Path) -> str:
    raw = yaml.safe_load(Path(path).read_text())
    return raw.get("version", "unversioned")
```

Pass `dataset_version` as metadata on the Dataset or as a span attribute — shows up in the Logfire Evals UI next to each run.

---

## Priority 3: Tier 3 — Faster, Prod-Grade E2E Evals

### Problem with Current Tier 3

Tier 3 runs a Sonnet respondent per turn — 20 turns × 6 personas = 120 Sonnet calls + 120 Haiku calls + 6 Opus synthesis = ~$1.50/run, ~5 minutes, non-deterministic. That's not CI-able and score variance is high.

### Solution: Three-Layer Tier 3 Strategy

**Layer A — Replay Evals (new, fast, deterministic, CI-safe)**

Store canned full transcripts as YAML. The respondent lines are fixed; only the interviewer runs live. Tests multi-turn state effects that Tier 1 can't cover: `covered_subtopics` accumulation, probe staleness, scripted cursor advancement, barge-in reconciliation.

**New file: `evals/datasets/replay_transcripts.yaml`**
```yaml
version: "2026-05-05.1"
transcripts:
  - name: power_user_arc
    scripted_questions: [...]
    turns:
      - speaker: respondent
        text: "We use it for every external call, about 30 a week..."
        expected_actions: [scripted, probe]
      - speaker: respondent
        text: "Honestly the Slack integration is the biggest thing missing..."
        expected_actions: [probe, scripted]
```

`expected_actions` is a set — any listed action passes. Absorbs LLM variance while catching regressions.

**New file: `evals/test_replay.py`**

```python
@pytest.mark.replay
async def test_replay_transcripts():
    for transcript in load_replay_transcripts():
        engine, call_id = seed_replay(transcript)
        history = []
        for turn in transcript.turns:
            history.append({"role": "user", "content": turn.text})
            result = await run_speech_turn(engine, call_id, vapi_messages=history)
            assert result["action"] in turn.expected_actions
            history.append({"role": "assistant", "content": result["message"]})
```

**Runtime:** 3 transcripts × 10 turns = 30 Haiku calls ≈ 20s. Mark `@pytest.mark.replay`, run in CI fast path.

**Layer B — Focused Simulation (existing Tier 3, scoped down)**

Keep the full simulation but reduce scope: run only the **2 persona-specific behaviors** that require live simulation — `contradictory` (analyst contradiction detection) and `off_topic_rambler` (redirect action). Cap at 12 turns. Skip synthesis for speed.

These two are the only personas testing emergent multi-agent behavior (analyst ↔ interviewer interaction) that replay can't cover. The other 4 personas (power_user, skeptical_buyer, ai_skeptic, churn_risk) become replay transcripts instead.

Result: **2 Sonnet personas × 12 turns = 24 Sonnet calls** vs current 120. ~$0.25/run, ~45s. Mark `@pytest.mark.slow` — run in nightly CI or pre-deploy only.

**Layer C — Synthesis Quality (separate, infrequent)**

Pull the `ReportQuality` LLMJudge into its own test file `evals/test_synthesis.py`. Run against a fixed set of stored transcripts (the replay transcripts) rather than re-simulating. This makes synthesis evals reproducible and separable from trajectory evals.

### After REPL Path Fix

Replace `run_interviewer()` calls in the remaining simulation loop (the 2 persona sim) with `run_speech_turn()`. Remove manual Turn row inserts — `commit()` handles them.

---

## Priority 4: Online Evals → Logfire Live Evals

### Mechanism

`pydantic_evals.online` provides two hooks:

1. **`@evaluate` decorator** — wraps any async function; evaluators run in background after each call
2. **`OnlineEvaluation` capability** — attaches evaluators directly to a PydanticAI agent

Results are emitted as `gen_ai.evaluation.result` OTel events and show up on the Live Evals page with sparklines over time. No new file, no new DB table.

### What to Instrument

**`TurnPipeline.stream_tokens()` in `turn.py`** — wrap the commit result with `@evaluate` or emit inline. The natural place is after `commit()` returns, using the existing `logfire.info("interviewer_stream_done", ...)` span as the hook point.

Evaluators to attach:

```python
from pydantic_evals.online import OnlineEvalConfig

online_eval_config = OnlineEvalConfig(emit_otel_events=True)

@dataclass
class SingleQuestionOnline(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return ctx.output.get("utterance", "").count("?") <= 1

@dataclass
class ActionIsValid(Evaluator):
    VALID_ACTIONS = {"probe", "scripted", "clarify", "off_topic", "wrap_up", "skip_scripted"}
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return ctx.output.get("action") in self.VALID_ACTIONS

@dataclass
class FillerRate(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> float:
        # Returns 0.0 or 1.0 per turn; Live Evals shows the rolling average
        return 1.0 if ctx.output.get("filler_injected") else 0.0
```

**Post-call (analyst output)** — attach evaluators to `run_analyst_safely` in `analyst.py`:

```python
@dataclass
class ProbesGenerated(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return bool(ctx.output and ctx.output.new_probes)
```

### Configuration

Initialize at server startup in `voice_agent/server.py` lifespan (or `tracing.py`):

```python
from pydantic_evals.online import configure as configure_online_evals
configure_online_evals(
    emit_otel_events=True,
    default_sample_rate=1.0,  # score every production turn
)
```

`emit_otel_events=True` is the only thing needed — Logfire's OTel exporter picks up `gen_ai.evaluation.result` events automatically.

### Live Evals Page

After a few calls, the `/live-evals` page shows:
- `SingleQuestionOnline` pass rate over time per interviewer turn
- `FillerRate` rolling average
- `ProbesGenerated` rate per analyst run

Alerts can be set in Logfire when pass rates drop below threshold.

---

## Implementation Sequence

| Step | File(s) | What | Blocker |
|---|---|---|---|
| 1 | `evals/test_interviewer.py` | Use `run_speech_turn`, remove session block | **Do first** |
| 2 | `evals/test_trajectories.py` | Use `run_speech_turn`, remove manual Turn inserts | After Step 1 |
| 3 | `evals/test_*.py` | `logfire.configure(send_to_logfire='if-token-present')`, add `Dataset(name=...)` | None |
| 4 | `evals/datasets/interviewer_turns.yaml` | +7 Tier 1 cases | After Step 1 (verify they pass) |
| 5 | `evals/datasets/analyst_probes.yaml` | +2 Tier 2 cases | None |
| 6 | `evals/cases.py` | Add `get_dataset_version()` | None |
| 6 | `evals/datasets/*.yaml` | Add `version:` + `description:` fields | None |
| 7 | `evals/datasets/replay_transcripts.yaml` | 3 canned transcripts | None |
| 7 | `evals/test_replay.py` | Replay eval harness | After Step 1 |
| 8 | `evals/test_trajectories.py` | Scope to 2 simulation personas, remove other 4 | After Step 2 |
| 8 | `evals/test_synthesis.py` | Pull out synthesis eval, replay-transcript driven | After Step 7 |
| 9 | `voice_agent/turn.py` + `analyst.py` | Wire online eval decorators | None |
| 9 | `voice_agent/server.py` or `tracing.py` | `configure_online_evals(emit_otel_events=True)` | None |

---

## Key Design Decisions

**Why `run_speech_turn` and not a new `run_interviewer_for_eval()`?**
Any wrapper that partially replicates `TurnPipeline` will drift again. Calling `run_speech_turn` directly means evals and production share one code path — the divergence problem can't recur.

**Why Logfire Live Evals instead of a custom `online_eval.py`?**
`pydantic_evals.online` is already installed, emits OTel events natively, shows up in the Live Evals page with no extra code, and is maintained by Pydantic. A hand-rolled post-call scorer would be more work, harder to visualize, and would drift from the framework.

**Why Logfire Evals UI instead of just logging aggregate scores?**
`logfire.configure()` + `dataset.evaluate()` sends full per-case traces (each evaluator's score + rationale for LLMJudge) to the UI, not just aggregates. You can drill into which specific case regressed, compare runs side-by-side, and manage datasets from the UI. Score-only logging loses this.

**Why scoped simulation (2 personas) instead of full 6-persona simulation?**
Only 2 personas (`contradictory`, `off_topic_rambler`) test emergent multi-agent behavior that replay can't cover — analyst contradiction detection and off_topic redirect. The other 4 test interviewer output quality, which is better tested deterministically via replay. Cutting from 6 to 2 personas + capping at 12 turns reduces simulation cost by ~80%.

**Why replay transcripts instead of always running the simulator for Tier 3?**
Simulator is non-deterministic (LLM-driven) and slow. Replay transcripts fix the respondent text, making multi-turn evals deterministic and cheap (~20s in CI). They test the state machine (covered_subtopics, probe staleness, scripted cursor) which single-turn Tier 1 can't catch, without paying for a live Sonnet respondent.

---

## Critical Files

- [evals/test_interviewer.py](evals/test_interviewer.py) — Step 1 (REPL fix, most critical)
- [evals/test_trajectories.py](evals/test_trajectories.py) — Steps 2 & 8 (REPL fix + scope reduction)
- [evals/cases.py](evals/cases.py) — Step 6 (versioning)
- [evals/datasets/interviewer_turns.yaml](evals/datasets/interviewer_turns.yaml) — Step 4
- [evals/datasets/analyst_probes.yaml](evals/datasets/analyst_probes.yaml) — Step 5
- [evals/datasets/replay_transcripts.yaml](evals/datasets/replay_transcripts.yaml) — Step 7 (new file)
- [evals/test_replay.py](evals/test_replay.py) — Step 7 (new file)
- [evals/test_synthesis.py](evals/test_synthesis.py) — Step 8 (new file)
- [voice_agent/turn.py](voice_agent/turn.py) — Step 9 (online eval decorators)
- [voice_agent/agents/analyst.py](voice_agent/agents/analyst.py) — Step 9 (online eval decorators)
- [voice_agent/server.py](voice_agent/server.py) or [voice_agent/tracing.py](voice_agent/tracing.py) — Step 9 (configure online evals)

## Verification

- After Step 1: `uv run pytest evals/test_interviewer.py -v` — all 18 cases pass
- After Step 4: `uv run pytest evals/test_interviewer.py -v` — 25 cases pass
- After Step 7: `uv run pytest evals/test_replay.py -v -m replay` — 3 transcripts × ~10 turns in ~20s
- After Step 8: `uv run pytest evals/test_trajectories.py -v -s -m slow` — 2 personas only, ~45s
- After Step 9: trigger a test call → check Logfire `/live-evals` page for `gen_ai.evaluation.result` events
- Offline scores: after CI run with `LOGFIRE_TOKEN` set → check `/evals` page for tier1/tier2 datasets
