# Evals Gap Analysis & Plan

Current state: 35 Tier 1 cases, 14 Tier 2 cases, 6 replay transcripts, 3 simulation personas. All evals route through `run_speech_turn()` (production path). This document tracks what's not yet covered and what to build next.

---

## What's Already Covered

| Behavior | Where tested |
|---|---|
| Action selection (probe/scripted/clarify/wrap_up/off_topic/skip_scripted) | Tier 1 — 35 single-turn cases |
| Utterance warmth, single question, non-leading, topical relevance | Tier 1 — LLMJudge evaluators |
| Analyst probe quality: specificity, non-leading, priority calibration | Tier 2 — 14 cases |
| Analyst incremental behavior (no re-probe from prior snapshot) | Tier 2 — snapshot-seeded cases |
| Scripted cursor advancement over a full arc | Replay — all 6 transcripts |
| `skip_scripted` on organic coverage | Replay — `skip_scripted_organic_coverage` |
| Silence → "Still there?" (single occurrence) | Replay — `non_happy_path_silence_vague` |
| Vague answer → clarify | Replay — `non_happy_path_silence_vague` |
| Loop guard: no consecutive "Still there?" | Replay — `loop_guard_no_repeat` evaluator |
| `WrapUpOnlyAfterAllScripted` — no premature exit | Replay — evaluator on all 6 transcripts |
| Analyst catches contradiction + interviewer probes it | Simulation — `contradictory` persona |
| Off-topic redirect | Simulation — `off_topic_rambler` persona |
| Extended silence → wrap_up after 3× "Still there?" | Simulation — `silent_respondent` persona |
| Probe specificity and stale-probe bridging | Simulation — `ProbesAreSpecific`, `StaleProbesBridged` |

---

## Gaps by Category

### 1. Stale probe bridging ("Earlier you mentioned…")

**What the prompt says:** When a probe is 3–8 turns old (`turns_ago 3–8`), bridge with "Earlier you mentioned X…" before asking.

**What's tested:** `StaleProbesBridged` in the simulation eval scores this, but only when stale probes actually appear in the trajectory — and even then only at a 75% threshold. No deterministic replay case verifies that phrasing.

**Gap:** No replay transcript seeds a probe at turn N and then forces the interviewer to use it at turn N+4, verifying the utterance contains a bridging phrase.

**Fix — new replay transcript: `stale_probe_bridging`**
- Turn 1–2: analyst generates a probe (seed in DB directly, or verify via analyst output)
- Turns 3–5: three scripted turns that don't trigger the probe
- Turn 6: respondent answer that has no new signal — interviewer should reach for the now-stale probe and bridge with "Earlier you mentioned…"
- Add a new evaluator `BridgingPhrasePresent` that checks for "earlier" / "you mentioned" / "going back" in utterances where `action=probe` and the probe is 3+ turns old.

---

### 2. Silence handling — full 3× sequence

**What the prompt says:** silence → "Still there?" → silence → "Still there?" → silence → `wrap_up`.

**What's tested:** `NoStillThereLoop` (in replay) catches *consecutive* "Still there?" turns. `HandledSilenceGracefully` (in simulation) checks that at least one "Still there?" and one "Take your time." appeared and that `wrap_up` eventually fired.

**Gap:** No replay transcript walks through the full 3-silence chain to assert `wrap_up` fires at exactly the right point. The simulation test covers this but non-deterministically.

**Fix — new replay transcript: `extended_silence_wrap_up`**
```yaml
turns:
  - text: "We use it for customer calls."   # substantive turn
    expected_actions: [scripted, probe]
  - text: "[silence]"                        # → "Still there?"
    expected_actions: [clarify]
  - text: "[silence]"                        # → "Still there?" again (count=2)
    expected_actions: [clarify]
  - text: "[silence]"                        # → wrap_up (count=3)
    expected_actions: [wrap_up]
```
Add evaluator `WrapUpAfterThirdSilence` — checks `wrap_up` fires by turn 4 of this transcript.

Also add a case for the `um` → "Take your time." → silence → "Still there?" chain (not "Take your time." twice in a row).

---

### 3. `skip_scripted` — multiple consecutive skips

**What's tested:** `skip_scripted_organic_coverage` has two `skip_scripted` turns but they're separated by substantive content.

**Gap:** No test for the pathological case where the respondent's first answer organically covers Q2 and Q3 simultaneously, requiring two consecutive `skip_scripted` actions before landing on an unanswered question.

**Fix — new replay transcript: `multi_skip_scripted`**
- Q1–Q5 loaded; respondent's turn 1 covers Q1, Q2, Q3 all at once
- Expected actions on turn 1: `[skip_scripted]`
- Expected actions on turn 2: `[skip_scripted]` (skipping Q3 too)
- Expected actions on turn 3: `[scripted]` (Q4 genuinely unanswered)

---

### 4. Early wrap-up when respondent is disengaged

**What the prompt says (step 8):** Wrap up when scripted remaining is 0 *and no important threads remain*. But the prompt also implies wrapping up on sustained disengagement (repeated silence, off-topic with no return).

**What's tested:** `WrapUpOnlyAfterAllScripted` prevents *premature* wrap-up. But there's no test for the inverse: does the interviewer *actually* wrap up promptly on genuine disengagement rather than grinding through all scripted questions robotically?

**Gap:** A transcript where the respondent gives 3 consecutive one-word or dismissive answers after a natural conversational endpoint — does the interviewer wrap up, or does it keep asking scripted questions?

**Fix — new replay transcript: `disengaged_early_exit`**
- Turn 1: good answer
- Turn 2: "Not really." (vague → clarify)
- Turn 3: "I dunno." (vague again)
- Turn 4: "Fine I guess." (third low-signal answer)
- Expected actions on turn 4 or 5: `[wrap_up]` — interviewer should read disengagement and close gracefully rather than continuing to ask scripted questions

This requires a new evaluator or a looser `expected_actions` definition; the key is asserting `wrap_up` fires before all scripted are done, which the current `WrapUpOnlyAfterAllScripted` evaluator would *fail* — so we'd need a `DisengagementWrapsUp` evaluator that passes on this transcript only.

---

### 5. Context drift / topic already covered in COVERED_SUBTOPICS

**What the prompt says (step 2):** Before choosing any action, check COVERED_SUBTOPICS. If the specific subtopic you're about to ask about was already addressed, skip it.

**What's tested:** `skip_scripted_organic_coverage` tests this at the scripted-question level. But there's no test for the analyst generating a probe about a topic that's already in `COVERED_SUBTOPICS` — does the interviewer skip it rather than re-ask?

**Gap:** No replay or Tier 1 case seeds a probe whose topic is already in `covered_subtopics` and asserts the probe is skipped (either `scripted` action or a different probe is used).

**Fix — new Tier 1 case: `skip_probe_already_covered`**
```yaml
covered_subtopics:
  - "Slack integration missing feature"
probes:
  - question: "You mentioned Slack — is that integration missing for the whole team or just you?"
    priority: 2
prior_turns: [...]
last_respondent: "Yeah so that's basically it."
expected_action: scripted  # probe topic is covered; should move to scripted
```

---

### 6. Re-ask / rabbit hole prevention

**What the prompt says:** "Never re-ask what the respondent has already answered." Hard rule.

**What's tested:** `loop_guard_no_repeat` checks for consecutive silence handlers. But no eval checks whether the model re-asks a *content question* (e.g., asks about Slack integration twice in the same transcript).

**Gap:** No multi-turn test where the same topic comes up in both a scripted question and a probe, verifying only one is asked.

**Fix — new replay transcript: `no_topic_repeat`**
- Turn 1: respondent mentions Slack integration gap organically
- Turn 2: interviewer asks scripted Q3 ("What frustrated you?") — respondent restates Slack
- Turn 3: analyst probe seeded: "How often do you manually copy-paste summaries?"
- Expected: interviewer should NOT re-ask about Slack integration; should advance to next scripted or use a different probe
- Add evaluator `NoTopicRepeat` — checks whether the same named entity (e.g. "Slack") appears in both an interviewer question and the subsequent turn where it was already answered, and flags if the interviewer re-raises it anyway.

---

### 7. Depth without breadth trade-off (probe at correct depth, don't skip scripted)

**What the prompt says:** Priority 1 and 2 probes beat scripted. Priority 3 yields to scripted.

**What's tested:** Tier 1 has single-turn cases for probe vs. scripted priority. But no multi-turn test verifies that a Priority 3 probe doesn't cause the interviewer to skip scripted questions it should be asking.

**Gap:** No replay transcript that seeds a Priority 3 probe and verifies the interviewer still asks the next scripted question rather than burning the turn on the low-priority probe.

**Fix — new replay transcript: `priority3_yields_to_scripted`**
- Seed a Priority 3 probe in DB
- Turn 1: substantive answer with no urgent signals
- Expected action: `[scripted]` — Priority 3 should yield
- Turn 2: scripted done, now Priority 3 probe is the only option
- Expected action: `[probe]`

---

### 8. Wrap-up completeness — all scripted + key threads before closing

**What's tested:** `WrapUpOnlyAfterAllScripted` checks wrap_up doesn't fire before all scripted are done. `CoveredAllScripted` in simulation checks coverage at end. But neither checks whether important *open threads* (pending Priority 1 or 2 probes) were addressed before wrap_up.

**Gap:** No test for: interviewer fires `wrap_up` while a Priority 1 probe is still pending — this is a quality failure.

**Fix — new Tier 1 case: `no_wrap_up_with_p1_probe_pending`**
```yaml
scripted_remaining: 0
probes:
  - question: "You mentioned evaluating Fireflies — what specifically prompted that?"
    priority: 1
last_respondent: "That pretty much covers it from my end."
expected_action: probe  # must probe the P1 signal before wrapping
```

---

### 9. Structured logging / observability gap (not an eval — a production monitoring gap)

**What's missing:** There's no structured per-turn log event in `TurnPipeline.commit()` that records `action`, `utterance`, `filler_injected`, `probe_id_used` as queryable fields in Logfire. All the state machine signal is in spans but not as a discrete event you can aggregate across calls.

**Fix:** Add one `logfire.info("turn_committed", call_id=call_id, turn_number=..., action=..., filler_injected=..., probe_id_used=..., utterance_chars=len(utterance))` in `commit()`. Lets you query: action distribution, how often probes are stale when asked, filler injection rate, silence handler rate — all from real calls.

---

## Implementation Priority

| Priority | Gap | Type | Effort |
|---|---|---|---|
| 1 | Stale probe bridging ("Earlier you mentioned…") | New replay transcript + evaluator | Medium |
| 1 | Full 3× silence → wrap_up chain | New replay transcript + evaluator | Low |
| 2 | Early wrap-up on disengagement | New replay transcript + new evaluator | Medium |
| 2 | `wrap_up` blocked by pending P1 probe | New Tier 1 case | Low |
| 2 | `skip_probe` when topic in `covered_subtopics` | New Tier 1 case | Low |
| 3 | Multi-`skip_scripted` consecutive | New replay transcript | Low |
| 3 | No topic repeat within a transcript | New replay transcript + evaluator | Medium |
| 3 | Priority 3 probe yields to scripted | New replay transcript | Low |
| 4 | Structured `turn_committed` log in `commit()` | Production code change | Low |

All replay additions are YAML-only unless a new evaluator is needed. All Tier 1 additions are YAML-only. The only code changes are the new evaluators (`BridgingPhrasePresent`, `WrapUpAfterThirdSilence`, `DisengagementWrapsUp`, `NoTopicRepeat`) and the Logfire log line.
