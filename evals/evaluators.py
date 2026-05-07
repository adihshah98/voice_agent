"""Evaluators for interviewer and analyst evals.

Deterministic scorers: ActionMatches, SingleQuestion.
LLM-as-judge scorers: all use Sonnet 4.6 — fast enough for both numeric scores
and binary assertions at eval scale.

All LLMJudge calls include include_reason=True on score/assertion OutputConfig
so the Logfire Evals UI shows *why* each case passed/failed, not just the value.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, EvaluationReason, LLMJudge

from evals.cases import AnalystCaseInputs, InterviewerCaseInputs
from voice_agent.models import AnalysisUpdate, InterviewerOutput


_JUDGE_MODEL = "anthropic:claude-sonnet-4-6"

# Matches the filler + flush token injected by TurnPipeline when TTS is slow,
# e.g. "Mm-hm, <flush /> " or "Got it, <flush /> ".
_FILLER_RE = re.compile(r"^[^<]*<flush\s*/>\s*", re.IGNORECASE)


def _clean_utterance(utterance: str) -> str:
    """Strip the filler-prefix + <flush /> artifact injected by the streaming pipeline.

    The pipeline yields e.g. "Mm-hm, <flush /> " before the real LLM tokens so
    TTS can start immediately. Evaluators should see only the actual spoken text.
    """
    return _FILLER_RE.sub("", utterance).strip()


@dataclass
class ActionMatches(
    Evaluator[InterviewerCaseInputs, InterviewerOutput, None]
):
    """Fraction of action criteria met (0.0–1.0).

    Returns a numeric score so the aggregate is the mean pass rate across all
    runs rather than a hard bool, making it visible as a continuous metric in
    the Logfire Evals UI.

    Scores 1.0 when action matches (and probe_source matches when specified),
    0.0 otherwise.
    """

    def evaluate(
        self,
        ctx: EvaluatorContext[InterviewerCaseInputs, InterviewerOutput, None],
    ) -> float:
        if ctx.expected_output is None:
            return 0.0
        if ctx.output.action != ctx.expected_output.action:
            return 0.0
        if (
            ctx.expected_output.probe_source is not None
            and ctx.output.probe_source != ctx.expected_output.probe_source
        ):
            return 0.0
        return 1.0


@dataclass
class SingleQuestion(
    Evaluator[InterviewerCaseInputs, InterviewerOutput, None]
):
    """Cleaned utterance must contain at most one `?`.

    Filler prefix and <flush /> are stripped first so "Still there?" from a
    clarify case isn't double-counted and filler phrases like "Mm-hm," don't
    skew the count.
    """

    def evaluate(
        self,
        ctx: EvaluatorContext[InterviewerCaseInputs, InterviewerOutput, None],
    ) -> bool:
        return _clean_utterance(ctx.output.utterance).count("?") <= 1


def utterance_warmth_judge() -> LLMJudge:
    """LLMJudge on conversational warmth — numeric 1-5 score."""
    return LLMJudge(
        rubric=(
            "You are grading a single spoken line from a phone-call market "
            "researcher. The line has already had any filler prefix stripped. "
            "Score 1–5 on conversational warmth:\n"
            "  5 = warm, natural, acknowledges what the respondent said\n"
            "  4 = friendly but slightly formal\n"
            "  3 = neutral / businesslike\n"
            "  2 = cold, transactional, or robotic\n"
            "  1 = rude, interrogative, or off-putting\n"
            "Grade only the `utterance` field; ignore `reasoning`. Pass the "
            "line if the score is >= 3."
        ),
        model=_JUDGE_MODEL,
        include_input=False,
        score={"evaluation_name": "utterance_warmth", "include_reason": True},
        assertion=False,
    )


def no_leading_questions_judge() -> LLMJudge:
    """Pass/fail: does the utterance contain a leading question?"""
    return LLMJudge(
        rubric=(
            "Grade whether the interviewer's utterance contains a LEADING "
            "question. A leading question presupposes the answer (e.g. "
            "'So you loved it, right?', 'That was frustrating, wasn't it?') "
            "or funnels the respondent toward a particular view. Pass if the "
            "question is open and non-leading. Fail if it is leading or "
            "obviously biased. If the utterance contains no question (e.g. "
            "pure acknowledgement or wrap-up), pass.\n"
            "The case input contains the prior transcript so you can verify "
            "whether the interviewer is echoing the respondent's own words "
            "(which is fine) versus introducing a framing the respondent never "
            "expressed (which is leading).\n"
            "Pass neutral research patterns: inviting detail on a hedge "
            "('fine I guess'), gentle off-topic redirects with an open "
            "follow-up, and probes that reuse the respondent's own words "
            "without adding unstated emotion or blame."
        ),
        model=_JUDGE_MODEL,
        include_input=True,
        score=False,
        assertion={"evaluation_name": "non_leading", "include_reason": True},
    )


def response_relevance_judge() -> LLMJudge:
    """Pass/fail: is the utterance topically on-point given the conversation context?

    Catches cases where action is correct but the spoken content is off — e.g.
    the model picks `probe` but asks about the wrong thing, or picks `wrap_up`
    but says something that invites more conversation.
    """
    return LLMJudge(
        rubric=(
            "You are grading a single spoken response from an investor-style "
            "B2B SaaS market research interviewer on a 30-minute customer call.\n\n"
            "You are given:\n"
            "  - The full conversation so far (prior_turns + last_respondent)\n"
            "  - The interviewer's chosen action and utterance\n\n"
            "PASS if the utterance is TOPICALLY APPROPRIATE: it addresses what "
            "the respondent just said, follows naturally from the conversation, "
            "and pursues a goal that makes sense at this moment in the interview.\n\n"
            "FAIL if:\n"
            "  - The utterance asks about or references something the respondent "
            "never mentioned and that doesn't flow from the conversation\n"
            "  - The utterance is a wrap-up but the conversation has clear open "
            "threads that haven't been addressed\n"
            "  - The utterance completely ignores a striking thing the respondent "
            "just said (e.g. a low NPS score, a named competitor, a churn signal) "
            "and pivots to an unrelated topic\n"
            "  - For action=probe: the question probes something unrelated to "
            "what the respondent just said or the recent conversation\n"
            "  - For action=clarify: the clarification doesn't address the "
            "specific ambiguity in the respondent's last utterance\n"
            "  - For action=off_topic: the redirect is abrupt or rude rather "
            "than warm and professional\n\n"
            "Do NOT fail on warmth, leading-ness, or whether it's a single "
            "question — those are graded separately. Grade only topical relevance "
            "and conversational fit."
        ),
        model=_JUDGE_MODEL,
        include_input=True,
        score=False,
        assertion={"evaluation_name": "response_relevant", "include_reason": True},
    )


# ---------------------------------------------------------------------------
# Tier 2 — Analyst probe quality evaluators
# ---------------------------------------------------------------------------


def _jaccard(a: str, b: str) -> float:
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa and not wb:
        return 1.0
    return len(wa & wb) / len(wa | wb)


@dataclass
class HasProbes(Evaluator[AnalystCaseInputs, AnalysisUpdate, None]):
    """At least one probe was generated — necessary for other scorers to mean anything."""

    def evaluate(
        self, ctx: EvaluatorContext[AnalystCaseInputs, AnalysisUpdate, None]
    ) -> bool:
        return len(ctx.output.new_probes) >= 1


@dataclass
class NoDuplicateProbes(Evaluator[AnalystCaseInputs, AnalysisUpdate, None]):
    """No two probes share more than `threshold` Jaccard word-overlap.

    Catches cases where the analyst generates two near-identical questions
    phrased slightly differently. Embedding-level dedup would be more precise
    but requires an extra dep; word Jaccard is accurate enough for a prototype.
    """

    threshold: float = 0.60

    def evaluate(
        self, ctx: EvaluatorContext[AnalystCaseInputs, AnalysisUpdate, None]
    ) -> bool:
        probes = ctx.output.new_probes
        for i, a in enumerate(probes):
            for b in probes[i + 1 :]:
                if _jaccard(a.question, b.question) > self.threshold:
                    return False
        return True


@dataclass
class ProbeUrgencyOrdered(Evaluator[AnalystCaseInputs, AnalysisUpdate, None]):
    """Priority-1 probes must all appear before any priority-2 or priority-3 probe.

    The interviewer works through probes in order, so a priority-1 churn signal
    buried after three priority-3 depth questions may never be asked.
    """

    def evaluate(
        self, ctx: EvaluatorContext[AnalystCaseInputs, AnalysisUpdate, None]
    ) -> bool:
        probes = ctx.output.new_probes
        if len(probes) <= 1:
            return True
        seen_lower = False
        for p in probes:
            if p.priority > 1:
                seen_lower = True
            elif seen_lower:
                return False
        return True


@dataclass
class NoReprobeFromSnapshot(Evaluator[AnalystCaseInputs, AnalysisUpdate, None]):
    """For incremental cases: no probe should ask about topics already in the
    prior snapshot's covered_subtopics.

    Uses word-Jaccard overlap against each covered subtopic label. Passes when
    there is no prior snapshot (non-incremental cases are unconstrained).
    """

    threshold: float = 0.50

    def evaluate(
        self, ctx: EvaluatorContext[AnalystCaseInputs, AnalysisUpdate, None]
    ) -> bool:
        snapshot = ctx.inputs.prior_snapshot
        if snapshot is None:
            return True
        covered = snapshot.covered_subtopics
        for probe in ctx.output.new_probes:
            for subtopic in covered:
                if _jaccard(probe.question, subtopic) > self.threshold:
                    return False
        return True


def probes_specific_judge() -> LLMJudge:
    """Score 1–5: do probes reference specifics from the transcript?"""
    return LLMJudge(
        rubric=(
            "You are grading the PROBES produced by a qualitative research "
            "analyst who just read an interview transcript.\n\n"
            "Score 1–5 on SPECIFICITY:\n"
            "  5 = every probe quotes or directly references a specific detail "
            "from the transcript (a named thing, a verbatim phrase, a concrete "
            "event like a date, route, or product name)\n"
            "  4 = most probes are specific; one may be a bit generic\n"
            "  3 = half specific, half could apply to any interview\n"
            "  2 = mostly generic — probes could have been written without "
            "reading this transcript\n"
            "  1 = entirely generic; no transcript specifics referenced\n\n"
            "Judge the `new_probes` list. The transcript is provided as the "
            "input. Pass (score >= 3) if specificity is adequate."
        ),
        model=_JUDGE_MODEL,
        include_input=True,
        score={"evaluation_name": "probes_specific", "include_reason": True},
        assertion=False,
    )


def probes_non_leading_judge() -> LLMJudge:
    """Pass/fail: are all probes open and non-leading?"""
    return LLMJudge(
        rubric=(
            "You are grading the PROBES produced by a qualitative research "
            "analyst after reading an interview transcript.\n\n"
            "A LEADING probe presupposes the answer, adds an unstated emotion, "
            "or funnels the respondent toward a specific view. Examples:\n"
            "  BAD: 'You said you were frustrated — how badly did that hurt you?'\n"
            "  BAD: 'So the alerts are basically useless, right?'\n"
            "  GOOD: 'What was that experience like for you?'\n"
            "  GOOD: 'You mentioned the alerts changed — can you walk me through that?'\n\n"
            "PASS if all probes are open and neutral. FAIL if any probe is "
            "leading, presupposes an answer, or adds judgment the respondent "
            "didn't express. Probes that echo the respondent's own words "
            "without adding blame or emotion are fine."
        ),
        model=_JUDGE_MODEL,
        include_input=True,
        score=False,
        assertion={"evaluation_name": "probes_non_leading", "include_reason": True},
    )


class _TopicCoverageJudgement(BaseModel):
    covered: list[str]
    missing: list[str]
    all_covered: bool


_TOPIC_COVERAGE_SYSTEM = (
    "You are a research QA evaluator checking whether analyst-generated probes "
    "cover a set of required topics.\n\n"
    "A probe COVERS a topic if a respondent answering that probe would plausibly "
    "reveal the information the topic describes. The probe does not need to mention "
    "the topic explicitly — it just needs to be on a path to elicit it.\n\n"
    "Be generous: if there is a reasonable reading of the probe under which it "
    "would surface the topic, count it as covered. Only mark a topic missing if "
    "none of the probes could plausibly lead to that information.\n\n"
    "Return:\n"
    "  covered: list of required topics addressed by at least one probe\n"
    "  missing: list of required topics with no probe that could surface them\n"
    "  all_covered: true iff missing is empty\n\n"
    "If required_topics is empty, return all_covered=true with empty lists."
)


def _get_topic_coverage_agent() -> Agent[None, _TopicCoverageJudgement]:
    return Agent(
        _JUDGE_MODEL,
        output_type=_TopicCoverageJudgement,
        system_prompt=_TOPIC_COVERAGE_SYSTEM,
    )


@dataclass
class CoversExpectedTopics(Evaluator[AnalystCaseInputs, AnalysisUpdate, None]):
    """Pass/fail: do the probes collectively surface every expected_topic?

    Uses a dedicated PydanticAI agent with a focused prompt — just the topic
    list and probe questions, no transcript noise — for reliable semantic matching.
    Cases with no expected_topics pass unconditionally.
    """

    async def evaluate(
        self, ctx: EvaluatorContext[AnalystCaseInputs, AnalysisUpdate, None]
    ) -> dict:
        topics = ctx.inputs.expected_topics
        if not topics:
            return {"covers_expected_topics": True}

        probes = ctx.output.new_probes if ctx.output else []
        probe_lines = "\n".join(
            f"  [{p.priority}] {p.question}" for p in probes
        ) or "  (no probes generated)"

        topic_lines = "\n".join(f"  - {t}" for t in topics)
        user_prompt = (
            f"REQUIRED TOPICS:\n{topic_lines}\n\n"
            f"PROBES GENERATED:\n{probe_lines}"
        )

        result = await _get_topic_coverage_agent().run(user_prompt)
        judgement = result.output

        score = len(judgement.covered) / len(topics)
        reason = (
            f"covered: {judgement.covered}; missing: {judgement.missing}"
            if judgement.missing
            else f"all {len(topics)} topics covered"
        )
        return {"covers_expected_topics": EvaluationReason(value=score, reason=reason)}


def priority_calibrated_judge() -> LLMJudge:
    """Pass/fail: is priority-1 reserved for real contradictions/surprises/red-flags?"""
    return LLMJudge(
        rubric=(
            "You are grading the priority assignments on PROBES produced by a "
            "qualitative research analyst after reading an interview transcript.\n\n"
            "Priority rules:\n"
            "  Priority 1 = reserved for:\n"
            "    - a REAL contradiction (respondent clearly said two conflicting things)\n"
            "    - a MAJOR surprise (unexpected admission that changes interpretation)\n"
            "    - a RED FLAG (explicit churn signal, renewal threat, low adoption with "
            "      stated deadline, alternatives evaluation underway)\n"
            "  Priority 2 = competitive signal, revenue detail, or strong PMF thread "
            "needing clarification.\n"
            "  Priority 3 = nice-to-have depth or context question.\n\n"
            "PASS if ALL of the following are true:\n"
            "  1. Every priority-1 probe targets a genuine contradiction, major surprise, "
            "or red flag that is clearly visible in the transcript.\n"
            "  2. No priority-1 probe is assigned to a routine depth question that does "
            "not involve a contradiction, surprise, or red flag.\n"
            "  3. If the transcript contains NO contradiction, surprise, or red flag, "
            "then NO priority-1 probes should appear — and if that is the case, PASS.\n\n"
            "FAIL if:\n"
            "  - A probe is marked priority 1 when the transcript shows no real "
            "contradiction, surprise, or red flag to justify it.\n\n"
            "IMPORTANT: Do NOT fail a case simply because a contradiction/surprise has "
            "no priority-1 probe — the analyst is capped at 3 probes and may have chosen "
            "not to probe it. Only fail on mis-assignment (priority-1 where it shouldn't be).\n\n"
            "The transcript is provided as the input."
        ),
        model=_JUDGE_MODEL,
        include_input=True,
        score=False,
        assertion={"evaluation_name": "priority_calibrated", "include_reason": True},
    )
