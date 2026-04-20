"""Evaluators for Tier 1 interviewer single-turn eval.

Two deterministic scorers (ActionMatches, SingleQuestion) and two LLM-as-judge
scorers (UtteranceWarmth, NoLeadingQuestions). LLMJudge is imported and
configured with a project-appropriate rubric + judge model; the plan calls for
Opus as the analyst/synthesizer, so we reuse that class for judging too.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext, LLMJudge

from evals.cases import InterviewerCaseInputs
from models import InterviewerOutput


JUDGE_MODEL = "anthropic:claude-opus-4-6"


@dataclass
class ActionMatches(
    Evaluator[InterviewerCaseInputs, InterviewerOutput, None]
):
    """Exact match on the `action` field — the headline Tier 1 score."""

    def evaluate(
        self,
        ctx: EvaluatorContext[
            InterviewerCaseInputs, InterviewerOutput, None
        ],
    ) -> bool:
        if ctx.expected_output is None:
            return False
        return ctx.output.action == ctx.expected_output.action


@dataclass
class SingleQuestion(
    Evaluator[InterviewerCaseInputs, InterviewerOutput, None]
):
    """Utterance must contain at most one `?`.

    Stacking questions in one breath is the single most common interviewer
    failure mode and is trivial to check deterministically.
    """

    def evaluate(
        self,
        ctx: EvaluatorContext[
            InterviewerCaseInputs, InterviewerOutput, None
        ],
    ) -> bool:
        return ctx.output.utterance.count("?") <= 1


def utterance_warmth_judge() -> LLMJudge:
    """LLMJudge on conversational warmth of the spoken utterance."""
    return LLMJudge(
        rubric=(
            "You are grading a single spoken line from a phone-call market "
            "researcher. Score 1–5 on conversational warmth:\n"
            "  5 = warm, natural, acknowledges what the respondent said\n"
            "  4 = friendly but slightly formal\n"
            "  3 = neutral / businesslike\n"
            "  2 = cold, transactional, or robotic\n"
            "  1 = rude, interrogative, or off-putting\n"
            "Grade only the `utterance` field; ignore `reasoning`. Pass the "
            "line if the score is >= 3."
        ),
        model=JUDGE_MODEL,
        include_input=False,
        score={"evaluation_name": "utterance_warmth"},
        assertion=False,
    )


def no_leading_questions_judge() -> LLMJudge:
    """LLMJudge on whether the question biases the respondent."""
    return LLMJudge(
        rubric=(
            "Grade whether the interviewer's utterance contains a LEADING "
            "question. A leading question presupposes the answer (e.g. "
            "'So you loved it, right?', 'That was frustrating, wasn't it?') "
            "or funnels the respondent toward a particular view. Pass if the "
            "question is open and non-leading. Fail if it is leading or "
            "obviously biased. If the utterance contains no question (e.g. "
            "pure acknowledgement or wrap-up), pass.\n"
            "Pass neutral research patterns: inviting detail on a hedge "
            "('fine I guess'), gentle off-topic redirects with an open "
            "follow-up, and probes that reuse the respondent's own words "
            "without adding unstated emotion or blame."
        ),
        model=JUDGE_MODEL,
        include_input=False,
        score=False,
        assertion={"evaluation_name": "non_leading"},
    )
