"""Happy-path unit tests for state.py + models.py."""

from __future__ import annotations

import pytest

from voice_agent.models import (
    AnalysisUpdate,
    InterviewerOutput,
    NewProbe,
    ReportOutput,
    ThemeWithQuotes,
)
from voice_agent.state import (
    AnalystSnapshot,
    Call,
    Probe,
    SynthesisReport,
    Turn,
    call_llm_token_totals,
    init_db,
    make_engine,
    mark_probe_asked,
    mark_scripted_asked,
    next_scripted,
    next_turn_number,
    recent_turns,
    scripted_remaining,
    session_scope,
    top_probes,
)


@pytest.fixture()
def engine():
    eng = make_engine("sqlite:///:memory:")
    init_db(eng)
    return eng


@pytest.fixture()
def seeded_call(engine):
    with session_scope(engine) as s:
        call = Call(
            id="call-123",
            phone_number="+15550100",
            scripted_questions=[
                "How do you currently use the product?",
                "What would you change about it?",
                "Who else might benefit from it?",
            ],
            status="active",
        )
        s.add(call)
    return "call-123"


def test_call_roundtrip(engine, seeded_call):
    with session_scope(engine) as s:
        call = s.get(Call, seeded_call)
        assert call is not None
        assert call.status == "active"
        assert len(call.scripted_questions) == 3
        assert call.scripted_questions[0].startswith("How do you")


def test_next_turn_number_returns_int(engine, seeded_call):
    with session_scope(engine) as s:
        n = next_turn_number(s, seeded_call)
        assert n == 1
        assert type(n) is int
        # Writing a Turn row advances the counter.
        s.add(Turn(call_id=seeded_call, turn_number=1, speaker="respondent", text="hi"))
        assert next_turn_number(s, seeded_call) == 2


def test_turns_order_and_cascade(engine, seeded_call):
    with session_scope(engine) as s:
        s.add_all([
            Turn(
                call_id=seeded_call,
                turn_number=1,
                speaker="interviewer",
                text="How do you currently use the product?",
                action="scripted",
            ),
            Turn(
                call_id=seeded_call,
                turn_number=2,
                speaker="respondent",
                text="Every morning for my commute planning.",
            ),
        ])

    with session_scope(engine) as s:
        turns = recent_turns(s, seeded_call, n=10)
        assert [t.turn_number for t in turns] == [1, 2]
        assert turns[0].action == "scripted"
        assert turns[1].speaker == "respondent"

    # Cascade delete: removing the Call removes its Turns.
    with session_scope(engine) as s:
        s.delete(s.get(Call, seeded_call))
    with session_scope(engine) as s:
        assert recent_turns(s, seeded_call) == []


def test_next_scripted_and_remaining(engine, seeded_call):
    with session_scope(engine) as s:
        assert scripted_remaining(s, seeded_call) == 3
        assert next_scripted(s, seeded_call) == "How do you currently use the product?"
        mark_scripted_asked(s, seeded_call)

    with session_scope(engine) as s:
        assert scripted_remaining(s, seeded_call) == 2
        assert next_scripted(s, seeded_call) == "What would you change about it?"


def test_probe_priority_pop_and_mark(engine, seeded_call):
    with session_scope(engine) as s:
        s.add_all([
            Probe(call_id=seeded_call, question="Why specifically?", priority=1, rationale="flag"),
            Probe(call_id=seeded_call, question="Anything else?", priority=3, rationale="soft"),
            Probe(call_id=seeded_call, question="Who told you?", priority=2, rationale="source"),
        ])

    with session_scope(engine) as s:
        tops = top_probes(s, seeded_call, n=1)
        assert tops
        top = tops[0]
        assert top.priority == 1
        assert top.question == "Why specifically?"
        mark_probe_asked(s, top.id)

    with session_scope(engine) as s:
        tops = top_probes(s, seeded_call, n=1)
        assert tops and tops[0].priority == 2


def test_analyst_snapshot_and_synthesis(engine, seeded_call):
    with session_scope(engine) as s:
        s.add(AnalystSnapshot(
            call_id=seeded_call,
            after_turn=4,
            themes=["commute-first usage"],
            contradictions=[],
            surprises=["uses it offline"],
            latency_ms=11_400,
        ))
        s.add(SynthesisReport(
            call_id=seeded_call,
            summary="Power user, commute-focused.",
            themes=[{"theme": "commute", "quotes": ["every morning"]}],
            contradictions=[],
            key_quotes=["every morning for my commute planning"],
            follow_up_questions=["What breaks the flow on bad-signal days?"],
        ))

    with session_scope(engine) as s:
        call = s.get(Call, seeded_call)
        assert len(call.analyst_snapshots) == 1
        assert call.analyst_snapshots[0].themes == ["commute-first usage"]
        assert call.synthesis_report is not None
        assert call.synthesis_report.themes[0]["theme"] == "commute"


# --- Pydantic model contracts ---------------------------------------------


def test_interviewer_output_actions():
    out = InterviewerOutput(
        utterance="Got it — what made you stop using it?",
        action="probe",
        reasoning="Respondent hinted at churn.",
    )
    assert out.action == "probe"


def test_interviewer_output_rejects_bad_action():
    with pytest.raises(Exception):
        InterviewerOutput(utterance="hi", action="banter", reasoning="")  # type: ignore[arg-type]


def test_analysis_update_probe_priority_bounds():
    ok = AnalysisUpdate(
        themes=["trust"],
        new_probes=[NewProbe(question="Why?", priority=1, rationale="contradiction")],
    )
    assert ok.new_probes[0].priority == 1
    with pytest.raises(Exception):
        NewProbe(question="Why?", priority=5, rationale="x")


def test_report_output_shape():
    r = ReportOutput(
        summary="s",
        themes=[ThemeWithQuotes(theme="t", quotes=["q"])],
        follow_up_questions=["more?"],
    )
    assert r.themes[0].theme == "t"
    assert r.contradictions == []


def test_call_llm_token_totals_sums_turns_and_snapshots(engine, seeded_call):
    with session_scope(engine) as s:
        s.add(
            Turn(
                call_id=seeded_call,
                turn_number=1,
                speaker="interviewer",
                text="Hi",
                tokens_input=10,
                tokens_output=5,
                tokens_cache_read=1,
                tokens_cache_write=0,
            )
        )
        s.add(
            AnalystSnapshot(
                call_id=seeded_call,
                after_turn=1,
                themes=[],
                tokens_input=1000,
                tokens_output=200,
                tokens_cache_read=50,
                tokens_cache_write=10,
            )
        )
    with session_scope(engine) as s:
        totals = call_llm_token_totals(s, seeded_call)
    assert totals == {
        "tokens_input": 1010,
        "tokens_output": 205,
        "tokens_cache_read": 51,
        "tokens_cache_write": 10,
    }
