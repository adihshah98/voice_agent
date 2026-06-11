"""SQLModel tables and DB helpers.

Single SQLite file in dev; schema is Postgres-compatible. All cross-agent
coordination flows through these tables — agents never call each other.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

import logfire
from sqlalchemy import Column, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.types import JSON
from sqlalchemy.pool import NullPool, StaticPool
from sqlmodel import Field, Relationship, Session, SQLModel, create_engine, func, select


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# Outbound dial lifecycle (see server Phase 4). None on Call = no outbound dial.
DIAL_QUEUED = "queued"
DIAL_DIALING = "dialing"
DIAL_DIALED = "dialed"
DIAL_FAILED = "dial_failed"


class Call(SQLModel, table=True):
    __tablename__ = "calls"

    id: str = Field(primary_key=True)
    vapi_call_id: Optional[str] = Field(default=None, unique=True, index=True)
    phone_number: Optional[str] = None
    scripted_questions: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    scripted_cursor: int = Field(default=0)
    status: str = Field(default="pending")  # pending|active|ended
    dial_status: Optional[str] = None  # queued|dialing|dialed|dial_failed; None = no dial
    dial_error: Optional[str] = None
    end_reason: Optional[str] = None
    started_at: datetime = Field(default_factory=_utcnow)
    ended_at: Optional[datetime] = None

    turns: list["Turn"] = Relationship(
        sa_relationship=relationship(
            "Turn",
            back_populates="call",
            cascade="all, delete-orphan",
            order_by="Turn.turn_number",
        ),
    )
    probes: list["Probe"] = Relationship(
        sa_relationship=relationship(
            "Probe",
            back_populates="call",
            cascade="all, delete-orphan",
        ),
    )
    analyst_snapshots: list["AnalystSnapshot"] = Relationship(
        sa_relationship=relationship(
            "AnalystSnapshot",
            back_populates="call",
            cascade="all, delete-orphan",
            order_by="AnalystSnapshot.id",
        ),
    )
    synthesis_report: Optional["SynthesisReport"] = Relationship(
        sa_relationship=relationship(
            "SynthesisReport",
            back_populates="call",
            cascade="all, delete-orphan",
            uselist=False,
        ),
    )


class Turn(SQLModel, table=True):
    __tablename__ = "turns"
    __table_args__ = (UniqueConstraint("call_id", "turn_number", name="uq_turns_call_turn_number"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    call_id: str = Field(foreign_key="calls.id", index=True)
    turn_number: int
    speaker: str  # "interviewer" | "respondent"
    text: str
    action: Optional[str] = None  # scripted|skip_scripted|probe|clarify|off_topic|wrap_up
    probe_source: Optional[str] = None  # "analyst" | "interviewer" | None (non-probe turns)
    reasoning: Optional[str] = None
    latency_ms: Optional[int] = None
    # Interviewer LLM only (null for respondent rows, or if timeout/fallback had no usage)
    tokens_input: Optional[int] = None
    tokens_output: Optional[int] = None
    tokens_cache_read: Optional[int] = None
    tokens_cache_write: Optional[int] = None
    barge_in_truncated: bool = Field(default=False)
    created_at: datetime = Field(default_factory=_utcnow)

    call: Call = Relationship(
        sa_relationship=relationship("Call", back_populates="turns")
    )


class Probe(SQLModel, table=True):
    __tablename__ = "probes"

    id: Optional[int] = Field(default=None, primary_key=True)
    call_id: str = Field(foreign_key="calls.id", index=True)
    question: str
    priority: int  # 1 (highest) .. 3
    rationale: Optional[str] = None
    generated_after_turn: Optional[int] = None  # turn number after which this probe was created
    asked: bool = Field(default=False)
    asked_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=_utcnow)

    call: Call = Relationship(
        sa_relationship=relationship("Call", back_populates="probes")
    )


class AnalystSnapshot(SQLModel, table=True):
    __tablename__ = "analyst_snapshots"

    id: Optional[int] = Field(default=None, primary_key=True)
    call_id: str = Field(foreign_key="calls.id", index=True)
    after_turn: int
    after_scripted_cursor: int = Field(default=0)
    themes: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    contradictions: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    surprises: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    investor_signals: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    covered_subtopics: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    latency_ms: Optional[int] = None
    tokens_input: Optional[int] = None
    tokens_output: Optional[int] = None
    tokens_cache_read: Optional[int] = None
    tokens_cache_write: Optional[int] = None
    created_at: datetime = Field(default_factory=_utcnow)

    call: Call = Relationship(
        sa_relationship=relationship("Call", back_populates="analyst_snapshots")
    )


class SynthesisReport(SQLModel, table=True):
    __tablename__ = "synthesis_reports"

    id: Optional[int] = Field(default=None, primary_key=True)
    call_id: str = Field(foreign_key="calls.id", unique=True, index=True)
    summary: str
    themes: list[dict] = Field(default_factory=list, sa_column=Column(JSON))
    contradictions: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    key_quotes: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    follow_up_questions: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    pmf_score: int = Field(default=0)
    pmf_score_rationale: str = Field(default="")
    competitive_signals: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    revenue_signals: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    ai_adoption_signals: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    red_flags: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    investment_thesis_bullets: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    tokens_input: Optional[int] = None
    tokens_output: Optional[int] = None
    tokens_cache_read: Optional[int] = None
    tokens_cache_write: Optional[int] = None
    created_at: datetime = Field(default_factory=_utcnow)

    call: Call = Relationship(
        sa_relationship=relationship("Call", back_populates="synthesis_report")
    )


# --- DB helpers -------------------------------------------------------------


def make_engine(url: str = "sqlite:///voice_agent.db", *, echo: bool = False):
    # Normalize bare postgresql:// → postgresql+psycopg2:// (Supabase/Render copy-paste).
    if url.startswith("postgresql://") or url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg2://", 1).replace(
            "postgresql://", "postgresql+psycopg2://", 1
        )
    if url.startswith("sqlite"):
        if ":memory:" in url:
            # StaticPool keeps a single connection so all sessions share the
            # same in-memory DB — required for tests/evals.
            return create_engine(url, echo=echo, connect_args={"check_same_thread": False}, poolclass=StaticPool)
        # NullPool: open/close a connection per session. File-based SQLite has
        # no TCP overhead, and pooling only causes QueuePool exhaustion under
        # concurrent requests.
        return create_engine(url, echo=echo, connect_args={"check_same_thread": False}, poolclass=NullPool)
    return create_engine(url, echo=echo)





def init_db(engine) -> None:
    # Schema is managed by Alembic migrations. create_all() is kept for
    # in-memory SQLite only (tests/evals), where there is no migration history.
    if str(engine.url).startswith("sqlite:///:memory:"):
        SQLModel.metadata.create_all(engine)


@contextmanager
def session_scope(engine) -> Iterator[Session]:
    session = Session(engine, expire_on_commit=False)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        logfire.exception("db_session_rollback")
        raise
    finally:
        session.close()


# --- Read helpers used by interviewer tools --------------------------------


def next_turn_number(session: Session, call_id: str) -> int:
    """Return COUNT(turns for this call) + 1 — the turn_number of the next utterance to be written."""
    return session.exec(select(func.count()).where(Turn.call_id == call_id)).one() + 1


def next_scripted(session: Session, call_id: str) -> Optional[str]:
    call = session.get(Call, call_id)
    if call is None:
        logfire.warning("next_scripted_call_missing", call_id=call_id)
        return None
    if call.scripted_cursor >= len(call.scripted_questions):
        return None
    return call.scripted_questions[call.scripted_cursor]


def scripted_remaining(session: Session, call_id: str) -> int:
    call = session.get(Call, call_id)
    if call is None:
        logfire.warning("scripted_remaining_call_missing", call_id=call_id)
        return 0
    return max(0, len(call.scripted_questions) - call.scripted_cursor)


def mark_scripted_asked(session: Session, call_id: str) -> None:
    call = session.get(Call, call_id)
    if call is None:
        logfire.warning("mark_scripted_asked_call_missing", call_id=call_id)
        return
    call.scripted_cursor += 1
    session.add(call)


def top_probes(session: Session, call_id: str, n: int = 3, min_turn: int = 0) -> list[Probe]:
    stmt = (
        select(Probe)
        .where(
            Probe.call_id == call_id,
            Probe.asked == False,  # noqa: E712
            (Probe.generated_after_turn == None) | (Probe.generated_after_turn >= min_turn),  # noqa: E711
        )
        .order_by(Probe.priority.asc(), Probe.created_at.asc())
        .limit(n)
    )
    return list(session.exec(stmt))


def probe_utilization(session: Session, call_id: str) -> tuple[int, int]:
    """Return (asked, total) probe counts for the call."""
    probes = list(session.exec(select(Probe).where(Probe.call_id == call_id)))
    return sum(1 for p in probes if p.asked), len(probes)


def mark_probe_asked(session: Session, probe_id: int) -> None:
    probe = session.get(Probe, probe_id)
    if probe is None:
        logfire.warning("mark_probe_asked_probe_missing", probe_id=probe_id)
        return
    probe.asked = True
    probe.asked_at = _utcnow()
    session.add(probe)


def latest_snapshot(session: Session, call_id: str) -> Optional["AnalystSnapshot"]:
    stmt = (
        select(AnalystSnapshot)
        .where(AnalystSnapshot.call_id == call_id)
        .order_by(AnalystSnapshot.after_turn.desc())
        .limit(1)
    )
    return session.exec(stmt).first()


def call_llm_token_totals(session: Session, call_id: str) -> dict[str, int]:
    """Sum LLM token fields for this call: interviewer (Turn) + analyst (AnalystSnapshot) + synthesis."""
    out: dict[str, int] = {}
    for key in ("tokens_input", "tokens_output", "tokens_cache_read", "tokens_cache_write"):
        tcol = getattr(Turn, key)
        acol = getattr(AnalystSnapshot, key)
        scol = getattr(SynthesisReport, key)
        t = session.exec(select(func.coalesce(func.sum(tcol), 0)).where(Turn.call_id == call_id)).one()
        a = session.exec(select(func.coalesce(func.sum(acol), 0)).where(AnalystSnapshot.call_id == call_id)).one()
        s = session.exec(select(func.coalesce(func.sum(scol), 0)).where(SynthesisReport.call_id == call_id)).one()
        out[key] = int(t or 0) + int(a or 0) + int(s or 0)
    return out


ANALYST_TURN_INTERVAL = 10


def should_run_analyst(session: Session, call_id: str) -> bool:
    """True when a scripted question was just answered OR >= 10 exchanges have passed since last run.

    Counts interviewer (bot) turns since the last snapshot, not call.turn_count, because Vapi
    can split one user utterance into multiple Turn rows at pauses — counting bot rows gives
    exactly one count per complete exchange regardless of how the user's speech was segmented.
    """
    call = session.get(Call, call_id)
    if call is None:
        logfire.warning("should_run_analyst_call_missing", call_id=call_id)
        return False
    snapshot = latest_snapshot(session, call_id)
    scripted_advanced = call.scripted_cursor > (snapshot.after_scripted_cursor if snapshot else 0)
    after_turn = snapshot.after_turn if snapshot else 0
    exchanges_since = session.exec(
        select(func.count()).where(
            Turn.call_id == call_id,
            Turn.turn_number > after_turn,
            Turn.speaker == "interviewer",
        )
    ).one()
    turns_elapsed = exchanges_since >= ANALYST_TURN_INTERVAL
    return scripted_advanced or turns_elapsed


def turns_since(session: Session, call_id: str, after_turn: int) -> list[Turn]:
    stmt = (
        select(Turn)
        .where(Turn.call_id == call_id, Turn.turn_number > after_turn)
        .order_by(Turn.turn_number.asc())
    )
    return list(session.exec(stmt))


def call_summary_stats(session: Session, call_id: str) -> dict:
    """Return post-call aggregate stats for online eval summary.

    Returns scripted arc completion fraction, probe utilization, and barge-in count.
    """
    call = session.get(Call, call_id)
    if call is None:
        return {}

    total_scripted = len(call.scripted_questions)
    scripted_pct = round(100 * call.scripted_cursor / total_scripted) if total_scripted else None

    probes_asked, probes_total = probe_utilization(session, call_id)
    probe_pct = round(100 * probes_asked / probes_total) if probes_total else None

    barge_in_count = session.exec(
        select(func.count()).where(
            Turn.call_id == call_id,
            Turn.barge_in_truncated == True,  # noqa: E712
        )
    ).one()

    interviewer_turns = session.exec(
        select(func.count()).where(
            Turn.call_id == call_id,
            Turn.speaker == "interviewer",
        )
    ).one()

    fallback_count = session.exec(
        select(func.count()).where(
            Turn.call_id == call_id,
            Turn.speaker == "interviewer",
            Turn.action == None,  # noqa: E711 — fallback rows have no action set
        )
    ).one()

    return {
        "scripted_cursor": call.scripted_cursor,
        "scripted_total": total_scripted,
        "scripted_arc_pct": scripted_pct,
        "probes_asked": probes_asked,
        "probes_total": probes_total,
        "probe_utilization_pct": probe_pct,
        "barge_in_count": int(barge_in_count),
        "interviewer_turns": int(interviewer_turns),
        "fallback_count": int(fallback_count),
    }


def consecutive_clarify_count(session: Session, call_id: str) -> int:
    """Count trailing consecutive interviewer turns with action='clarify'.

    Used to populate SILENCE_HANDLER_COUNT in the interviewer prompt so the
    model knows how many re-engagement attempts have already been made without
    scanning utterance text (which is fragile when fillers are prepended).
    """
    recent = recent_turns(session, call_id, n=20)
    count = 0
    for turn in reversed(recent):
        if turn.speaker != "interviewer":
            continue
        if turn.action == "clarify":
            count += 1
        else:
            break
    return count


def recent_turns(session: Session, call_id: str, n: int = 6) -> list[Turn]:
    stmt = (
        select(Turn)
        .where(Turn.call_id == call_id)
        .order_by(Turn.turn_number.desc())
        .limit(n)
    )
    rows = list(session.exec(stmt))
    rows.reverse()
    return rows
