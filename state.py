"""SQLModel tables and DB helpers.

Single SQLite file in dev; schema is Postgres-compatible. All cross-agent
coordination flows through these tables — agents never call each other.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Column
from sqlalchemy.orm import relationship
from sqlalchemy.types import JSON
from sqlmodel import Field, Relationship, Session, SQLModel, create_engine, select


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Call(SQLModel, table=True):
    __tablename__ = "calls"

    id: str = Field(primary_key=True)
    vapi_call_id: Optional[str] = Field(default=None, index=True)
    phone_number: Optional[str] = None
    scripted_questions: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    scripted_cursor: int = Field(default=0)
    status: str = Field(default="pending")  # pending|active|ended
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

    id: Optional[int] = Field(default=None, primary_key=True)
    call_id: str = Field(foreign_key="calls.id", index=True)
    turn_number: int
    speaker: str  # "interviewer" | "respondent"
    text: str
    action: Optional[str] = None  # scripted|probe|clarify|acknowledge|off_topic|wrap_up
    reasoning: Optional[str] = None
    latency_ms: Optional[int] = None
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
    themes: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    contradictions: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    surprises: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    latency_ms: Optional[int] = None
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
    created_at: datetime = Field(default_factory=_utcnow)

    call: Call = Relationship(
        sa_relationship=relationship("Call", back_populates="synthesis_report")
    )


class EvalRun(SQLModel, table=True):
    __tablename__ = "eval_runs"

    id: Optional[int] = Field(default=None, primary_key=True)
    commit_sha: Optional[str] = None
    dataset: str
    case_name: str
    scorer: str
    score: float
    logfire_trace_id: Optional[str] = None
    created_at: datetime = Field(default_factory=_utcnow)


# --- DB helpers -------------------------------------------------------------


def make_engine(url: str = "sqlite:///voice_agent.db", *, echo: bool = False):
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, echo=echo, connect_args=connect_args)


def init_db(engine) -> None:
    SQLModel.metadata.create_all(engine)


@contextmanager
def session_scope(engine) -> Iterator[Session]:
    session = Session(engine)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# --- Read helpers used by interviewer tools --------------------------------


def next_scripted(session: Session, call_id: str) -> Optional[str]:
    call = session.get(Call, call_id)
    if call is None or call.scripted_cursor >= len(call.scripted_questions):
        return None
    return call.scripted_questions[call.scripted_cursor]


def scripted_remaining(session: Session, call_id: str) -> int:
    call = session.get(Call, call_id)
    if call is None:
        return 0
    return max(0, len(call.scripted_questions) - call.scripted_cursor)


def mark_scripted_asked(session: Session, call_id: str) -> None:
    call = session.get(Call, call_id)
    if call is None:
        return
    call.scripted_cursor += 1
    session.add(call)


def pop_top_probe(session: Session, call_id: str) -> Optional[Probe]:
    stmt = (
        select(Probe)
        .where(Probe.call_id == call_id, Probe.asked == False)  # noqa: E712
        .order_by(Probe.priority.asc(), Probe.created_at.asc())
        .limit(1)
    )
    return session.exec(stmt).first()


def mark_probe_asked(session: Session, probe_id: int) -> None:
    probe = session.get(Probe, probe_id)
    if probe is None:
        return
    probe.asked = True
    probe.asked_at = _utcnow()
    session.add(probe)


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
