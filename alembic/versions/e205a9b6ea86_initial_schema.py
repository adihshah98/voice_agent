"""initial_schema

Revision ID: e205a9b6ea86
Revises:
Create Date: 2026-06-10

"""
from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from alembic import op

revision: str = "e205a9b6ea86"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "calls",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("vapi_call_id", sa.String(), nullable=True),
        sa.Column("phone_number", sa.String(), nullable=True),
        sa.Column("scripted_questions", sa.JSON(), nullable=False),
        sa.Column("scripted_cursor", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(), nullable=False, server_default="pending"),
        sa.Column("dial_status", sa.String(), nullable=True),
        sa.Column("dial_error", sa.String(), nullable=True),
        sa.Column("end_reason", sa.String(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("vapi_call_id"),
    )
    op.create_index("ix_calls_vapi_call_id", "calls", ["vapi_call_id"])

    op.create_table(
        "turns",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("call_id", sa.String(), nullable=False),
        sa.Column("turn_number", sa.Integer(), nullable=False),
        sa.Column("speaker", sa.String(), nullable=False),
        sa.Column("text", sa.String(), nullable=False),
        sa.Column("action", sa.String(), nullable=True),
        sa.Column("probe_source", sa.String(), nullable=True),
        sa.Column("reasoning", sa.String(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("tokens_input", sa.Integer(), nullable=True),
        sa.Column("tokens_output", sa.Integer(), nullable=True),
        sa.Column("tokens_cache_read", sa.Integer(), nullable=True),
        sa.Column("tokens_cache_write", sa.Integer(), nullable=True),
        sa.Column("barge_in_truncated", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["call_id"], ["calls.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("call_id", "turn_number", name="uq_turns_call_turn_number"),
    )
    op.create_index("ix_turns_call_id", "turns", ["call_id"])

    op.create_table(
        "probes",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("call_id", sa.String(), nullable=False),
        sa.Column("question", sa.String(), nullable=False),
        sa.Column("priority", sa.Integer(), nullable=False),
        sa.Column("rationale", sa.String(), nullable=True),
        sa.Column("generated_after_turn", sa.Integer(), nullable=True),
        sa.Column("asked", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("asked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["call_id"], ["calls.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_probes_call_id", "probes", ["call_id"])

    op.create_table(
        "analyst_snapshots",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("call_id", sa.String(), nullable=False),
        sa.Column("after_turn", sa.Integer(), nullable=False),
        sa.Column("after_scripted_cursor", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("themes", sa.JSON(), nullable=False),
        sa.Column("contradictions", sa.JSON(), nullable=False),
        sa.Column("surprises", sa.JSON(), nullable=False),
        sa.Column("investor_signals", sa.JSON(), nullable=False),
        sa.Column("covered_subtopics", sa.JSON(), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("tokens_input", sa.Integer(), nullable=True),
        sa.Column("tokens_output", sa.Integer(), nullable=True),
        sa.Column("tokens_cache_read", sa.Integer(), nullable=True),
        sa.Column("tokens_cache_write", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["call_id"], ["calls.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_analyst_snapshots_call_id", "analyst_snapshots", ["call_id"])

    op.create_table(
        "synthesis_reports",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("call_id", sa.String(), nullable=False),
        sa.Column("summary", sa.String(), nullable=False),
        sa.Column("themes", sa.JSON(), nullable=False),
        sa.Column("contradictions", sa.JSON(), nullable=False),
        sa.Column("key_quotes", sa.JSON(), nullable=False),
        sa.Column("follow_up_questions", sa.JSON(), nullable=False),
        sa.Column("pmf_score", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("pmf_score_rationale", sa.String(), nullable=False, server_default=""),
        sa.Column("competitive_signals", sa.JSON(), nullable=False),
        sa.Column("revenue_signals", sa.JSON(), nullable=False),
        sa.Column("ai_adoption_signals", sa.JSON(), nullable=False),
        sa.Column("red_flags", sa.JSON(), nullable=False),
        sa.Column("investment_thesis_bullets", sa.JSON(), nullable=False),
        sa.Column("tokens_input", sa.Integer(), nullable=True),
        sa.Column("tokens_output", sa.Integer(), nullable=True),
        sa.Column("tokens_cache_read", sa.Integer(), nullable=True),
        sa.Column("tokens_cache_write", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["call_id"], ["calls.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("call_id"),
    )
    op.create_index("ix_synthesis_reports_call_id", "synthesis_reports", ["call_id"])


def downgrade() -> None:
    op.drop_table("synthesis_reports")
    op.drop_table("analyst_snapshots")
    op.drop_table("probes")
    op.drop_table("turns")
    op.drop_table("calls")
