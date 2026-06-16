"""add projects table and project_id to calls

Revision ID: a1b2c3d4e5f6
Revises: k3a85gcrh19b
Create Date: 2026-06-15

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "k3a85gcrh19b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "projects",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("product", sa.String(), nullable=False),
        sa.Column("product_description", sa.String(), nullable=True),
        sa.Column("focus_areas", sa.JSON(), nullable=False),
        sa.Column("deprioritize", sa.JSON(), nullable=False),
        sa.Column("investor_thesis", sa.String(), nullable=True),
        sa.Column("scripted_questions", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    # SQLite doesn't enforce FK constraints, so a plain add_column is sufficient.
    # The FK is declared in the ORM for relationship loading only.
    op.add_column("calls", sa.Column("project_id", sa.String(), nullable=True))
    op.create_index("ix_calls_project_id", "calls", ["project_id"])


def downgrade() -> None:
    op.drop_index("ix_calls_project_id", table_name="calls")
    op.drop_column("calls", "project_id")
    op.drop_table("projects")
