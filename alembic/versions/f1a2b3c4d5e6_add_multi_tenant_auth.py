"""add multi-tenant auth: organizations, users, org_members; org_id on projects/calls

Revision ID: f1a2b3c4d5e6
Revises: a1b2c3d4e5f6
Create Date: 2026-06-16

"""
from typing import Sequence, Union
import uuid

import sqlalchemy as sa
from alembic import op

revision: str = "f1a2b3c4d5e6"
down_revision: Union[str, Sequence[str], None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

DEFAULT_ORG_ID = "00000000-0000-0000-0000-000000000001"


def upgrade() -> None:
    op.create_table(
        "organizations",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("slug", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("slug", name="uq_organizations_slug"),
    )

    op.create_table(
        "users",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("email", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=True),
        sa.Column("avatar_url", sa.String(), nullable=True),
        sa.Column("google_sub", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email", name="uq_users_email"),
        sa.UniqueConstraint("google_sub", name="uq_users_google_sub"),
    )
    op.create_index("ix_users_email", "users", ["email"])
    op.create_index("ix_users_google_sub", "users", ["google_sub"])

    op.create_table(
        "org_members",
        sa.Column("org_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("role", sa.String(), nullable=False),  # admin | member
        sa.Column("invited_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("joined_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("org_id", "user_id"),
    )
    op.create_index("ix_org_members_org_id", "org_members", ["org_id"])
    op.create_index("ix_org_members_user_id", "org_members", ["user_id"])

    # Seed the default organization for existing data
    now = sa.func.now()
    op.execute(
        sa.text(
            "INSERT INTO organizations (id, name, slug, created_at) "
            "VALUES (:id, :name, :slug, :created_at)"
        ).bindparams(
            id=DEFAULT_ORG_ID,
            name="Default",
            slug="default",
            created_at="2026-06-16T00:00:00+00:00",
        )
    )

    # Add org_id to projects (nullable first, then backfill, then NOT NULL)
    op.add_column("projects", sa.Column("org_id", sa.String(), nullable=True))
    op.execute(
        sa.text("UPDATE projects SET org_id = :org_id").bindparams(org_id=DEFAULT_ORG_ID)
    )
    # SQLite doesn't support ALTER COLUMN to add NOT NULL after the fact,
    # but the app code will enforce non-null at write time.
    # For Postgres compatibility we leave a comment; real enforcement is in the ORM.
    op.create_index("ix_projects_org_id", "projects", ["org_id"])

    # Add org_id to calls (same pattern)
    op.add_column("calls", sa.Column("org_id", sa.String(), nullable=True))
    op.execute(
        sa.text("UPDATE calls SET org_id = :org_id").bindparams(org_id=DEFAULT_ORG_ID)
    )
    op.create_index("ix_calls_org_id", "calls", ["org_id"])


def downgrade() -> None:
    op.drop_index("ix_calls_org_id", table_name="calls")
    op.drop_column("calls", "org_id")

    op.drop_index("ix_projects_org_id", table_name="projects")
    op.drop_column("projects", "org_id")

    op.drop_index("ix_org_members_user_id", table_name="org_members")
    op.drop_index("ix_org_members_org_id", table_name="org_members")
    op.drop_table("org_members")

    op.drop_index("ix_users_google_sub", table_name="users")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")

    op.drop_table("organizations")
