"""add brain to calls

Revision ID: k3a85gcrh19b
Revises: e205a9b6ea86
Create Date: 2026-06-13

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "k3a85gcrh19b"
down_revision: Union[str, Sequence[str], None] = "e205a9b6ea86"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("calls", sa.Column("brain", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("calls", "brain")
