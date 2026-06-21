"""add_total_tokens_to_cost_ledger

Revision ID: 73a8b0323eb3
Revises: a1b2c3d4e5f6
Create Date: 2026-06-14 14:11:04.534227

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '73a8b0323eb3'
down_revision: Union[str, None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ``total_tokens`` is declared on the ORM model (models.CostLedger) and
    # written by CostTracker.record_usage, but the initial schema migration
    # (6f3695ef31c6) created cost_ledger WITHOUT this column. Every cost
    # INSERT therefore failed silently, leaving cost_ledger empty and the
    # budget gate inert — F10.1. This reconciles the migration history with
    # the ORM so CostTracker rows land.
    op.add_column(
        "cost_ledger",
        sa.Column("total_tokens", sa.Integer(), nullable=False, server_default="0"),
    )
    # Backfill any pre-existing rows with the true total (input + output)
    # rather than the 0 placeholder. A no-op on an empty table (the common
    # case — the column's absence meant no rows were ever written).
    op.execute("UPDATE cost_ledger SET total_tokens = input_tokens + output_tokens")


def downgrade() -> None:
    op.drop_column("cost_ledger", "total_tokens")
