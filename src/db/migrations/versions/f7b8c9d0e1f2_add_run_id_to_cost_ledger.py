"""add_run_id_to_cost_ledger

Revision ID: f7b8c9d0e1f2
Revises: e6a7b8c9d0e1
Create Date: 2026-06-19 12:00:00.000000

Per-run cost attribution. ``cost_ledger.task_id`` is a UUID FK into
``task_executions`` (which the agent never populates), so it was always NULL and
per-run cost attribution was impossible. This adds a free ``run_id`` Text column
populated by ``CostTracker.record_usage`` from the graph ``thread_id`` of the
issuing run, plus a supporting index for the per-run spend queries.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = "f7b8c9d0e1f2"
down_revision: Union[str, None] = "e6a7b8c9d0e1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Nullable, additive: legacy rows (and any future unattributed call) simply
    # carry NULL run_id. No backfill is possible — the run identifier was never
    # recorded for prior rows, so historical cost stays unattributable.
    op.add_column("cost_ledger", sa.Column("run_id", sa.Text(), nullable=True))
    op.create_index(
        "idx_cost_ledger_run", "cost_ledger", ["run_id", "created_at"], unique=False
    )


def downgrade() -> None:
    op.drop_index("idx_cost_ledger_run", table_name="cost_ledger")
    op.drop_column("cost_ledger", "run_id")
