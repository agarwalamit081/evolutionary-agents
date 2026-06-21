"""add_attempt_id_to_eval_results

Revision ID: g9d0e1f2a3b4
Revises: f7b8c9d0e1f2
Create Date: 2026-06-19 16:00:00.000000

Per-run-attempt scoring for the eval harness. ``eval_results.run_id`` is the
graph ``thread_id`` (``cli-{run_id}``), which is deliberately STABLE across
re-runs of the same ``--run_id`` so ``--resume`` can reuse the checkpoint. The
side effect is that every re-run of, say, q03 blends its check rows under one
``run_id`` — so a queried "score" for ``cli-q03`` was a cross-attempt blend,
not one attempt's result. This adds a free ``attempt_id`` Text column populated
by the verify node from ``state.eval_attempt_id`` (generated once per
invocation in ``main.py``), plus a supporting ``(run_id, attempt_id,
created_at)`` index so ``EvalStore.query_latest_attempt`` can return the newest
attempt's rows. Nullable/additive: legacy rows carry NULL and remain queryable
by run as before; backfill is impossible (the attempt discriminator was never
recorded for prior rows).
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = "g9d0e1f2a3b4"
down_revision: Union[str, None] = "f7b8c9d0e1f2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "eval_results", sa.Column("attempt_id", sa.Text(), nullable=True)
    )
    op.create_index(
        "idx_eval_results_attempt",
        "eval_results",
        ["run_id", "attempt_id", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("idx_eval_results_attempt", table_name="eval_results")
    op.drop_column("eval_results", "attempt_id")
