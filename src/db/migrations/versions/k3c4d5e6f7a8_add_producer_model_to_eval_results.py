"""add_producer_model_to_eval_results

Revision ID: k3c4d5e6f7a8
Revises: j2b3c4d5e6f7
Create Date: 2026-07-01 18:00:00.000000

Producer-model attribution for the eval harness. ``eval_results`` previously
had no model column, so the capability curve (``src/eval/curve.py``) read a
blended system-wide correctness trend — the thesis ("self-improvement") could
not be read off a model-specific trend. This adds a free ``producer_model``
Text column populated by the verify node with the model id that ran the goal's
execute step (``gateway.resolve_model(complexity, NODE_EXECUTE)``), so the
curve's ``--curve-model`` flag can slice a single model's trend
(``EvalStore.fetch_rows(..., producer_model=m)``). Nullable/additive: legacy
rows carry NULL and remain queryable as before; backfill is impossible (the
producer was never recorded for prior rows). No index: the curve fetches by
``goal_id IN (...)`` + ``created_at`` window (already covered by
``idx_eval_results_goal``) then filters ``producer_model`` from that small
result set (~tens of rows/night).
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = "k3c4d5e6f7a8"
down_revision: Union[str, None] = "j2b3c4d5e6f7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "eval_results", sa.Column("producer_model", sa.Text(), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("eval_results", "producer_model")
