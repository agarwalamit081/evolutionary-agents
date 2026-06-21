"""add_eval_results

Revision ID: d5f6a7b8c9d0
Revises: c4e5a6b7c8d9
Create Date: 2026-06-17 00:00:00.000000

Adds the ``eval_results`` table — the durable, queryable projection of the
Phase-3 correctness harness. One row per (run, goal, check): goal_id, run_id,
spec_id, check_name, check_type, passed, score (Numeric(5,4), 0–1), skipped,
evidence (JSONB), cost_usd, created_at. Indexed by (goal_id, created_at) for
per-goal regression tracking and (run_id, check_name) for per-run rollups; the
Phase-8 evolution canary queries the former to decide promotion vs rollback.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


# revision identifiers, used by Alembic.
revision: str = "d5f6a7b8c9d0"
down_revision: Union[str, None] = "c4e5a6b7c8d9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "eval_results",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("goal_id", sa.Text(), nullable=False),
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("spec_id", sa.Text(), nullable=True),
        sa.Column("check_name", sa.Text(), nullable=False),
        sa.Column("check_type", sa.Text(), nullable=False),
        sa.Column("passed", sa.Boolean(), nullable=False),
        sa.Column("score", sa.Numeric(5, 4), nullable=False, server_default="0"),
        sa.Column("skipped", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("evidence", JSONB, nullable=True),
        sa.Column("cost_usd", sa.Numeric(10, 6), nullable=False, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_eval_results_goal",
        "eval_results",
        ["goal_id", "created_at"],
    )
    op.create_index(
        "idx_eval_results_run",
        "eval_results",
        ["run_id", "check_name"],
    )


def downgrade() -> None:
    op.drop_index("idx_eval_results_run", table_name="eval_results")
    op.drop_index("idx_eval_results_goal", table_name="eval_results")
    op.drop_table("eval_results")
