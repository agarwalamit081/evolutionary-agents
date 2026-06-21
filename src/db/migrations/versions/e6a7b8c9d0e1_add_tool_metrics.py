"""add_tool_metrics

Revision ID: e6a7b8c9d0e1
Revises: d5f6a7b8c9d0
Create Date: 2026-06-17 00:00:00.000000

Adds per-tool success metrics (M4): running-aggregate columns on
``tool_registrations`` (``calls``, ``success_rate``, ``empty_output_rate``,
``last_run_at``) maintained incrementally by ``ToolMetricsRecorder``, plus the
append-only ``tool_call_metrics`` detail table (one row per invocation) behind
them. The aggregates let governance retire chronic low performers
(``calls >= RETIRE_MIN_RUNS`` and ``success_rate < RETIRE_SUCCESS_FLOOR``)
without re-aggregating the detail table. New columns are ``NOT NULL`` with
server defaults so existing rows (untried tools) score 1.0 success / 0 calls
and are never retired for performance.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "e6a7b8c9d0e1"
down_revision: Union[str, None] = "d5f6a7b8c9d0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Running aggregates on tool_registrations. server_default seeds existing
    # rows (an untried tool succeeds-by-default / never-empty so it survives).
    op.add_column(
        "tool_registrations",
        sa.Column("calls", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "tool_registrations",
        sa.Column("success_rate", sa.Float(), nullable=False, server_default="1.0"),
    )
    op.add_column(
        "tool_registrations",
        sa.Column(
            "empty_output_rate", sa.Float(), nullable=False, server_default="0.0"
        ),
    )
    op.add_column(
        "tool_registrations",
        sa.Column("last_run_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_table(
        "tool_call_metrics",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tool_name", sa.Text(), nullable=False),
        sa.Column("run_id", sa.Text(), nullable=True),
        sa.Column("success", sa.Boolean(), nullable=False),
        sa.Column("empty_output", sa.Boolean(), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_tool_call_metrics_tool",
        "tool_call_metrics",
        ["tool_name", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("idx_tool_call_metrics_tool", table_name="tool_call_metrics")
    op.drop_table("tool_call_metrics")
    op.drop_column("tool_registrations", "last_run_at")
    op.drop_column("tool_registrations", "empty_output_rate")
    op.drop_column("tool_registrations", "success_rate")
    op.drop_column("tool_registrations", "calls")
