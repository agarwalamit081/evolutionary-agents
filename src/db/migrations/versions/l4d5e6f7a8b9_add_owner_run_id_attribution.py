"""add_owner_run_id_attribution

Revision ID: l4d5e6f7a8b9
Revises: k3c4d5e6f7a8
Create Date: 2026-07-13 00:00:00.000000

Track-1 attribution columns so ``scripts/run_metrics.py`` can attribute tools,
sub-agents, and per-node timings to the run that produced them WITHOUT fuzzy
time-window joins (the prior proxy joined on created_at overlap, which smeared
attribution across concurrent/adjacent runs).

- ``tool_registrations.owner_run_id`` — the run that generated the tool
  (channel-A create→reuse circuit). NULL for built-ins + pre-migration rows.
- ``sub_agent_definitions.owner_run_id`` — the run that spawned this agent
  version (each SubAgentModel row is a version).
- ``execution_steps.run_id`` — per-node wall-clock timing rows keyed by run_id
  (written by the graph ``_wrap`` node-timer; replaces the ``llm_span_seconds``
  proxy). ``task_id`` is relaxed to NULL so timing-only rows (no
  ``task_executions`` parent) can be written.
- ``idx_tool_call_metrics_run`` — index on
  ``tool_call_metrics(run_id, created_at)`` so the per-run tool-metrics query is
  index-backed. The ``run_id`` column itself was added by ``e6a7b8c9d0e1``; only
  the index is new here.

All additive/nullable: legacy rows carry NULL and remain queryable. Downgrade
restores ``task_id`` NOT NULL after deleting the timing-only rows it introduced.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = "l4d5e6f7a8b9"
down_revision: Union[str, None] = "k3c4d5e6f7a8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Channel-A tool attribution: which run generated this tool.
    op.add_column(
        "tool_registrations",
        sa.Column("owner_run_id", sa.Text(), nullable=True),
    )
    # Sub-agent version attribution: which run spawned this version row.
    op.add_column(
        "sub_agent_definitions",
        sa.Column("owner_run_id", sa.Text(), nullable=True),
    )
    # Per-node timing attribution. Relax task_id so the node-timer can write
    # timing-only rows carrying no task_executions parent (run_metrics keys off
    # run_id for per-node wall-clock per run). Batch mode is the cross-dialect
    # portable form: native ALTER COLUMN on Postgres, table-rebuild on SQLite
    # (which has no ALTER COLUMN … DROP NOT NULL).
    op.add_column(
        "execution_steps",
        sa.Column("run_id", sa.Text(), nullable=True),
    )
    with op.batch_alter_table("execution_steps", schema=None) as batch:
        batch.alter_column("task_id", existing_type=sa.Uuid(), nullable=True)
    # Index the run-scoped tool-metrics query (run_id column predates this
    # migration; only the composite index is new).
    op.create_index(
        "idx_tool_call_metrics_run",
        "tool_call_metrics",
        ["run_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("idx_tool_call_metrics_run", table_name="tool_call_metrics")
    # Restore task_id NOT NULL: timing-only rows (task_id IS NULL) are an
    # artifact of this migration's node-timer — delete them so the NOT-NULL
    # constraint can be reapplied. Static DDL in a migration (no user input).
    op.execute("DELETE FROM execution_steps WHERE task_id IS NULL")
    with op.batch_alter_table("execution_steps", schema=None) as batch:
        batch.alter_column("task_id", existing_type=sa.Uuid(), nullable=False)
    op.drop_column("execution_steps", "run_id")
    op.drop_column("sub_agent_definitions", "owner_run_id")
    op.drop_column("tool_registrations", "owner_run_id")
