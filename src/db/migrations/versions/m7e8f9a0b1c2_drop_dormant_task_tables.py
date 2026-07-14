"""drop_dormant_task_tables

Revision ID: m7e8f9a0b1c2
Revises: l4d5e6f7a8b9
Create Date: 2026-07-14 00:00:00.000000

Drops two fully-dormant tables — ``task_executions`` and ``feedback_events`` —
plus the four now-orphaned FK columns that pointed at ``task_executions``.

Both tables have **0 rows** and **0 readers/writers** in the application code
(``execution_steps`` is the live per-node-timing replacement, written by the
graph node-timer keyed on ``run_id``; ``RunStatusStore`` in Redis is the live
run-status store). The FK columns ``execution_steps.task_id``,
``cost_ledger.task_id``, ``warm_memories.source_task_id`` and
``sub_agent_runs.parent_task_id`` are all 0-populated (verified pre-migration),
so dropping them loses no data. The dead ``parent_task_id`` plumbing in
``SubAgentPersister.record_run_and_update_metrics`` is removed in the same change.

Fully reversible: ``downgrade`` recreates both tables + columns + FKs + indexes
so the schema round-trips to its prior shape.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'm7e8f9a0b1c2'
down_revision: Union[str, None] = 'l4d5e6f7a8b9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# ── Tables/columns being dropped ────────────────────────────────────────────
_FK_COLS = [
    ("execution_steps", "task_id"),
    ("cost_ledger", "task_id"),
    ("warm_memories", "source_task_id"),
    ("sub_agent_runs", "parent_task_id"),
]

# FK constraints (surviving-table side) that point at task_executions:
# (constraint_name, table_name).
_FKS = [
    ("execution_steps_task_id_fkey", "execution_steps"),
    ("cost_ledger_task_id_fkey", "cost_ledger"),
    ("warm_memories_source_task_id_fkey", "warm_memories"),
    ("sub_agent_runs_parent_task_id_fkey", "sub_agent_runs"),
]

# Indexes on the columns being dropped (surviving tables).
_INDEXES = [
    ("idx_execution_steps_task_number", "execution_steps"),
    ("idx_execution_steps_failed", "execution_steps"),
    ("idx_cost_ledger_task", "cost_ledger"),
]


def upgrade() -> None:
    # 1. Drop the surviving-side FK constraints that reference task_executions
    #    (so the parent table can be dropped).
    for name, table in _FKS:
        op.drop_constraint(name, table, type_="foreignkey")

    # 2. Drop indexes that reference the about-to-be-dropped columns.
    for idx_name, table in _INDEXES:
        op.drop_index(idx_name, table_name=table)

    # 3. Drop the orphaned FK columns.
    for table, col in _FK_COLS:
        op.drop_column(table, col)

    # 4. Drop the two dormant tables (feedback_events first — it references
    #    task_executions).
    op.drop_table("feedback_events")
    op.drop_table("task_executions")


def downgrade() -> None:
    # Recreate in reverse dependency order: parent table → child table →
    # orphan columns → FK constraints → indexes.

    # ── task_executions ────────────────────────────────────────────────────
    op.create_table(
        "task_executions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("thread_id", sa.Text(), nullable=False),
        sa.Column("goal_text", sa.Text(), nullable=False),
        sa.Column("goal_priority", sa.SmallInteger(), nullable=False),
        sa.Column("strategy", sa.Text(), nullable=False),
        sa.Column("complexity", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("model_used", sa.Text(), nullable=True),
        sa.Column("total_tokens", sa.Integer(), nullable=False),
        sa.Column("total_cost_usd", sa.Numeric(10, 6), nullable=False),
        sa.Column("result_summary", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("extra_data", postgresql.JSONB().with_variant(sa.JSON(), "sqlite"), nullable=False),
        sa.CheckConstraint(
            "goal_priority BETWEEN 1 AND 10", name="check_goal_priority_range"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_task_executions_thread", "task_executions", ["thread_id"])
    op.create_index("idx_task_executions_created", "task_executions", ["created_at"])
    op.create_index("idx_task_executions_status", "task_executions", ["status"])

    # ── feedback_events ────────────────────────────────────────────────────
    op.create_table(
        "feedback_events",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("task_id", sa.Uuid(), nullable=False),
        sa.Column("step_id", sa.Uuid(), nullable=True),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("rating", sa.SmallInteger(), nullable=True),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("rating BETWEEN 1 AND 5", name="check_rating_range"),
        sa.ForeignKeyConstraint(
            ["task_id"], ["task_executions.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["step_id"], ["execution_steps.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_feedback_events_task", "feedback_events", ["task_id", "created_at"]
    )
    op.create_index("idx_feedback_events_type", "feedback_events", ["event_type"])

    # ── orphan columns back onto the surviving tables ──────────────────────
    op.add_column(
        "execution_steps",
        sa.Column("task_id", sa.Uuid(), nullable=True),
    )
    op.add_column(
        "cost_ledger",
        sa.Column("task_id", sa.Uuid(), nullable=True),
    )
    op.add_column(
        "warm_memories",
        sa.Column("source_task_id", sa.Uuid(), nullable=True),
    )
    op.add_column(
        "sub_agent_runs",
        sa.Column("parent_task_id", sa.Uuid(), nullable=True),
    )

    # ── FK constraints back ─────────────────────────────────────────────────
    op.create_foreign_key(
        "execution_steps_task_id_fkey",
        "execution_steps",
        "task_executions",
        ["task_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "cost_ledger_task_id_fkey",
        "cost_ledger",
        "task_executions",
        ["task_id"],
        ["id"],
    )
    op.create_foreign_key(
        "warm_memories_source_task_id_fkey",
        "warm_memories",
        "task_executions",
        ["source_task_id"],
        ["id"],
    )
    op.create_foreign_key(
        "sub_agent_runs_parent_task_id_fkey",
        "sub_agent_runs",
        "task_executions",
        ["parent_task_id"],
        ["id"],
    )

    # ── indexes back ────────────────────────────────────────────────────────
    op.create_index(
        "idx_execution_steps_task_number",
        "execution_steps",
        ["task_id", "step_number"],
    )
    op.create_index(
        "idx_execution_steps_failed",
        "execution_steps",
        ["task_id"],
        postgresql_where=sa.text("status = 'failed'"),
    )
    op.create_index("idx_cost_ledger_task", "cost_ledger", ["task_id", "created_at"])
