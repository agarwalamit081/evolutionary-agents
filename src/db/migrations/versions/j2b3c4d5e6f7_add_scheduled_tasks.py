"""add_scheduled_tasks

Revision ID: j2b3c4d5e6f7
Revises: i1b2c3d4e5f6
Create Date: 2026-06-27 00:00:00.000000

I1 (Phase 5). Adds the ``scheduled_tasks`` table — the durable substrate for
agent-settable durable cron. One row = one future run the agent asked to fire
on a cron schedule via the ``create_scheduled_task`` builtin. The scheduler
consumer (``src.scheduler.cron_consumer``) polls this table and enqueues a
``RunJob`` per fire through the existing ``RunsQueue.enqueue`` seam, so a
scheduled run goes through the real deployed worker stack (lease-lock,
checkpoint, eval-resolution all apply unchanged).

Columns mirror the ORM ``ScheduledTask`` model: id (UUID PK), name (the agent's
stable upsert handle), cron (5-field crontab), goal (the enqueued
``RunJob.goal``), model (optional pin), owner_run_id (provenance), enabled
(whether the consumer fires it), timezone (IANA), next_fire_at (informational
UTC; APScheduler is authoritative), created_at, updated_at. ``name`` carries a
UNIQUE constraint backing the upsert-by-name semantics (a re-call with the same
name revises the schedule instead of duplicating).

Schema-only + fully reversible (drop the table); no data backfill.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = "j2b3c4d5e6f7"
down_revision: Union[str, None] = "i1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "scheduled_tasks",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("cron", sa.Text(), nullable=False),
        sa.Column("goal", sa.Text(), nullable=False),
        sa.Column("model", sa.Text(), nullable=True),
        sa.Column("owner_run_id", sa.Text(), nullable=True),
        sa.Column(
            "enabled",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
        ),
        sa.Column(
            "timezone",
            sa.Text(),
            nullable=False,
            server_default=sa.text("'UTC'"),
        ),
        sa.Column("next_fire_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", name="uq_scheduled_tasks_name"),
    )
    # The consumer's primary read: every enabled row, soonest-first.
    op.create_index(
        "idx_scheduled_tasks_enabled",
        "scheduled_tasks",
        ["enabled", "next_fire_at"],
    )
    # Provenance lookups ("which tasks did run X author?").
    op.create_index(
        "idx_scheduled_tasks_owner",
        "scheduled_tasks",
        ["owner_run_id"],
    )


def downgrade() -> None:
    op.drop_index("idx_scheduled_tasks_owner", table_name="scheduled_tasks")
    op.drop_index("idx_scheduled_tasks_enabled", table_name="scheduled_tasks")
    op.drop_table("scheduled_tasks")
