"""add_sub_agent_tables

Revision ID: a1b2c3d4e5f6
Revises: 6f3695ef31c6
Create Date: 2026-06-09 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = "6f3695ef31c6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── Sub-Agent Definitions ────────────────────────────────────────────
    op.create_table(
        "sub_agent_definitions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        # Configuration
        sa.Column("template_type", sa.Text(), nullable=False, server_default="fixed"),
        sa.Column("tool_scope", sa.Text(), nullable=False, server_default="inherit_all"),
        sa.Column(
            "tool_subset",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="[]",
        ),
        sa.Column("budget_mode", sa.Text(), nullable=False, server_default="shared"),
        sa.Column(
            "budget_limit",
            sa.Numeric(precision=10, scale=6),
            nullable=False,
            server_default="0",
        ),
        sa.Column("model_tier", sa.Text(), nullable=False, server_default="simple"),
        sa.Column("max_iterations", sa.Integer(), nullable=False, server_default="10"),
        sa.Column("depth_limit", sa.Integer(), nullable=False, server_default="0"),
        # Custom config
        sa.Column(
            "node_config",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column("system_prompt_override", sa.Text(), nullable=True),
        # Rolling metrics
        sa.Column("total_runs", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("success_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("success_rate", sa.Float(), nullable=False, server_default="0"),
        sa.Column(
            "avg_cost",
            sa.Numeric(precision=10, scale=6),
            nullable=False,
            server_default="0",
        ),
        sa.Column("avg_latency_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("quality_score", sa.Float(), nullable=False, server_default="0.5"),
        # Lifecycle
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column(
            "source_mutation_id", sa.Uuid(), sa.ForeignKey("mutations.id"), nullable=True
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
        sa.CheckConstraint("success_rate BETWEEN 0 AND 1", name="check_sa_success_rate"),
        sa.CheckConstraint("quality_score BETWEEN 0 AND 1", name="check_sa_quality_score"),
    )
    op.create_index(
        "idx_sub_agents_active",
        "sub_agent_definitions",
        ["name"],
        postgresql_where=sa.text("is_active = true"),
    )
    op.create_index(
        "idx_sub_agents_template",
        "sub_agent_definitions",
        ["template_type"],
    )
    op.create_index(
        "idx_sub_agents_performance",
        "sub_agent_definitions",
        ["success_rate"],
        postgresql_where=sa.text("is_active = true"),
    )

    # ── Sub-Agent Runs ──────────────────────────────────────────────────
    op.create_table(
        "sub_agent_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "sub_agent_id",
            sa.Uuid(),
            sa.ForeignKey("sub_agent_definitions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "parent_task_id",
            sa.Uuid(),
            sa.ForeignKey("task_executions.id"),
            nullable=True,
        ),
        sa.Column("parent_thread_id", sa.Text(), nullable=False),
        # I/O
        sa.Column("goal_text", sa.Text(), nullable=False),
        sa.Column("result_summary", sa.Text(), nullable=True),
        # Metrics
        sa.Column("status", sa.Text(), nullable=False, server_default="pending"),
        sa.Column("iterations_used", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("tokens_used", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "cost_usd",
            sa.Numeric(precision=10, scale=6),
            nullable=False,
            server_default="0",
        ),
        sa.Column("latency_ms", sa.Integer(), nullable=False, server_default="0"),
        # Quality
        sa.Column("quality_rating", sa.Float(), nullable=True),
        sa.Column(
            "extra_data",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_sub_agent_runs_agent",
        "sub_agent_runs",
        ["sub_agent_id", "created_at"],
    )
    op.create_index(
        "idx_sub_agent_runs_parent",
        "sub_agent_runs",
        ["parent_thread_id"],
    )
    op.create_index(
        "idx_sub_agent_runs_status",
        "sub_agent_runs",
        ["status"],
    )


def downgrade() -> None:
    op.drop_table("sub_agent_runs")
    op.drop_table("sub_agent_definitions")
