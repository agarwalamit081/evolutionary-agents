"""Tests for the Track-1 attribution migration (l4d5e6f7a8b9).

Adds three per-run attribution columns + relaxes one NOT NULL so run_metrics
can attribute tools/subagents/node-timings to the run that produced them
without fuzzy time-window joins:

- ``tool_registrations.owner_run_id`` (Text, nullable)
- ``sub_agent_definitions.owner_run_id`` (Text, nullable)
- ``execution_steps.run_id`` (Text, nullable)
- ``execution_steps.task_id`` relaxed from NOT NULL → nullable (timing-only rows
  carry no ``task_executions`` parent)
- ``idx_tool_call_metrics_run`` on ``tool_call_metrics(run_id, created_at)`` — the
  ``run_id`` column itself predates this migration (added by ``e6a7b8c9d0e1``).

Runs the migration in isolation against an in-memory SQLite DB seeded with the
pre-migration schema, proving the columns are added nullable, task_id is
relaxed, the named index is created, and everything reverses on downgrade.

Note: the nullable ``task_id`` relaxation is the only ``alter_column`` in the
repo's migration set; alembic renders it as a batch table-rebuild on SQLite, so
the test asserts the post-upgrade nullable flag rather than the DDL shape.
"""

from __future__ import annotations

import importlib

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations


MIGRATION_MODULE = "src.db.migrations.versions.l4d5e6f7a8b9_add_owner_run_id_attribution"


def _pre_migration_schema() -> sa.MetaData:
    """The four tables as they exist BEFORE this migration (no attribution cols,
    execution_steps.task_id NOT NULL, no tool_call_metrics_run index)."""
    metadata = sa.MetaData()
    sa.Table(
        "execution_steps",
        metadata,
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("task_id", sa.Uuid(), nullable=False),  # relaxed by this migration
        sa.Column("step_number", sa.Integer(), nullable=False),
        sa.Column("phase", sa.Text(), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    sa.Table(
        "tool_registrations",
        metadata,
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("tool_name", sa.Text(), nullable=False),
        sa.Column("source_mutation_id", sa.Uuid(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    sa.Table(
        "sub_agent_definitions",
        metadata,
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("source_mutation_id", sa.Uuid(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    sa.Table(
        "tool_call_metrics",
        metadata,
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("tool_name", sa.Text(), nullable=False),
        # run_id predates this migration (added by e6a7b8c9d0e1).
        sa.Column("run_id", sa.Text(), nullable=True),
        sa.Column("success", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    return metadata


def _run(engine: sa.Engine, fn_name: str) -> None:
    """Run the migration's upgrade/downgrade against ``engine``."""
    migration = importlib.import_module(MIGRATION_MODULE)
    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            getattr(migration, fn_name)()


def _columns(engine: sa.Engine, table: str) -> dict[str, dict[str, object]]:
    with engine.connect() as conn:
        return {c["name"]: dict(c) for c in sa.inspect(conn).get_columns(table)}


def _indexes(engine: sa.Engine, table: str) -> set[str | None]:
    with engine.connect() as conn:
        return {ix["name"] for ix in sa.inspect(conn).get_indexes(table)}


@pytest.fixture()
def seeded_engine() -> sa.Engine:
    """An in-memory engine with the pre-migration schema created."""
    engine = sa.create_engine("sqlite:///:memory:")
    _pre_migration_schema().create_all(engine)
    return engine


class TestUpgrade:
    def test_adds_owner_run_id_to_tool_registrations(
        self, seeded_engine: sa.Engine
    ) -> None:
        assert "owner_run_id" not in _columns(seeded_engine, "tool_registrations")
        _run(seeded_engine, "upgrade")
        col = _columns(seeded_engine, "tool_registrations").get("owner_run_id")
        assert col is not None, "owner_run_id missing on tool_registrations"
        assert col["nullable"] is True

    def test_adds_owner_run_id_to_sub_agent_definitions(
        self, seeded_engine: sa.Engine
    ) -> None:
        _run(seeded_engine, "upgrade")
        col = _columns(seeded_engine, "sub_agent_definitions").get("owner_run_id")
        assert col is not None, "owner_run_id missing on sub_agent_definitions"
        assert col["nullable"] is True

    def test_adds_run_id_to_execution_steps(self, seeded_engine: sa.Engine) -> None:
        _run(seeded_engine, "upgrade")
        col = _columns(seeded_engine, "execution_steps").get("run_id")
        assert col is not None, "run_id missing on execution_steps"
        assert col["nullable"] is True

    def test_relaxes_execution_steps_task_id_to_nullable(
        self, seeded_engine: sa.Engine
    ) -> None:
        # Pre-migration: task_id is NOT NULL.
        assert _columns(seeded_engine, "execution_steps")["task_id"]["nullable"] is False
        _run(seeded_engine, "upgrade")
        # Post-upgrade: a timing-only row with task_id=NULL can now be written.
        col = _columns(seeded_engine, "execution_steps")["task_id"]
        assert col["nullable"] is True
        with seeded_engine.begin() as conn:
            conn.execute(
                sa.text(
                    "INSERT INTO execution_steps "
                    "(id, step_number, phase, duration_ms, status, created_at) "
                    "VALUES ('00000000-0000-0000-0000-000000000001', 0, "
                    "'execute', 42, 'completed', '2026-07-13T00:00:00+00:00')"
                )
            )

    def test_creates_tool_call_metrics_run_index(
        self, seeded_engine: sa.Engine
    ) -> None:
        _run(seeded_engine, "upgrade")
        assert "idx_tool_call_metrics_run" in _indexes(
            seeded_engine, "tool_call_metrics"
        )


class TestDowngrade:
    def test_drops_columns_and_index(self, seeded_engine: sa.Engine) -> None:
        _run(seeded_engine, "upgrade")
        assert "owner_run_id" in _columns(seeded_engine, "tool_registrations")
        assert "owner_run_id" in _columns(seeded_engine, "sub_agent_definitions")
        assert "run_id" in _columns(seeded_engine, "execution_steps")
        assert "idx_tool_call_metrics_run" in _indexes(
            seeded_engine, "tool_call_metrics"
        )
        _run(seeded_engine, "downgrade")
        assert "owner_run_id" not in _columns(seeded_engine, "tool_registrations")
        assert "owner_run_id" not in _columns(seeded_engine, "sub_agent_definitions")
        assert "run_id" not in _columns(seeded_engine, "execution_steps")
        assert "idx_tool_call_metrics_run" not in _indexes(
            seeded_engine, "tool_call_metrics"
        )

    def test_downgrade_restores_task_id_not_null(
        self, seeded_engine: sa.Engine
    ) -> None:
        _run(seeded_engine, "upgrade")
        assert (
            _columns(seeded_engine, "execution_steps")["task_id"]["nullable"] is True
        )
        _run(seeded_engine, "downgrade")
        # The downgrade deletes timing-only rows then reapplies NOT NULL.
        assert (
            _columns(seeded_engine, "execution_steps")["task_id"]["nullable"] is False
        )

    def test_upgrade_downgrade_upgrade_round_trip(
        self, seeded_engine: sa.Engine
    ) -> None:
        _run(seeded_engine, "upgrade")
        _run(seeded_engine, "downgrade")
        _run(seeded_engine, "upgrade")
        assert "owner_run_id" in _columns(seeded_engine, "tool_registrations")
        assert "run_id" in _columns(seeded_engine, "execution_steps")
        assert "idx_tool_call_metrics_run" in _indexes(
            seeded_engine, "tool_call_metrics"
        )
