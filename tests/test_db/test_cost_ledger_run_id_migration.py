"""Tests for the cost_ledger ``run_id`` attribution migration (f7b8c9d0e1f2).

``cost_ledger.task_id`` is a UUID FK into ``task_executions`` (which the agent
never populates), so it was always NULL and per-run cost attribution was
impossible. Migration ``f7b8c9d0e1f2`` adds a free nullable ``run_id`` Text
column — populated by ``CostTracker.record_usage`` from the run's graph
``thread_id`` — plus a supporting index.

These tests run the migration in isolation against an in-memory SQLite DB
seeded with the pre-migration schema, proving: the column is added NULLABLE, the
index is created, an INSERT that sets ``run_id`` succeeds, and the column +
index are droppable on downgrade.
"""

from __future__ import annotations

import datetime
import importlib

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations


MIGRATION_MODULE = "src.db.migrations.versions.f7b8c9d0e1f2_add_run_id_to_cost_ledger"


def _pre_migration_cost_ledger() -> sa.Table:
    """The cost_ledger columns as they exist BEFORE this migration — NO run_id."""
    metadata = sa.MetaData()
    return sa.Table(
        "cost_ledger",
        metadata,
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("task_id", sa.String(36), nullable=True),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("total_tokens", sa.Integer(), nullable=False),
        sa.Column("cached_tokens", sa.Integer(), nullable=False),
        sa.Column("cost_usd", sa.Numeric(10, 6), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )


def _run(engine: sa.Engine, fn_name: str) -> None:
    """Run the migration's upgrade/downgrade against ``engine``."""
    migration = importlib.import_module(MIGRATION_MODULE)
    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            getattr(migration, fn_name)()


def _columns(engine: sa.Engine) -> dict[str, dict[str, object]]:
    with engine.connect() as conn:
        return {c["name"]: dict(c) for c in sa.inspect(conn).get_columns("cost_ledger")}


def _indexes(engine: sa.Engine) -> set[str | None]:
    with engine.connect() as conn:
        return {ix["name"] for ix in sa.inspect(conn).get_indexes("cost_ledger")}


@pytest.fixture()
def seeded_engine() -> sa.Engine:
    """An in-memory engine with the pre-migration cost_ledger + a row."""
    engine = sa.create_engine("sqlite:///:memory:")
    table = _pre_migration_cost_ledger()
    table.create(engine)
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    with engine.begin() as conn:
        conn.execute(
            table.insert(),
            [
                {
                    "id": "r1",
                    "provider": "deepseek",
                    "model": "deepseek-v4-flash",
                    "input_tokens": 50,
                    "output_tokens": 100,
                    "total_tokens": 150,
                    "cached_tokens": 0,
                    "cost_usd": 0.001,
                    "created_at": now,
                }
            ],
        )
    return engine


class TestUpgrade:
    def test_adds_run_id_nullable(self, seeded_engine: sa.Engine) -> None:
        assert "run_id" not in _columns(seeded_engine)
        _run(seeded_engine, "upgrade")
        col = _columns(seeded_engine).get("run_id")
        assert col is not None, "run_id column missing after upgrade"
        assert col["nullable"] is True

    def test_creates_run_index(self, seeded_engine: sa.Engine) -> None:
        _run(seeded_engine, "upgrade")
        assert "idx_cost_ledger_run" in _indexes(seeded_engine)

    def test_existing_rows_get_null_run_id(self, seeded_engine: sa.Engine) -> None:
        """Legacy rows carry NULL run_id — no backfill is possible or attempted."""
        _run(seeded_engine, "upgrade")
        with seeded_engine.connect() as conn:
            run_id = conn.execute(
                sa.text("SELECT run_id FROM cost_ledger WHERE id='r1'")
            ).scalar_one()
        assert run_id is None

    def test_insert_with_run_id_succeeds_after_upgrade(
        self, seeded_engine: sa.Engine
    ) -> None:
        """The fix: a cost row carrying run_id now lands and is attributable."""
        _run(seeded_engine, "upgrade")
        with seeded_engine.begin() as conn:
            conn.execute(
                sa.text(
                    "INSERT INTO cost_ledger (id, provider, model, input_tokens, "
                    "output_tokens, total_tokens, cached_tokens, cost_usd, "
                    "created_at, run_id) "
                    "VALUES ('r2', 'zai', 'glm-5.1', 5, 7, 12, 0, 0.0002, :ts, :rid)"
                ),
                {
                    "ts": datetime.datetime.now(tz=datetime.timezone.utc),
                    "rid": "cli-q05",
                },
            )
        with seeded_engine.connect() as conn:
            run_id = conn.execute(
                sa.text("SELECT run_id FROM cost_ledger WHERE id='r2'")
            ).scalar_one()
        assert run_id == "cli-q05"


class TestDowngrade:
    def test_removes_run_id_and_index(self, seeded_engine: sa.Engine) -> None:
        _run(seeded_engine, "upgrade")
        assert "run_id" in _columns(seeded_engine)
        assert "idx_cost_ledger_run" in _indexes(seeded_engine)
        _run(seeded_engine, "downgrade")
        assert "run_id" not in _columns(seeded_engine)
        assert "idx_cost_ledger_run" not in _indexes(seeded_engine)

    def test_upgrade_downgrade_upgrade_round_trip(
        self, seeded_engine: sa.Engine
    ) -> None:
        _run(seeded_engine, "upgrade")
        _run(seeded_engine, "downgrade")
        _run(seeded_engine, "upgrade")
        col = _columns(seeded_engine).get("run_id")
        assert col is not None and col["nullable"] is True
        assert "idx_cost_ledger_run" in _indexes(seeded_engine)
