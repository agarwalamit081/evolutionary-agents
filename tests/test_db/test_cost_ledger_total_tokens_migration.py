"""Tests for the cost_ledger ``total_tokens`` reconciliation migration (F10.1).

The initial schema migration (6f3695ef31c6) created ``cost_ledger`` WITHOUT a
``total_tokens`` column, while the ORM (``models.CostLedger``) and
``CostTracker.record_usage`` both require it. Every cost INSERT therefore
failed silently, leaving ``cost_ledger`` empty. Migration
``73a8b0323eb3`` reconciles the table with the ORM by adding the column.

These tests run the migration in isolation against an in-memory SQLite DB
seeded with the pre-migration schema, proving: the column is added NOT NULL,
existing rows are backfilled with ``input_tokens + output_tokens``, the
column is droppable on downgrade, and (the actual fix) an INSERT that sets
``total_tokens`` now succeeds.
"""

from __future__ import annotations

import datetime
import importlib

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations


MIGRATION_MODULE = "src.db.migrations.versions.73a8b0323eb3_add_total_tokens_to_cost_ledger"


def _pre_migration_cost_ledger() -> sa.Table:
    """The cost_ledger columns the initial schema created — NO total_tokens."""
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


@pytest.fixture()
def seeded_engine() -> sa.Engine:
    """An in-memory engine with the pre-migration cost_ledger + two rows."""
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
                    "provider": "openai",
                    "model": "gpt-4o-mini-2024-07-18",
                    "input_tokens": 50,
                    "output_tokens": 100,
                    "cached_tokens": 0,
                    "cost_usd": 0.001,
                    "created_at": now,
                },
                {
                    "id": "r2",
                    "provider": "openai",
                    "model": "gpt-4o-mini-2024-07-18",
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "cached_tokens": 0,
                    "cost_usd": 0.0005,
                    "created_at": now,
                },
            ],
        )
    return engine


class TestUpgrade:
    def test_adds_total_tokens_not_null(self, seeded_engine: sa.Engine) -> None:
        assert "total_tokens" not in _columns(seeded_engine)
        _run(seeded_engine, "upgrade")
        col = _columns(seeded_engine).get("total_tokens")
        assert col is not None, "total_tokens column missing after upgrade"
        assert col["nullable"] is False

    def test_backfills_existing_rows_with_input_plus_output(
        self, seeded_engine: sa.Engine
    ) -> None:
        _run(seeded_engine, "upgrade")
        with seeded_engine.connect() as conn:
            rows = conn.execute(
                sa.text("SELECT id, total_tokens FROM cost_ledger ORDER BY id")
            ).fetchall()
        assert rows == [("r1", 150), ("r2", 30)]  # 50+100, 10+20

    def test_insert_with_total_tokens_succeeds_after_upgrade(
        self, seeded_engine: sa.Engine
    ) -> None:
        """The F10.1 fix: a cost row carrying total_tokens now lands."""
        _run(seeded_engine, "upgrade")
        with seeded_engine.begin() as conn:
            conn.execute(
                sa.text(
                    "INSERT INTO cost_ledger (id, provider, model, input_tokens, "
                    "output_tokens, cached_tokens, total_tokens, cost_usd, created_at) "
                    "VALUES ('r3', 'openai', 'gpt-4o-mini', 5, 7, 0, 12, 0.001, :ts)"
                ),
                {"ts": datetime.datetime.now(tz=datetime.timezone.utc)},
            )
        with seeded_engine.connect() as conn:
            total = conn.execute(
                sa.text("SELECT total_tokens FROM cost_ledger WHERE id='r3'")
            ).scalar_one()
        assert total == 12


class TestDowngrade:
    def test_removes_total_tokens(self, seeded_engine: sa.Engine) -> None:
        _run(seeded_engine, "upgrade")
        assert "total_tokens" in _columns(seeded_engine)
        _run(seeded_engine, "downgrade")
        assert "total_tokens" not in _columns(seeded_engine)

    def test_upgrade_downgrade_upgrade_round_trip(
        self, seeded_engine: sa.Engine
    ) -> None:
        _run(seeded_engine, "upgrade")
        _run(seeded_engine, "downgrade")
        _run(seeded_engine, "upgrade")
        col = _columns(seeded_engine).get("total_tokens")
        assert col is not None and col["nullable"] is False
