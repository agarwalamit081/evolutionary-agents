"""Tests for the ToolVersion review-lifecycle migration (i1b2c3d4e5f6, D10).

``tool_versions`` had no review lifecycle — every persisted version was
implicitly approved, so an operator-edited tool had nowhere to park short of
the active row (which ``load_active_tools`` would immediately materialize,
defeating the review gate). Migration ``i1b2c3d4e5f6`` adds a NOT NULL
``status`` Text column defaulting to ``'approved'`` (values
approved|pending_review|rejected) + a partial index backing the
"latest pending_review version" lookup the approve/reject endpoints run.

These tests run the migration in isolation against an in-memory SQLite DB
seeded with the pre-migration schema, proving: the column is added NOT NULL
with the ``approved`` default, existing rows backfill to ``approved``, the
named index is created, and the column + index are droppable on downgrade.

Note: the index is partial on Postgres (``WHERE status='pending_review'``);
SQLite ignores ``postgresql_where`` and creates a plain index, so the
predicate itself isn't asserted here — only the index's existence + name.
"""

from __future__ import annotations

import datetime
import importlib

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations


MIGRATION_MODULE = "src.db.migrations.versions.i1b2c3d4e5f6_add_tool_version_status"


def _pre_migration_tool_versions() -> sa.Table:
    """tool_versions columns as they exist BEFORE this migration — NO status."""
    metadata = sa.MetaData()
    return sa.Table(
        "tool_versions",
        metadata,
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tool_id", sa.String(36), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("code_content", sa.Text(), nullable=False),
        sa.Column("test_content", sa.Text(), nullable=True),
        sa.Column("test_pass_rate", sa.Float(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
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
        return {
            c["name"]: dict(c) for c in sa.inspect(conn).get_columns("tool_versions")
        }


def _indexes(engine: sa.Engine) -> set[str | None]:
    with engine.connect() as conn:
        return {ix["name"] for ix in sa.inspect(conn).get_indexes("tool_versions")}


@pytest.fixture()
def seeded_engine() -> sa.Engine:
    """An in-memory engine with the pre-migration tool_versions + a row."""
    engine = sa.create_engine("sqlite:///:memory:")
    table = _pre_migration_tool_versions()
    table.create(engine)
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    with engine.begin() as conn:
        conn.execute(
            table.insert(),
            [
                {
                    "id": "v1",
                    "tool_id": "t1",
                    "version": 1,
                    "code_content": "async def h() -> str: return 'ok'",
                    "test_content": None,
                    "test_pass_rate": None,
                    "is_active": True,
                    "created_at": now,
                }
            ],
        )
    return engine


class TestUpgrade:
    def test_adds_status_not_null_with_approved_default(
        self, seeded_engine: sa.Engine
    ) -> None:
        assert "status" not in _columns(seeded_engine)
        _run(seeded_engine, "upgrade")
        col = _columns(seeded_engine).get("status")
        assert col is not None, "status column missing after upgrade"
        assert col["nullable"] is False

    def test_backfills_existing_rows_to_approved(self, seeded_engine: sa.Engine) -> None:
        """The regression guard: legacy rows become ``approved`` (not NULL), so
        ``load_active_tools`` (which requires status='approved') keeps loading
        every pre-existing tool."""
        _run(seeded_engine, "upgrade")
        with seeded_engine.connect() as conn:
            status = conn.execute(
                sa.text("SELECT status FROM tool_versions WHERE id='v1'")
            ).scalar_one()
        assert status == "approved"

    def test_new_insert_defaults_to_approved(self, seeded_engine: sa.Engine) -> None:
        _run(seeded_engine, "upgrade")
        with seeded_engine.begin() as conn:
            conn.execute(
                sa.text(
                    "INSERT INTO tool_versions "
                    "(id, tool_id, version, code_content, is_active, created_at) "
                    "VALUES ('v2', 't1', 2, 'x', 0, :ts)"
                ),
                {"ts": datetime.datetime.now(tz=datetime.timezone.utc)},
            )
        with seeded_engine.connect() as conn:
            status = conn.execute(
                sa.text("SELECT status FROM tool_versions WHERE id='v2'")
            ).scalar_one()
        assert status == "approved"

    def test_creates_pending_index(self, seeded_engine: sa.Engine) -> None:
        _run(seeded_engine, "upgrade")
        assert "idx_tool_versions_pending" in _indexes(seeded_engine)


class TestDowngrade:
    def test_removes_status_and_index(self, seeded_engine: sa.Engine) -> None:
        _run(seeded_engine, "upgrade")
        assert "status" in _columns(seeded_engine)
        assert "idx_tool_versions_pending" in _indexes(seeded_engine)
        _run(seeded_engine, "downgrade")
        assert "status" not in _columns(seeded_engine)
        assert "idx_tool_versions_pending" not in _indexes(seeded_engine)

    def test_upgrade_downgrade_upgrade_round_trip(
        self, seeded_engine: sa.Engine
    ) -> None:
        _run(seeded_engine, "upgrade")
        _run(seeded_engine, "downgrade")
        _run(seeded_engine, "upgrade")
        col = _columns(seeded_engine).get("status")
        assert col is not None and col["nullable"] is False
        assert "idx_tool_versions_pending" in _indexes(seeded_engine)
