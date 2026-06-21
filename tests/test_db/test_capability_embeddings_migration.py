"""Tests for the ``capability_embeddings`` migration (B3 semantic dedup).

Migration ``c4e5a6b7c8d9`` adds a nullable ``Vector(768)`` capability embedding
(+ ``capability_text``) and an HNSW cosine index to ``tool_registrations`` and
``sub_agent_definitions``. This is the storage backing for semantic dedup /
consolidation: before creating a tool/sub-agent the agent embeds the capability
gap and cosine-searches these indexes to reuse a semantically identical one.

This migration uses pgvector-specific DDL (``Vector`` columns + HNSW indexes),
so it **cannot** be exercised on the in-memory SQLite that the pure-column
migration tests use. PostgreSQL + pgvector is the project's sole DB, so these
tests run against a real pgvector instance: a throwaway scratch database is
created, the migration's ``upgrade``/``downgrade`` are applied in isolation on
top of a minimal table shell, and the scratch database is dropped afterwards.
The whole module is skipped when the pgvector container is unreachable (so it
stays green in CI / offline without forcing pgvector).
"""

from __future__ import annotations

import importlib
import uuid

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

MIGRATION_MODULE = "src.db.migrations.versions.c4e5a6b7c8d9_add_capability_embeddings"

# Standard dev stack container (CLAUDE.md): pgvector on host port 5433,
# postgres/changeme. Override via env for non-default deployments.
_PG_HOST = "localhost"
_PG_PORT = 5433
_PG_USER = "postgres"
_PG_PASSWORD = "changeme"


def _pg_available() -> bool:
    """True iff a connectable pgvector instance answers on the dev port."""
    try:
        import psycopg2  # type: ignore[import-not-found]
    except Exception:
        return False
    try:
        conn = psycopg2.connect(
            host=_PG_HOST,
            port=_PG_PORT,
            user=_PG_USER,
            password=_PG_PASSWORD,
            dbname="postgres",
            connect_timeout=2,
        )
        conn.close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _pg_available(), reason="pgvector container not reachable on localhost:5433"
)


@pytest.fixture()
def scratch_engine():
    """Create a scratch pgvector DB, install the extension, yield a sync engine,
    then drop the scratch DB. Leaves the dev database untouched."""
    import psycopg2  # type: ignore[import-not-found]

    scratch = f"turing_agent_captest_{uuid.uuid4().hex[:12]}"
    admin = psycopg2.connect(
        host=_PG_HOST,
        port=_PG_PORT,
        user=_PG_USER,
        password=_PG_PASSWORD,
        dbname="postgres",
    )
    admin.autocommit = True
    admin.cursor().execute(f'CREATE DATABASE "{scratch}"')

    # Install pgvector in the scratch DB (vector types/ops + HNSW access method).
    ext_conn = psycopg2.connect(
        host=_PG_HOST,
        port=_PG_PORT,
        user=_PG_USER,
        password=_PG_PASSWORD,
        dbname=scratch,
    )
    ext_conn.autocommit = True
    ext_conn.cursor().execute("CREATE EXTENSION IF NOT EXISTS vector")
    ext_conn.close()

    url = f"postgresql+psycopg2://{_PG_USER}:{_PG_PASSWORD}@{_PG_HOST}:{_PG_PORT}/{scratch}"
    engine = sa.create_engine(url)

    # Minimal pre-migration shell: the two tables exist with just a PK. The
    # migration only ADDS the capability columns + index, so a shell is enough
    # to exercise add_column / create_index against pgvector.
    with engine.begin() as conn:
        conn.execute(
            sa.text("CREATE TABLE tool_registrations (id UUID PRIMARY KEY)")
        )
        conn.execute(
            sa.text("CREATE TABLE sub_agent_definitions (id UUID PRIMARY KEY)")
        )

    try:
        yield engine
    finally:
        engine.dispose()
        admin.cursor().execute(f'DROP DATABASE IF EXISTS "{scratch}"')
        admin.close()


def _run(engine: sa.Engine, fn_name: str) -> None:
    """Run the migration's upgrade/downgrade against ``engine``."""
    migration = importlib.import_module(MIGRATION_MODULE)
    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            getattr(migration, fn_name)()


def _columns(engine: sa.Engine, table: str) -> dict[str, str]:
    # ``udt_name`` holds the base type name — pgvector columns report
    # ``data_type = USER-DEFINED`` but ``udt_name = vector``.
    with engine.connect() as conn:
        rows = conn.execute(
            sa.text(
                "SELECT column_name, udt_name FROM information_schema.columns "
                "WHERE table_name = :t"
            ),
            {"t": table},
        ).fetchall()
    return {name: udt for name, udt in rows}


def _index_def(engine: sa.Engine, index_name: str) -> str | None:
    with engine.connect() as conn:
        row = conn.execute(
            sa.text("SELECT indexdef FROM pg_indexes WHERE indexname = :n"),
            {"n": index_name},
        ).fetchone()
    return row[0] if row else None


class TestUpgrade:
    def test_adds_capability_columns_to_tool_registrations(
        self, scratch_engine: sa.Engine
    ) -> None:
        assert "capability_embedding" not in _columns(scratch_engine, "tool_registrations")
        _run(scratch_engine, "upgrade")
        cols = _columns(scratch_engine, "tool_registrations")
        assert "capability_embedding" in cols
        assert "capability_text" in cols
        # pgvector reports the user type name as "vector" (with the dimension
        # as a type modifier, not a separate data_type).
        assert cols["capability_embedding"] == "vector"

    def test_adds_capability_columns_to_sub_agent_definitions(
        self, scratch_engine: sa.Engine
    ) -> None:
        _run(scratch_engine, "upgrade")
        cols = _columns(scratch_engine, "sub_agent_definitions")
        assert "capability_embedding" in cols
        assert "capability_text" in cols
        assert cols["capability_embedding"] == "vector"

    def test_creates_hnsw_cosine_indexes(self, scratch_engine: sa.Engine) -> None:
        _run(scratch_engine, "upgrade")
        tool_idx = _index_def(scratch_engine, "idx_tool_registrations_capability_emb")
        agent_idx = _index_def(scratch_engine, "idx_sub_agent_capability_emb")
        assert tool_idx is not None and "USING hnsw" in tool_idx
        assert "vector_cosine_ops" in tool_idx
        assert agent_idx is not None and "USING hnsw" in agent_idx
        assert "vector_cosine_ops" in agent_idx


class TestDowngrade:
    def test_removes_capability_columns_and_indexes(
        self, scratch_engine: sa.Engine
    ) -> None:
        _run(scratch_engine, "upgrade")
        assert "capability_embedding" in _columns(scratch_engine, "tool_registrations")

        _run(scratch_engine, "downgrade")
        for table in ("tool_registrations", "sub_agent_definitions"):
            cols = _columns(scratch_engine, table)
            assert "capability_embedding" not in cols
            assert "capability_text" not in cols
        assert _index_def(scratch_engine, "idx_tool_registrations_capability_emb") is None
        assert _index_def(scratch_engine, "idx_sub_agent_capability_emb") is None

    def test_upgrade_downgrade_upgrade_round_trip(
        self, scratch_engine: sa.Engine
    ) -> None:
        _run(scratch_engine, "upgrade")
        _run(scratch_engine, "downgrade")
        _run(scratch_engine, "upgrade")
        cols = _columns(scratch_engine, "tool_registrations")
        assert cols["capability_embedding"] == "vector"
        assert _index_def(
            scratch_engine, "idx_tool_registrations_capability_emb"
        ) is not None
        assert _index_def(
            scratch_engine, "idx_sub_agent_capability_emb"
        ) is not None
