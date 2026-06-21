"""Real-Postgres integration tests for ``find_similar`` (B3 semantic dedup).

The unit tests in ``tests/test_tools/test_dynamic/test_persister_dedup.py`` and
``tests/test_agents/test_persister_dedup.py`` verify the post-query contract
(threshold math, error degradation) with a mocked session. These tests exercise
the parts only a real pgvector instance can validate: the ``cosine_distance``
operator, the HNSW index, and the ``capability_embedding IS NOT NULL`` /
``is_active`` filters — against actual inserted vectors.

A throwaway scratch pgvector database is created, seeded with unit vectors, and
dropped afterwards; the app's ``get_session`` is pointed at it. Skipped when the
pgvector container is unreachable.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import patch

import pytest
import pytest_asyncio
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

_PG_HOST = "localhost"
_PG_PORT = 5433
_PG_USER = "postgres"
_PG_PASSWORD = "changeme"


def _pg_available() -> bool:
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


def _unit_vec(dim: int, idx: int) -> list[float]:
    """A unit vector with a single 1.0 at ``idx`` (orthogonal basis vector)."""
    v = [0.0] * dim
    v[idx] = 1.0
    return v


def _vec_literal(vec: list[float]) -> str:
    """pgvector text literal, e.g. ``[1.0,0.0,...]``."""
    return "[" + ",".join(repr(x) for x in vec) + "]"


def _scratch_get_session(sessionmaker: Any):  # type: ignore[no-untyped-def]
    """Return a get_session-equivalent bound to the scratch sessionmaker."""
    @asynccontextmanager
    async def _getter():  # type: ignore[no-untyped-def]
        async with sessionmaker() as session:
            yield session

    return _getter


@pytest_asyncio.fixture()
async def seeded_sessionmaker():
    """Scratch pgvector DB with tool/sub-agent capability rows; yields an
    async sessionmaker bound to it. Torn down in finally."""
    import psycopg2  # type: ignore[import-not-found]

    scratch = f"turing_agent_simtest_{uuid.uuid4().hex[:12]}"
    admin = psycopg2.connect(
        host=_PG_HOST, port=_PG_PORT, user=_PG_USER, password=_PG_PASSWORD, dbname="postgres"
    )
    admin.autocommit = True
    admin.cursor().execute(f'CREATE DATABASE "{scratch}"')

    ext_conn = psycopg2.connect(
        host=_PG_HOST, port=_PG_PORT, user=_PG_USER, password=_PG_PASSWORD, dbname=scratch
    )
    ext_conn.autocommit = True
    ext_conn.cursor().execute("CREATE EXTENSION IF NOT EXISTS vector")
    ext_conn.close()

    url = f"postgresql+asyncpg://{_PG_USER}:{_PG_PASSWORD}@{_PG_HOST}:{_PG_PORT}/{scratch}"
    engine = create_async_engine(url)

    async with engine.begin() as conn:
        # Minimal tables matching what find_similar reads. tool_name vs name
        # mirrors the two ORM models (ToolRegistration vs SubAgentModel).
        await conn.execute(
            sa.text(
                "CREATE TABLE tool_registrations ("
                "  id UUID PRIMARY KEY,"
                "  tool_name TEXT,"
                "  description TEXT,"
                "  capability_embedding vector(768),"
                "  is_active BOOLEAN DEFAULT TRUE)"
            )
        )
        await conn.execute(
            sa.text(
                "CREATE TABLE sub_agent_definitions ("
                "  id UUID PRIMARY KEY,"
                "  name TEXT,"
                "  description TEXT,"
                "  capability_embedding vector(768),"
                "  is_active BOOLEAN DEFAULT TRUE)"
            )
        )
        await conn.execute(
            sa.text(
                "CREATE INDEX idx_tool_registrations_capability_emb "
                "ON tool_registrations USING hnsw (capability_embedding vector_cosine_ops)"
            )
        )

    e0, e1 = _unit_vec(768, 0), _unit_vec(768, 1)
    async with engine.begin() as conn:
        # Each table: an active match (e0), an active non-match (e1), a
        # NULL-embedding row (filtered), and an inactive match (filtered).
        rows = [
            ("tool_registrations", "tool_name", "fetcher_a", "HTTP fetcher", e0, True),
            ("tool_registrations", "tool_name", "different_b", "Duplicate finder", e1, True),
            ("tool_registrations", "tool_name", "null_c", "No embedding", None, True),
            ("tool_registrations", "tool_name", "inactive_d", "Inactive fetcher", e0, False),
            ("sub_agent_definitions", "name", "analyzer_a", "Data analyzer", e0, True),
            ("sub_agent_definitions", "name", "reporter_b", "Report writer", e1, True),
            ("sub_agent_definitions", "name", "null_agent", "No embedding", None, True),
            ("sub_agent_definitions", "name", "inactive_agent", "Inactive analyzer", e0, False),
        ]
        for table, name_col, name, desc, emb, active in rows:
            if emb is None:
                await conn.execute(
                    sa.text(
                        f"INSERT INTO {table} (id, {name_col}, description, is_active) "
                        "VALUES (:id, :n, :d, :a)"
                    ),
                    {"id": str(uuid.uuid4()), "n": name, "d": desc, "a": active},
                )
            else:
                await conn.execute(
                    sa.text(
                        f"INSERT INTO {table} (id, {name_col}, description, "
                        f"capability_embedding, is_active) "
                        "VALUES (:id, :n, :d, CAST(:emb AS vector), :a)"
                    ),
                    {
                        "id": str(uuid.uuid4()),
                        "n": name,
                        "d": desc,
                        "emb": _vec_literal(emb),
                        "a": active,
                    },
                )

    SessionLocal = async_sessionmaker(engine, expire_on_commit=False)
    try:
        yield SessionLocal
    finally:
        await engine.dispose()
        admin.cursor().execute(f'DROP DATABASE IF EXISTS "{scratch}"')
        admin.close()


class TestToolFindSimilarIntegration:
    @pytest.mark.asyncio
    async def test_returns_only_active_above_threshold(
        self, seeded_sessionmaker: Any
    ) -> None:
        from src.tools.dynamic.persister import ToolPersister

        e0 = _unit_vec(768, 0)
        with patch(
            "src.db.session.get_session",
            new=_scratch_get_session(seeded_sessionmaker),
        ):
            matches = await ToolPersister().find_similar(e0, threshold=0.85)

        names = [m["tool_name"] for m in matches]
        assert names == ["fetcher_a"]
        assert matches[0]["similarity"] == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.asyncio
    async def test_different_query_matches_different_row(
        self, seeded_sessionmaker: Any
    ) -> None:
        from src.tools.dynamic.persister import ToolPersister

        e1 = _unit_vec(768, 1)
        with patch(
            "src.db.session.get_session",
            new=_scratch_get_session(seeded_sessionmaker),
        ):
            matches = await ToolPersister().find_similar(e1, threshold=0.85)
        assert [m["tool_name"] for m in matches] == ["different_b"]


class TestSubAgentFindSimilarIntegration:
    @pytest.mark.asyncio
    async def test_returns_only_active_above_threshold(
        self, seeded_sessionmaker: Any
    ) -> None:
        from src.agents.persister import SubAgentPersister

        e0 = _unit_vec(768, 0)
        with patch(
            "src.db.session.get_session",
            new=_scratch_get_session(seeded_sessionmaker),
        ):
            matches = await SubAgentPersister().find_similar(e0, threshold=0.85)
        assert [m["name"] for m in matches] == ["analyzer_a"]
        assert matches[0]["similarity"] == pytest.approx(1.0, abs=1e-6)
