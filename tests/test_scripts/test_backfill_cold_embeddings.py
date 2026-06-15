"""Unit tests for scripts/backfill_cold_embeddings.py — the NULL-embedding backfill.

These test the backfill's *logic* hermetically. The DB is an external dependency,
so the async session is mocked (the same convention the rest of the memory test
suite uses). We do NOT spin up an in-memory SQLite table: the ``cold_memories``
model carries pgvector ``Vector(768)`` + ``JSONB`` columns that are
Postgres-only, so a real SQLite table cannot host it. The mock exercises the real
script code — the SELECT NULL-filter, per-row embedding, per-row UPDATE, and
idempotency — without any provider key or infrastructure.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

# Load the standalone script (it lives in scripts/, not the src package).
# Register it in sys.modules BEFORE exec: the script's @dataclass decorator does
# sys.modules.get(cls.__module__) to resolve defaults, which returns None (and
# crashes) unless the module is keyed under its __name__ during execution.
_SCRIPT_NAME = "backfill_cold_embeddings"
_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "backfill_cold_embeddings.py"
_spec = importlib.util.spec_from_file_location(_SCRIPT_NAME, _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
backfill = importlib.util.module_from_spec(_spec)
sys.modules[_SCRIPT_NAME] = backfill
_spec.loader.exec_module(backfill)


_DIM = 768


def _row(rid: str, content: str) -> SimpleNamespace:
    return SimpleNamespace(id=rid, content=content)


def _mock_session(rows: list[SimpleNamespace]) -> MagicMock:
    """A session whose first execute() (the SELECT) yields `rows` via scalars().all()."""
    session = MagicMock()
    select_result = MagicMock()
    select_result.scalars.return_value.all.return_value = rows
    session.execute = AsyncMock(return_value=select_result)
    session.commit = AsyncMock()
    return session


def _gen_with_mapping(content_to_vec: dict[str, list[float]]) -> MagicMock:
    """A generator whose generate(content) returns the mapped vector."""
    gen = MagicMock()
    gen.generate = AsyncMock(side_effect=lambda c: content_to_vec[c])
    gen.model = "text-embedding-3-small"
    gen.dimension = _DIM
    return gen


# ─── _fetch_null_rows ────────────────────────────────────────────────


class TestFetchNullRows:
    """The backfill set is exactly the rows whose embedding is NULL."""

    @pytest.mark.asyncio
    async def test_select_filters_null_embeddings(self) -> None:
        """_fetch_null_rows issues a SELECT filtered on embedding IS NULL."""
        session = _mock_session([])

        await backfill._fetch_null_rows(session)

        session.execute.assert_awaited_once()
        stmt = session.execute.call_args.args[0]
        where = stmt.whereclause
        assert where is not None
        # IS NULL is a null check — no vector value bound, so this compiles safely.
        assert "embedding IS NULL" in str(where.compile())

    @pytest.mark.asyncio
    async def test_returns_rows_from_scalars_all(self) -> None:
        """The rows surfaced by scalars().all() are returned as a list."""
        rows = [_row("a", "alpha"), _row("b", "beta")]
        session = _mock_session(rows)

        out = await backfill._fetch_null_rows(session)

        assert out == rows


# ─── Core backfill behavior ──────────────────────────────────────────


class TestBackfillMissingEmbeddings:
    """NULL rows get vectors, non-NULL rows are skipped, re-runs are no-ops."""

    @pytest.mark.asyncio
    async def test_null_rows_get_vectors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each NULL row is embedded on its content and updated with that vector."""
        rows = [_row("r1", "alpha"), _row("r2", "beta")]
        vecs = {
            "alpha": [0.11] * _DIM,
            "beta": [0.22] * _DIM,
        }
        session = _mock_session(rows)
        gen = _gen_with_mapping(vecs)

        # Spy on _apply_embedding to capture (id, vector) without SQL introspection.
        applied: list[tuple[object, list[float]]] = []

        async def _spy(_session: object, mid: object, vec: list[float]) -> None:
            applied.append((mid, vec))

        monkeypatch.setattr(backfill, "_apply_embedding", _spy)

        stats = await backfill.backfill_missing_embeddings(session, gen)

        # Each row's content was embedded exactly once.
        assert gen.generate.await_count == 2
        gen.generate.assert_any_await("alpha")
        gen.generate.assert_any_await("beta")
        # The right vector reached the right row, in row order.
        assert applied == [("r1", vecs["alpha"]), ("r2", vecs["beta"])]
        # Committed once after applying all vectors.
        session.commit.assert_awaited_once()
        assert stats.scanned == 2
        assert stats.embedded == 2
        assert stats.failed == 0

    @pytest.mark.asyncio
    async def test_non_null_rows_are_skipped(self) -> None:
        """Only the rows the fetch returns are processed — never more.

        ``_fetch_null_rows`` only ever returns NULL-embedding rows, so non-NULL
        rows never reach the generator. With 2 NULL rows surfaced, generate is
        called exactly twice.
        """
        rows = [_row("n1", "only null rows"), _row("n2", "also null")]
        session = _mock_session(rows)
        gen = _gen_with_mapping(
            {"only null rows": [0.1] * _DIM, "also null": [0.2] * _DIM}
        )

        await backfill.backfill_missing_embeddings(session, gen)

        assert gen.generate.await_count == len(rows)

    @pytest.mark.asyncio
    async def test_rerun_is_noop_when_all_have_embeddings(self) -> None:
        """A second run finds no NULL rows → no embedding, no update, no commit."""
        session = _mock_session([])
        gen = MagicMock()
        gen.generate = AsyncMock(return_value=[0.5] * _DIM)

        stats = await backfill.backfill_missing_embeddings(session, gen)

        gen.generate.assert_not_awaited()
        session.commit.assert_not_awaited()
        assert stats.scanned == 0
        assert stats.embedded == 0
        assert stats.failed == 0

    @pytest.mark.asyncio
    async def test_failed_embedding_left_null_and_retriable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A row whose embedding raises is skipped (left NULL) without aborting the run.

        It is counted as failed and stays NULL, so a later run retries it
        idempotently. Other rows still get their vectors.
        """
        rows = [_row("ok", "good content"), _row("bad", "broken content")]
        vecs = {"good content": [0.9] * _DIM}

        async def _generate(content: str) -> list[float]:
            if content == "broken content":
                raise RuntimeError("embed service down")
            return vecs[content]

        gen = MagicMock()
        gen.generate = AsyncMock(side_effect=_generate)
        gen.model = "text-embedding-3-small"
        gen.dimension = _DIM
        session = _mock_session(rows)

        applied: list[object] = []

        async def _spy(_session: object, mid: object, _vec: list[float]) -> None:
            applied.append(mid)

        monkeypatch.setattr(backfill, "_apply_embedding", _spy)

        stats = await backfill.backfill_missing_embeddings(session, gen)

        assert stats.scanned == 2
        assert stats.embedded == 1
        assert stats.failed == 1
        # Only the successful row was written; the failed one stays NULL.
        assert applied == ["ok"]
        session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_empty_embedding_treated_as_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A generator returning an empty vector is a failure, not a silent write."""
        rows = [_row("e", "empty vec content")]
        gen = MagicMock()
        gen.generate = AsyncMock(return_value=[])  # malformed
        gen.model = "text-embedding-3-small"
        gen.dimension = _DIM
        session = _mock_session(rows)

        wrote: list[object] = []

        async def _spy(_session: object, mid: object, _vec: list[float]) -> None:
            wrote.append(mid)

        monkeypatch.setattr(backfill, "_apply_embedding", _spy)

        stats = await backfill.backfill_missing_embeddings(session, gen)

        assert stats.embedded == 0
        assert stats.failed == 1
        assert wrote == []

    @pytest.mark.asyncio
    async def test_dry_run_embeds_but_does_not_persist(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--dry-run exercises embedding but writes nothing and does not commit."""
        rows = [_row("d", "dry content")]
        session = _mock_session(rows)
        gen = _gen_with_mapping({"dry content": [0.4] * _DIM})

        wrote: list[object] = []

        async def _spy(_session: object, mid: object, _vec: list[float]) -> None:
            wrote.append(mid)

        monkeypatch.setattr(backfill, "_apply_embedding", _spy)

        stats = await backfill.backfill_missing_embeddings(
            session, gen, dry_run=True
        )

        assert gen.generate.await_count == 1
        assert wrote == []  # nothing persisted
        session.commit.assert_not_awaited()  # no commit in dry run
        assert stats.scanned == 1
        assert stats.embedded == 1  # would-have-embedded count


# ─── _apply_embedding ────────────────────────────────────────────────


class TestApplyEmbedding:
    """_apply_embedding issues a single UPDATE scoped to the given memory id."""

    @pytest.mark.asyncio
    async def test_issues_one_update_for_id(self) -> None:
        session = MagicMock()
        session.execute = AsyncMock()

        await backfill._apply_embedding(session, "mid-7", [0.3] * _DIM)

        session.execute.assert_awaited_once()
        stmt = session.execute.call_args.args[0]
        # The statement targets cold_memories and sets the embedding column.
        compiled = str(stmt.compile())
        assert "cold_memories" in compiled
        assert "embedding" in compiled


# ─── arg parsing ─────────────────────────────────────────────────────


class TestArgParsing:
    """CLI flags parse to the expected defaults/overrides."""

    def test_defaults(self) -> None:
        args = backfill._parse_args([])
        assert args.concurrency == 5
        assert args.dry_run is False

    def test_overrides(self) -> None:
        args = backfill._parse_args(["--concurrency", "8", "--dry-run"])
        assert args.concurrency == 8
        assert args.dry_run is True
