"""Unit tests for ``src.db.backfills.embeddings`` — the consolidated backfill.

Promoted from the two ``scripts/backfill_*_embeddings.py`` (SI-1). Tests the
*logic* hermetically: the DB is an external dependency, so the async session is
mocked (the same convention the rest of the memory/db test suite uses). We do
NOT spin up an in-memory SQLite table — the models carry pgvector ``Vector(768)``
+ ``JSONB`` columns that are Postgres-only, so a real SQLite table cannot host
them. The mock exercises the real module code — the NULL-filter SELECT, per-row
embedding, store-vs-skip decision, per-row UPDATE, idempotency, and the
``run_backfill`` orchestrator — without any provider key or infrastructure.

The two backfills differ in one way the tests pin explicitly:
- **capability** stores ONLY ``source == "api"`` vectors (hash/no-text skipped);
- **cold** stores EVERY non-empty vector (hash fallback included).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.backfills import embeddings as bf
from src.db.models import ToolRegistration
from src.memory.embeddings import EmbeddingGenerator

_DIM = 768


# ─── mock builders ───────────────────────────────────────────────────


def _cold_row(rid: str, content: str) -> SimpleNamespace:
    """A cold-memory-shaped row (id + content are all the cold path reads)."""
    return SimpleNamespace(id=rid, content=content)


def _cap_row(
    rid: str,
    *,
    capability_text: str | None = None,
    tool_name: str | None = None,
    name: str | None = None,
    description: str | None = None,
) -> SimpleNamespace:
    """A capability-shaped row (tool_name OR name + description + capability_text)."""
    return SimpleNamespace(
        id=rid,
        capability_text=capability_text,
        tool_name=tool_name,
        name=name,
        description=description,
    )


def _mock_session(rows: list[SimpleNamespace]) -> MagicMock:
    """A session whose execute() returns `rows` via scalars().all() (the SELECT).

    The same return value is reused for any UPDATE execute() call, but the
    store-path tests monkeypatch ``_apply_*`` to a spy, so UPDATEs never reach
    the session in those cases.
    """
    session = MagicMock()
    select_result = MagicMock()
    select_result.scalars.return_value.all.return_value = rows
    session.execute = AsyncMock(return_value=select_result)
    session.commit = AsyncMock()
    return session


def _gen_cold(content_to_vec: dict[str, list[float]]) -> MagicMock:
    """A generator whose generate(content) returns the mapped vector (cold path)."""
    gen = MagicMock(spec=EmbeddingGenerator)
    gen.generate = AsyncMock(side_effect=lambda c: content_to_vec[c])
    gen.model = "text-embedding-3-small"
    gen.dimension = _DIM
    return gen


def _gen_capability(
    text_to_vec: dict[str, list[float]], *, source: str = "api"
) -> MagicMock:
    """A generator that sets ``last_source`` then returns the mapped vector.

    The capability path decides store-vs-skip by reading ``generator.last_source``
    after each ``generate()``, so the mock must set it (matching the real
    ``EmbeddingGenerator.generate`` which sets ``last_source`` right before return).
    """
    gen = MagicMock(spec=EmbeddingGenerator)

    async def _generate(text: str) -> list[float]:
        gen.last_source = source
        return text_to_vec[text]

    gen.generate = AsyncMock(side_effect=_generate)
    gen.last_source = None
    gen.model = "text-embedding-3-small"
    gen.dimension = _DIM
    return gen


# ════════════════════ capability pure helpers ════════════════════════


class TestSelectCapabilityText:
    """Text selection prefers capability_text, then name+description, else None."""

    def test_prefers_capability_text(self) -> None:
        out = bf.select_capability_text("official capability blurb", "tool", "a description")
        assert out == "official capability blurb"

    def test_falls_back_to_name_and_description(self) -> None:
        out = bf.select_capability_text(None, "web_search", "Searches the web")
        assert out == "web_search: Searches the web"

    def test_falls_back_when_capability_text_blank(self) -> None:
        out = bf.select_capability_text("   \n\t ", "tool", "desc")
        assert out == "tool: desc"

    def test_returns_none_when_all_empty(self) -> None:
        assert bf.select_capability_text(None, None, None) is None

    def test_returns_none_when_name_or_description_missing(self) -> None:
        assert bf.select_capability_text(None, "name", None) is None
        assert bf.select_capability_text(None, None, "desc") is None

    def test_returns_none_for_blank_name_or_description(self) -> None:
        assert bf.select_capability_text(None, "  ", "desc") is None
        assert bf.select_capability_text(None, "name", "  ") is None

    def test_blank_capability_text_with_no_synthesis_is_none(self) -> None:
        assert bf.select_capability_text("   ", None, None) is None


class TestShouldStoreCapability:
    """Only "api" vectors are stored; hash/None/unknown are skipped."""

    def test_true_for_api(self) -> None:
        assert bf.should_store_capability("api") is True

    def test_false_for_hash(self) -> None:
        assert bf.should_store_capability("hash") is False

    def test_false_for_none(self) -> None:
        assert bf.should_store_capability(None) is False

    def test_false_for_unknown_source(self) -> None:
        assert bf.should_store_capability("anything-else") is False


# ════════════════════ capability backfill behavior ═══════════════════


class TestBackfillCapabilityTable:
    """api vectors stored; hash/no-text/failed skipped; idempotent; dry-run."""

    @pytest.mark.asyncio
    async def test_fetch_filters_null_capability(
        self,
    ) -> None:
        """_fetch_null_capability_rows issues a SELECT filtered on IS NULL."""
        session = _mock_session([])
        await bf._fetch_null_capability_rows(session, ToolRegistration)
        session.execute.assert_awaited_once()
        where = session.execute.call_args.args[0].whereclause
        assert where is not None
        assert "capability_embedding IS NULL" in str(where.compile())

    @pytest.mark.asyncio
    async def test_api_vector_is_stored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An api-source vector is embedded from capability_text and UPDATEd back."""
        rows = [_cap_row("c1", capability_text="alpha blurb", tool_name="t", description="d")]
        vec = [0.11] * _DIM
        session = _mock_session(rows)
        gen = _gen_capability({"alpha blurb": vec}, source="api")

        applied: list[tuple[object, list[float], str]] = []

        async def _spy(_s: AsyncSession, _mc: object, mid: object, v: list[float], txt: str) -> None:
            applied.append((mid, v, txt))

        monkeypatch.setattr(bf, "_apply_capability", _spy)

        stats = await bf.backfill_capability_table(session, ToolRegistration, gen)

        gen.generate.assert_awaited_once_with("alpha blurb")
        assert applied == [("c1", vec, "alpha blurb")]
        session.commit.assert_awaited_once()
        assert stats.scanned == 1
        assert stats.stored == 1
        assert stats.failed == 0
        assert stats.skipped_hash == 0
        assert stats.skipped_no_text == 0

    @pytest.mark.asyncio
    async def test_hash_vector_is_skipped_not_stored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A hash-fallback vector is counted skipped_hash and left NULL (no UPDATE)."""
        rows = [_cap_row("h1", capability_text="txt", tool_name="t", description="d")]
        session = _mock_session(rows)
        gen = _gen_capability({"txt": [0.5] * _DIM}, source="hash")

        wrote: list[object] = []

        async def _spy(_s: AsyncSession, _mc: object, mid: object, _v: list[float], _t: str) -> None:
            wrote.append(mid)

        monkeypatch.setattr(bf, "_apply_capability", _spy)

        stats = await bf.backfill_capability_table(session, ToolRegistration, gen)

        assert stats.stored == 0
        assert stats.skipped_hash == 1
        assert wrote == []  # nothing persisted
        session.commit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_text_row_is_skipped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A row with no embeddable text is counted skipped_no_text, never embedded."""
        rows = [_cap_row("nt1")]  # no capability_text, no name, no description
        session = _mock_session(rows)
        gen = _gen_capability({}, source="api")

        async def _spy(*args: object) -> None:  # never called
            raise AssertionError("should not store a no-text row")

        monkeypatch.setattr(bf, "_apply_capability", _spy)

        stats = await bf.backfill_capability_table(session, ToolRegistration, gen)

        gen.generate.assert_not_awaited()
        assert stats.skipped_no_text == 1
        assert stats.stored == 0

    @pytest.mark.asyncio
    async def test_failed_embedding_left_null_and_retriable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A row whose embedding raises is counted failed (left NULL) without aborting."""
        rows = [
            _cap_row("ok", capability_text="good", tool_name="t", description="d"),
            _cap_row("bad", capability_text="broken", tool_name="t2", description="d2"),
        ]
        gen = MagicMock(spec=EmbeddingGenerator)

        async def _generate(text: str) -> list[float]:
            if text == "broken":
                raise RuntimeError("embed service down")
            gen.last_source = "api"
            return [0.9] * _DIM

        gen.generate = AsyncMock(side_effect=_generate)
        gen.last_source = None
        session = _mock_session(rows)

        applied: list[object] = []

        async def _spy(_s: AsyncSession, _mc: object, mid: object, _v: list[float], _t: str) -> None:
            applied.append(mid)

        monkeypatch.setattr(bf, "_apply_capability", _spy)

        stats = await bf.backfill_capability_table(session, ToolRegistration, gen)

        assert stats.scanned == 2
        assert stats.stored == 1
        assert stats.failed == 1
        assert applied == ["ok"]  # only the good row was written
        session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_empty_embedding_treated_as_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty vector (api source) is a failure, not a silent write."""
        rows = [_cap_row("e1", capability_text="txt", tool_name="t", description="d")]
        gen = MagicMock(spec=EmbeddingGenerator)
        gen.generate = AsyncMock(return_value=[])  # malformed
        gen.last_source = "api"
        session = _mock_session(rows)

        async def _spy(*args: object) -> None:
            raise AssertionError("should not store an empty vector")

        monkeypatch.setattr(bf, "_apply_capability", _spy)

        stats = await bf.backfill_capability_table(session, ToolRegistration, gen)

        assert stats.stored == 0
        assert stats.failed == 1

    @pytest.mark.asyncio
    async def test_rerun_is_noop_when_all_have_vectors(self) -> None:
        """A run finding no NULL rows embeds/updates/commits nothing."""
        session = _mock_session([])
        gen = _gen_capability({}, source="api")
        stats = await bf.backfill_capability_table(session, ToolRegistration, gen)
        gen.generate.assert_not_awaited()
        session.commit.assert_not_awaited()
        assert stats.scanned == 0 and stats.stored == 0 and stats.failed == 0

    @pytest.mark.asyncio
    async def test_dry_run_embeds_but_does_not_persist(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--dry-run exercises embedding but writes nothing and does not commit."""
        rows = [_cap_row("d1", capability_text="dry txt", tool_name="t", description="d")]
        session = _mock_session(rows)
        gen = _gen_capability({"dry txt": [0.4] * _DIM}, source="api")

        async def _spy(*args: object) -> None:
            raise AssertionError("dry-run must not persist")

        monkeypatch.setattr(bf, "_apply_capability", _spy)

        stats = await bf.backfill_capability_table(
            session, ToolRegistration, gen, dry_run=True
        )

        assert gen.generate.await_count == 1
        session.commit.assert_not_awaited()
        assert stats.scanned == 1
        assert stats.stored == 1  # would-have-stored count

    @pytest.mark.asyncio
    async def test_apply_capability_issues_one_update(self) -> None:
        """_apply_capability issues a single UPDATE scoped to the row id."""
        session = MagicMock()
        session.execute = AsyncMock()
        await bf._apply_capability(session, ToolRegistration, "rid", [0.3] * _DIM, "txt")
        session.execute.assert_awaited_once()
        compiled = str(session.execute.call_args.args[0].compile())
        assert "tool_registrations" in compiled
        assert "capability_embedding" in compiled
        assert "capability_text" in compiled


# ════════════════════ cold-memory backfill behavior ══════════════════


class TestBackfillColdMemories:
    """NULL rows get vectors; non-NULL skipped; failed retriable; dry-run; idempotent."""

    @pytest.mark.asyncio
    async def test_fetch_filters_null_embeddings(self) -> None:
        session = _mock_session([])
        await bf._fetch_null_cold_rows(session)
        session.execute.assert_awaited_once()
        where = session.execute.call_args.args[0].whereclause
        assert where is not None
        assert "embedding IS NULL" in str(where.compile())

    @pytest.mark.asyncio
    async def test_null_rows_get_vectors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = [_cold_row("r1", "alpha"), _cold_row("r2", "beta")]
        vecs = {"alpha": [0.11] * _DIM, "beta": [0.22] * _DIM}
        session = _mock_session(rows)
        gen = _gen_cold(vecs)

        applied: list[tuple[object, list[float]]] = []

        async def _spy(_s: AsyncSession, mid: object, vec: list[float]) -> None:
            applied.append((mid, vec))

        monkeypatch.setattr(bf, "_apply_cold_embedding", _spy)

        stats = await bf.backfill_cold_memories(session, gen)

        gen.generate.assert_any_await("alpha")
        gen.generate.assert_any_await("beta")
        assert applied == [("r1", vecs["alpha"]), ("r2", vecs["beta"])]
        session.commit.assert_awaited_once()
        assert stats.scanned == 2 and stats.stored == 2 and stats.failed == 0

    @pytest.mark.asyncio
    async def test_non_null_rows_are_skipped(self) -> None:
        rows = [_cold_row("n1", "only null rows"), _cold_row("n2", "also null")]
        session = _mock_session(rows)
        gen = _gen_cold(
            {"only null rows": [0.1] * _DIM, "also null": [0.2] * _DIM}
        )
        await bf.backfill_cold_memories(session, gen)
        assert gen.generate.await_count == len(rows)

    @pytest.mark.asyncio
    async def test_rerun_is_noop_when_all_have_embeddings(self) -> None:
        session = _mock_session([])
        gen = MagicMock()
        gen.generate = AsyncMock(return_value=[0.5] * _DIM)
        stats = await bf.backfill_cold_memories(session, gen)
        gen.generate.assert_not_awaited()
        session.commit.assert_not_awaited()
        assert stats.scanned == 0 and stats.stored == 0 and stats.failed == 0

    @pytest.mark.asyncio
    async def test_failed_embedding_left_null_and_retriable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = [_cold_row("ok", "good content"), _cold_row("bad", "broken content")]

        async def _generate(content: str) -> list[float]:
            if content == "broken content":
                raise RuntimeError("embed service down")
            return [0.9] * _DIM

        gen = MagicMock(spec=EmbeddingGenerator)
        gen.generate = AsyncMock(side_effect=_generate)
        session = _mock_session(rows)

        applied: list[object] = []

        async def _spy(_s: AsyncSession, mid: object, _vec: list[float]) -> None:
            applied.append(mid)

        monkeypatch.setattr(bf, "_apply_cold_embedding", _spy)

        stats = await bf.backfill_cold_memories(session, gen)

        assert stats.scanned == 2 and stats.stored == 1 and stats.failed == 1
        assert applied == ["ok"]
        session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_empty_embedding_treated_as_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = [_cold_row("e", "empty vec content")]
        gen = MagicMock(spec=EmbeddingGenerator)
        gen.generate = AsyncMock(return_value=[])  # malformed
        session = _mock_session(rows)

        wrote: list[object] = []

        async def _spy(_s: AsyncSession, mid: object, _vec: list[float]) -> None:
            wrote.append(mid)

        monkeypatch.setattr(bf, "_apply_cold_embedding", _spy)

        stats = await bf.backfill_cold_memories(session, gen)
        assert stats.stored == 0 and stats.failed == 1
        assert wrote == []

    @pytest.mark.asyncio
    async def test_dry_run_embeds_but_does_not_persist(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = [_cold_row("d", "dry content")]
        session = _mock_session(rows)
        gen = _gen_cold({"dry content": [0.4] * _DIM})

        wrote: list[object] = []

        async def _spy(_s: AsyncSession, mid: object, _vec: list[float]) -> None:
            wrote.append(mid)

        monkeypatch.setattr(bf, "_apply_cold_embedding", _spy)

        stats = await bf.backfill_cold_memories(session, gen, dry_run=True)
        assert gen.generate.await_count == 1
        assert wrote == []
        session.commit.assert_not_awaited()
        assert stats.scanned == 1 and stats.stored == 1

    @pytest.mark.asyncio
    async def test_apply_cold_embedding_issues_one_update(self) -> None:
        """_apply_cold_embedding issues a single UPDATE scoped to the memory id."""
        session = MagicMock()
        session.execute = AsyncMock()
        await bf._apply_cold_embedding(session, "mid-7", [0.3] * _DIM)
        session.execute.assert_awaited_once()
        compiled = str(session.execute.call_args.args[0].compile())
        assert "cold_memories" in compiled
        assert "embedding" in compiled


# ════════════════════ run_backfill orchestrator (DI seam) ════════════


class TestRunBackfill:
    """run_backfill dispatches by table and merges BackfillStats across tables."""

    @pytest.mark.asyncio
    async def test_unknown_table_raises(self) -> None:
        session = MagicMock()
        gen = MagicMock()
        with pytest.raises(ValueError):
            await bf.run_backfill(table="bogus", session=session, generator=gen)

    @pytest.mark.asyncio
    async def test_cold_delegates_to_cold_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cap = AsyncMock(return_value=bf.BackfillStats())
        cold = AsyncMock(return_value=bf.BackfillStats(scanned=3, stored=2, failed=1))
        monkeypatch.setattr(bf, "backfill_capability_table", cap)
        monkeypatch.setattr(bf, "backfill_cold_memories", cold)

        stats = await bf.run_backfill(table="cold", session=MagicMock(), generator=MagicMock())

        cold.assert_awaited_once()
        cap.assert_not_awaited()
        assert stats.scanned == 3 and stats.stored == 2 and stats.failed == 1

    @pytest.mark.asyncio
    async def test_capability_delegates_to_both_tables(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # First call (ToolRegistration) stores 2; second (SubAgentModel) skips 1 hash.
        cap = AsyncMock(
            side_effect=[
                bf.BackfillStats(scanned=2, stored=2),
                bf.BackfillStats(scanned=1, stored=0, skipped_hash=1),
            ]
        )
        cold = AsyncMock(return_value=bf.BackfillStats())
        monkeypatch.setattr(bf, "backfill_capability_table", cap)
        monkeypatch.setattr(bf, "backfill_cold_memories", cold)

        stats = await bf.run_backfill(
            table="capability", session=MagicMock(), generator=MagicMock()
        )

        assert cap.await_count == 2  # tools + sub-agents
        cold.assert_not_awaited()
        assert stats.scanned == 3
        assert stats.stored == 2
        assert stats.skipped_hash == 1

    @pytest.mark.asyncio
    async def test_all_delegates_to_all_tables(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cap = AsyncMock(return_value=bf.BackfillStats(scanned=1, stored=1))
        cold = AsyncMock(return_value=bf.BackfillStats(scanned=4, stored=3, failed=1))
        monkeypatch.setattr(bf, "backfill_capability_table", cap)
        monkeypatch.setattr(bf, "backfill_cold_memories", cold)

        stats = await bf.run_backfill(table="all", session=MagicMock(), generator=MagicMock())

        assert cap.await_count == 2
        cold.assert_awaited_once()
        # capability scanned 1×2 + cold 4 = 6; stored 1×2 + 3 = 5.
        assert stats.scanned == 6
        assert stats.stored == 5
        assert stats.failed == 1
