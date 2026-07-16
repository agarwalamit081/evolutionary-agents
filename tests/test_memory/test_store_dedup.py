"""src.memory — store-time near-duplicate merge (Q81).

When ``MEMORY_DEDUP_ENABLED`` is on, ``WarmMemoryStore.store`` (skills path) and
``ColdMemory.store`` (episodes) skip the insert when an existing same-type
memory is within ``dedup_threshold`` cosine similarity, returning the existing
id instead — so the tiers don't accumulate near-identical crystallizations.
Gracefully skipped when no embedding is available. Default off.

The dedup query (pgvector ``<=>``) is built as a SQL expression but never
executed against pgvector here — the fake session returns a configured scalar
(``scalar_one_or_none``). ``get_settings().memory`` is monkeypatched per test so
the real .env never influences the verdict.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.config import get_settings
from src.db.models import ColdMemory as ColdMemoryModel, MemoryEmbedding, WarmMemory
from src.memory.cold import ColdMemory
from src.memory.warm import WarmMemoryStore


class _ScalarResult:
    def __init__(self, scalar: Any) -> None:
        self._scalar = scalar

    def scalar_one_or_none(self) -> Any:
        return self._scalar


class _FakeSession:
    """Captures add()/commit()/execute(); returns one configured scalar."""

    def __init__(self, scalar: Any = None) -> None:
        self.added: list[Any] = []
        self.commits = 0
        self.executes = 0
        self._scalar = scalar

    def add(self, obj: Any) -> None:
        self.added.append(obj)

    async def execute(self, _stmt: Any) -> _ScalarResult:
        self.executes += 1
        return _ScalarResult(self._scalar)

    async def commit(self) -> None:
        self.commits += 1


class _FakeGen:
    def __init__(self) -> None:
        self.model = "test-embed"

    async def generate(self, _text: str) -> list[float]:
        return [0.0] * 768


def _enable_dedup(monkeypatch: pytest.MonkeyPatch, *, threshold: float = 0.92) -> None:
    ms = get_settings().memory
    monkeypatch.setattr(ms, "dedup_enabled", True)
    monkeypatch.setattr(ms, "dedup_threshold", threshold)


def _disable_dedup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(get_settings().memory, "dedup_enabled", False)


class TestWarmStoreDedup:
    @pytest.mark.asyncio
    async def test_near_dup_returns_existing_id_no_insert(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _enable_dedup(monkeypatch)
        session = _FakeSession(scalar="existing-uuid")
        store = WarmMemoryStore(session, generator=_FakeGen())  # type: ignore[arg-type]

        returned = await store.store(
            memory_type="skill", name="dup_skill", content="normalize timestamps"
        )

        assert returned == "existing-uuid"
        # A dedup hit must NOT add a row or commit (the existing row is reused).
        assert session.added == []
        assert session.commits == 0
        assert session.executes == 1  # exactly the dedup similarity probe

    @pytest.mark.asyncio
    async def test_no_match_inserts_normally(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _enable_dedup(monkeypatch)
        session = _FakeSession(scalar=None)  # nothing similar
        store = WarmMemoryStore(session, generator=_FakeGen())  # type: ignore[arg-type]

        returned = await store.store(
            memory_type="skill", name="fresh_skill", content="brand new approach"
        )

        assert returned != "existing-uuid"
        # WarmMemory parent + MemoryEmbedding row both persisted.
        assert len(session.added) == 2
        assert any(isinstance(o, WarmMemory) for o in session.added)
        assert any(isinstance(o, MemoryEmbedding) for o in session.added)
        assert session.commits == 1

    @pytest.mark.asyncio
    async def test_dedup_disabled_stores_normally(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _disable_dedup(monkeypatch)
        session = _FakeSession(scalar="should-not-be-used")
        store = WarmMemoryStore(session, generator=_FakeGen())  # type: ignore[arg-type]

        returned = await store.store(
            memory_type="skill", name="s", content="c"
        )

        assert returned != "should-not-be-used"
        # Dedup query never runs when disabled.
        assert session.executes == 0
        assert len(session.added) == 2
        assert session.commits == 1

    @pytest.mark.asyncio
    async def test_no_embedding_skips_dedup_still_stores(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _enable_dedup(monkeypatch)
        session = _FakeSession(scalar="would-have-hit")
        # No generator ⇒ no embedding ⇒ dedup gracefully skipped.
        store = WarmMemoryStore(session, generator=None)  # type: ignore[arg-type]

        returned = await store.store(memory_type="skill", name="s", content="c")

        assert returned != "would-have-hit"
        assert session.executes == 0  # dedup needs an embedding to probe
        # No embedding ⇒ only the WarmMemory row (no MemoryEmbedding).
        assert len(session.added) == 1
        assert isinstance(session.added[0], WarmMemory)
        assert session.commits == 1


class TestColdStoreDedup:
    @pytest.mark.asyncio
    async def test_near_dup_returns_existing_id_no_insert(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _enable_dedup(monkeypatch)
        session = _FakeSession(scalar="existing-episode")
        cold = ColdMemory(session, generator=None)  # type: ignore[arg-type]

        returned = await cold.store(
            episode_type="execution",
            content="duplicate episode",
            embedding=[0.0] * 768,
        )

        assert returned == "existing-episode"
        assert session.added == []
        assert session.commits == 0
        assert session.executes == 1

    @pytest.mark.asyncio
    async def test_no_match_inserts_normally(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _enable_dedup(monkeypatch)
        session = _FakeSession(scalar=None)
        cold = ColdMemory(session, generator=None)  # type: ignore[arg-type]

        returned = await cold.store(
            episode_type="execution", content="fresh episode", embedding=[0.0] * 768
        )

        assert returned != "existing-episode"
        assert len(session.added) == 1
        assert isinstance(session.added[0], ColdMemoryModel)
        assert session.commits == 1

    @pytest.mark.asyncio
    async def test_dedup_disabled_stores_normally(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _disable_dedup(monkeypatch)
        session = _FakeSession(scalar="should-not-be-used")
        cold = ColdMemory(session, generator=None)  # type: ignore[arg-type]

        returned = await cold.store(
            episode_type="execution", content="c", embedding=[0.0] * 768
        )

        assert returned != "should-not-be-used"
        assert session.executes == 0
        assert len(session.added) == 1
        assert session.commits == 1
