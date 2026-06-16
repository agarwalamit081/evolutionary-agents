"""Tests for cumulative cap + redundancy enforcement in ToolPersister (B3).

Covers the DB-level de-bloat helpers (``retire``, ``_retire_excess_tools`` by
age, ``retire_redundant`` by cosine) and that ``load_active_tools`` wires them
only when ``settings`` is passed. Tools lack per-tool success metrics until M4,
so redundancy scoring is ``(version, created_ts)``.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.dynamic.persister import ToolPersister
from src.tools.registry import ToolRegistry


def _settings(**overrides: Any) -> Any:
    from src.config.settings import AgentSettings

    base: dict[str, Any] = {
        "max_active_tools": 25,
        "capability_redundancy_threshold": 0.92,
    }
    base.update(overrides)
    return AgentSettings(_env_file=None, **base)


def _bind_session(session: MagicMock) -> Any:
    """Return a no-arg get_session-equivalent (the callable, not a CM instance).

    Mirrors the recall-test idiom ``patch(..., new=_fake_get_session)``: the
    patched ``get_session`` must be *callable* (``async with get_session()``).
    """

    @asynccontextmanager
    async def _get_session() -> AsyncGenerator[MagicMock, None]:
        yield session

    return _get_session


class TestRetire:
    @pytest.mark.asyncio
    async def test_marks_named_tools_inactive(self) -> None:
        session = MagicMock()
        session.execute = AsyncMock()
        persister = ToolPersister()
        with patch("src.db.session.get_session", new=_bind_session(session)):
            count = await persister.retire(["a", "b"])
        assert count == 2
        assert session.execute.await_count == 1

    @pytest.mark.asyncio
    async def test_empty_names_noop(self) -> None:
        assert await ToolPersister().retire([]) == 0


class TestRetireExcessTools:
    @pytest.mark.asyncio
    async def test_retires_oldest_beyond_cap(self) -> None:
        """3 active (newest-first), cap 2 → the oldest id is retired via UPDATE."""
        newest, mid, oldest = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        id_result = MagicMock()
        id_result.all.return_value = [(newest,), (mid,), (oldest,)]

        captured: dict[str, Any] = {}

        async def _exec(stmt: Any) -> Any:
            if "update_stmt" not in captured:
                captured["update_stmt"] = stmt
                return id_result
            captured["real_update"] = stmt
            return MagicMock()

        session = MagicMock()
        session.execute = AsyncMock(side_effect=_exec)
        persister = ToolPersister()
        with patch("src.db.session.get_session", new=_bind_session(session)):
            count = await persister._retire_excess_tools(2)

        assert count == 1
        compiled = str(
            captured["real_update"].compile(compile_kwargs={"literal_binds": True})
        )
        # literal_binds renders UUIDs without hyphens → compare against .hex.
        assert oldest.hex in compiled  # oldest retired
        assert newest.hex not in compiled  # newest kept

    @pytest.mark.asyncio
    async def test_under_cap_noop(self) -> None:
        id_result = MagicMock()
        id_result.all.return_value = [(uuid.uuid4(),), (uuid.uuid4(),)]
        session = MagicMock()
        session.execute = AsyncMock(return_value=id_result)
        persister = ToolPersister()
        with patch("src.db.session.get_session", new=_bind_session(session)):
            count = await persister._retire_excess_tools(5)
        assert count == 0


class TestRetireRedundant:
    @pytest.mark.asyncio
    async def test_keeps_higher_version_twin(self) -> None:
        """Two semantically-identical tools → the lower-version one is retired."""
        emb = [1.0] + [0.0] * 767
        rows = [
            {"name": "v1_tool", "embedding": emb, "version": 1, "created_ts": 1.0, "score": (1, 1.0)},
            {"name": "v3_tool", "embedding": emb, "version": 3, "created_ts": 2.0, "score": (3, 2.0)},
        ]
        persister = ToolPersister()
        retire_mock = AsyncMock(return_value=0)
        persister.retire = retire_mock
        with patch.object(
            ToolPersister,
            "_active_tool_capability_rows",
            AsyncMock(return_value=rows),
        ):
            retired = await persister.retire_redundant(0.92)
        assert retired == ["v1_tool"]
        retire_mock.assert_awaited_once_with(["v1_tool"])

    @pytest.mark.asyncio
    async def test_below_threshold_not_retired(self) -> None:
        """Orthogonal embeddings (sim ~0.5) stay under the 0.92 cutoff."""
        emb_a = [1.0] + [0.0] * 767
        emb_b = [0.0, 1.0] + [0.0] * 766  # orthogonal → cosine 0.0
        rows = [
            {"name": "a", "embedding": emb_a, "version": 1, "created_ts": 1.0, "score": (1, 1.0)},
            {"name": "b", "embedding": emb_b, "version": 1, "created_ts": 1.0, "score": (1, 1.0)},
        ]
        persister = ToolPersister()
        persister.retire = AsyncMock(return_value=0)
        with patch.object(
            ToolPersister,
            "_active_tool_capability_rows",
            AsyncMock(return_value=rows),
        ):
            assert await persister.retire_redundant(0.92) == []


class TestLoadActiveToolsWiring:
    @pytest.mark.asyncio
    async def test_settings_invokes_debloat_passes(self) -> None:
        persister = ToolPersister()
        persister.retire_redundant = AsyncMock(return_value=[])
        persister._retire_excess_tools = AsyncMock(return_value=0)
        session = MagicMock()
        reg_result = MagicMock()
        reg_result.scalars.return_value.all.return_value = []
        session.execute = AsyncMock(return_value=reg_result)
        with patch("src.db.session.get_session", new=_bind_session(session)):
            loaded = await persister.load_active_tools(
                ToolRegistry(), settings=_settings()
            )
        assert loaded == []
        persister.retire_redundant.assert_awaited_once()
        persister._retire_excess_tools.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_settings_skips_debloat(self) -> None:
        persister = ToolPersister()
        persister.retire_redundant = AsyncMock(return_value=[])
        persister._retire_excess_tools = AsyncMock(return_value=0)
        session = MagicMock()
        reg_result = MagicMock()
        reg_result.scalars.return_value.all.return_value = []
        session.execute = AsyncMock(return_value=reg_result)
        with patch("src.db.session.get_session", new=_bind_session(session)):
            loaded = await persister.load_active_tools(
                ToolRegistry(), settings=None
            )
        assert loaded == []
        persister.retire_redundant.assert_not_awaited()
        persister._retire_excess_tools.assert_not_awaited()
