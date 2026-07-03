"""Tests for ``ToolPersister.retire`` fresh-tool protection (Phase-1c Layer 1).

When ``tool_protection_window_s > 0``, ``retire`` spares any candidate created
within the window so no name-based governance pass (semantic dedup /
performance / unused) can retire a prior run's tool before it is reused. The
session is mocked (mirroring ``test_persister_dedup``) so the post-query
contract is verified deterministically: the protected set is subtracted from
the retire list, and ``window=0`` (default) preserves unconditional retirement.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.settings import get_settings
from src.tools.dynamic.persister import ToolPersister


def _retire_session(protected_names: list[str]):
    """A ``get_session`` CM whose ``execute`` serves the [select, update] run.

    The protection scan is the first ``execute`` (returns ``[(name,), ...]`` via
    ``.all()``); the subsequent update ``execute`` returns an unused result.
    """

    select_result = MagicMock()
    select_result.all = MagicMock(
        return_value=[(n,) for n in protected_names]
    )

    @asynccontextmanager
    async def _ctx():  # type: ignore[no-untyped-def]
        session = MagicMock()
        session.execute = AsyncMock(side_effect=[select_result, MagicMock()])
        yield session

    return _ctx


@asynccontextmanager
async def _plain_session():  # type: ignore[no-untyped-def]
    """A ``get_session`` CM for the ``window=0`` path (a single update call)."""
    session = MagicMock()
    session.execute = AsyncMock(return_value=MagicMock())
    yield session


class TestRetireProtection:
    @pytest.mark.asyncio
    async def test_fresh_tool_spared_when_window_open(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A within-window tool is spared; only the older name is retired."""
        monkeypatch.setattr(
            get_settings().agent, "tool_protection_window_s", 86400
        )
        with patch(
            "src.db.session.get_session", _retire_session(["fresh_tool"])
        ):
            retired = await ToolPersister().retire(["fresh_tool", "old_tool"])
        assert retired == 1

    @pytest.mark.asyncio
    async def test_all_fresh_spared_returns_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When every candidate is within the window, none are retired."""
        monkeypatch.setattr(
            get_settings().agent, "tool_protection_window_s", 86400
        )
        with patch(
            "src.db.session.get_session",
            _retire_session(["fresh_one", "fresh_two"]),
        ):
            retired = await ToolPersister().retire(["fresh_one", "fresh_two"])
        assert retired == 0

    @pytest.mark.asyncio
    async def test_window_zero_retires_unconditionally(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``window=0`` (default) → no protection; every name is retired."""
        monkeypatch.setattr(
            get_settings().agent, "tool_protection_window_s", 0
        )
        with patch("src.db.session.get_session", _plain_session):
            retired = await ToolPersister().retire(["a", "b", "c"])
        assert retired == 3
