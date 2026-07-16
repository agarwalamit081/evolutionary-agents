"""Regression for Fix 3b — ``ToolPersister.list_tools`` bounds its result set.

Governance caps ACTIVE tools at 25, but retired registrations accumulate
unbounded over the project lifetime, so an unbounded ``select`` + a per-row
``_latest_version`` call (N+1) could yield a large operator listing on a
long-lived registry. Fix 3b adds
``.order_by(created_at.desc()).limit(DASHBOARD_TOOLS_MAX_ROWS)``.

This test captures the executed statement and asserts it carries a ``LIMIT``
clause, and that the bound equals the configured cap (``DashboardSettings``).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.settings import get_settings
from src.tools.dynamic.persister import ToolPersister


class _CapturingSessionCtx:
    """Async cm yielding a session whose ``execute`` records the statement.

    Returns a result with ``.scalars().all() == []`` so no rows iterate and the
    per-row ``_latest_version`` N+1 never fires — the test is about the query
    shape, not the row shaping.
    """

    def __init__(self) -> None:
        self.captured: Any = None
        self._session = MagicMock()
        self._session.execute = AsyncMock(side_effect=self._capture)
        self._session.rollback = AsyncMock()

    async def _capture(self, stmt: Any, *_a: Any, **_k: Any) -> Any:
        self.captured = stmt
        result = MagicMock()
        result.scalars.return_value.all.return_value = []
        return result

    async def __aenter__(self) -> MagicMock:
        return self._session

    async def __aexit__(self, *_exc: object) -> bool:
        return False


@pytest.mark.asyncio
class TestListToolsBounded:
    async def test_list_tools_query_carries_configured_limit(self) -> None:
        ctx = _CapturingSessionCtx()
        # list_tools imports get_session lazily from src.db.session, so the
        # patch target is the source module, not persister's namespace.
        with patch("src.db.session.get_session", return_value=ctx):
            tools = await ToolPersister().list_tools()

        # No rows in the fake → empty list (and no _latest_version N+1 fired).
        assert tools == []
        # A statement must have been executed (guards against a silent except → []).
        assert ctx.captured is not None
        compiled = str(
            ctx.captured.compile(compile_kwargs={"literal_binds": True})
        ).upper()
        assert "LIMIT" in compiled
        # The bound equals the configured DashboardSettings cap (not a hardcoded
        # literal) — proving the new knob is wired through.
        assert str(get_settings().dashboard.tools_max_rows) in compiled.lower()
