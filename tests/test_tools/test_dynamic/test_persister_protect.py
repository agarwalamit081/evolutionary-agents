"""Tests for ``ToolPersister.retire`` fresh-tool protection (Phase-1c Layer 1).

When ``tool_protection_window_s > 0``, ``retire`` spares any candidate created
within the window so no name-based governance pass (semantic dedup /
performance / unused) can retire a prior run's tool before it is reused. The
session is mocked (mirroring ``test_persister_dedup``) so the post-query
contract is verified deterministically: the protected set is subtracted from
the retire list, and ``window=0`` (default) preserves unconditional retirement.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Any
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


class TestPersistReactivation:
    """``persist`` over a deactivated registration must re-activate it.

    Regression for the channel-A G0→G1 inheritance break: when a run regenerates
    a tool whose ``ToolRegistration`` row is ``is_active=False`` (a prior
    ``clean_state`` reset / governance retirement), the version bump must flip
    the registration back to ``is_active=True`` — otherwise ``load_active_tools``
    / ``find_similar`` (which filter on ``ToolRegistration.is_active=True``)
    never recall it, so a later generation cannot reuse the capability.
    """

    @pytest.mark.asyncio
    async def test_persist_reactivates_deactivated_registration(self) -> None:
        existing = MagicMock()
        existing.id = uuid.UUID("12345678-1234-1234-1234-1234567890ab")

        select_reg = MagicMock()
        select_reg.scalar_one_or_none = MagicMock(return_value=existing)
        select_ver = MagicMock()
        select_ver.scalars = MagicMock(
            return_value=MagicMock(first=MagicMock(return_value=None))
        )

        executed: list[Any] = []
        call = {"n": 0}

        async def _exec(stmt: Any) -> Any:  # type: ignore[no-untyped-def]
            call["n"] += 1
            executed.append(stmt)
            # call order: select-reg(1), update-reg(2), select-ver(3), update-ver(4)
            if call["n"] == 1:
                return select_reg
            if call["n"] == 3:
                return select_ver
            return MagicMock()

        @asynccontextmanager
        async def _ctx():  # type: ignore[no-untyped-def]
            session = MagicMock()
            session.execute = AsyncMock(side_effect=_exec)
            session.add = MagicMock()
            session.flush = AsyncMock()
            yield session

        with patch("src.db.session.get_session", _ctx):
            returned = await ToolPersister().persist(
                tool_name="normalize_prices",
                description="d",
                input_schema={"type": "object"},
                handler_code="async def fn(args):\n    return args",
                # No embedding → reg_values must still carry is_active=True alone.
                capability_embedding=None,
            )

        assert returned == existing.id
        # executed[1] is the registration UPDATE (call 2); it must re-activate.
        reg_update_sql = str(
            executed[1].compile(compile_kwargs={"literal_binds": True})
        ).lower()
        assert "tool_registrations" in reg_update_sql
        assert "is_active" in reg_update_sql
        # And it must be the ONLY registration write that flipped it true here
        # (the version deactivation at call 4 targets tool_versions, not this).
        assert "is_active=true" in reg_update_sql or "is_active = true" in reg_update_sql
