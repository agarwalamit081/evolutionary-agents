"""Tests for src.graph.checkpoint — AsyncPostgresSaver checkpoint factory.

Regression note: in ``langgraph-checkpoint-postgres`` 3.x,
``AsyncPostgresSaver.from_conn_string`` is an ``@asynccontextmanager`` that
*yield* the saver on enter and closes the backing connection pool on exit — it
does NOT return a saver directly. An earlier revision called ``.setup()`` on the
un-entered context-manager object, which raised ``AttributeError``; that was
swallowed by ``_create_checkpointer``'s bare ``except Exception:`` so *every*
CLI run ran without persistence and ``--resume`` was a silent no-op. These tests
model the real context-manager shape so that regression cannot recur.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.graph.checkpoint import create_checkpointer

# The checkpoint module imports AsyncPostgresSaver inside the function body via
# ``from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver``, so patch
# it at its source module path.
_CHECKPOINT_SAVER_PATH = "langgraph.checkpoint.postgres.aio.AsyncPostgresSaver"


class _FakeFromConnStringCM:
    """Mimic the ``@asynccontextmanager`` returned by ``from_conn_string``.

    Entering yields ``saver`` (or raises ``enter_exc`` to model a connection
    failure during pool creation). Exiting records the cleanup so tests can
    assert the pool is closed when ``setup()`` fails. This mirrors the REAL API
    shape — the prior bug was that tests modeled ``from_conn_string`` as
    returning the saver directly, hiding the misuse.
    """

    def __init__(
        self, saver: MagicMock, enter_exc: BaseException | None = None
    ) -> None:
        self._saver = saver
        self._enter_exc = enter_exc
        self.entered = False
        self.exited = False

    async def __aenter__(self) -> MagicMock:
        self.entered = True
        if self._enter_exc is not None:
            raise self._enter_exc
        return self._saver

    async def __aexit__(self, *args: object) -> bool:
        self.exited = True
        return False


def _wire(
    mock_cls: MagicMock,
    saver: MagicMock,
    enter_exc: BaseException | None = None,
) -> _FakeFromConnStringCM:
    """Patch ``from_conn_string`` to return a fake CM wrapping ``saver``."""
    cm = _FakeFromConnStringCM(saver, enter_exc)
    mock_cls.from_conn_string = MagicMock(return_value=cm)
    return cm


def _saver() -> MagicMock:
    saver = MagicMock()
    saver.setup = AsyncMock()
    return saver


class TestCreateCheckpointer:
    """Tests for the create_checkpointer factory function."""

    @pytest.mark.asyncio
    async def test_enters_cm_returns_saver_and_setups(self) -> None:
        """The CM is entered, the yielded saver is returned, and setup() runs."""
        saver = _saver()
        with patch(_CHECKPOINT_SAVER_PATH) as mock_cls:
            cm = _wire(mock_cls, saver)
            result = await create_checkpointer(
                "postgresql://user:pass@localhost:5432/testdb"
            )

        assert cm.entered is True
        assert result is saver
        saver.setup.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_regression_returns_saver_not_the_cm(self) -> None:
        """Regression: must return the saver the CM yields, NOT the CM object.

        The original bug returned ``from_conn_string``'s return value directly
        (the un-entered CM), so ``.setup()`` raised ``AttributeError``. This test
        fails against that broken revision.
        """
        saver = _saver()
        with patch(_CHECKPOINT_SAVER_PATH) as mock_cls:
            cm = _wire(mock_cls, saver)
            result = await create_checkpointer("postgresql://localhost/db")

        assert result is saver
        assert result is not cm
        assert hasattr(result, "setup")

    @pytest.mark.asyncio
    async def test_strips_asyncpg_from_url(self) -> None:
        """``+asyncpg`` is removed from the URL for the psycopg driver."""
        saver = _saver()
        with patch(_CHECKPOINT_SAVER_PATH) as mock_cls:
            _wire(mock_cls, saver)
            await create_checkpointer(
                "postgresql+asyncpg://user:pass@localhost:5432/testdb"
            )

        mock_cls.from_conn_string.assert_called_once_with(
            "postgresql://user:pass@localhost:5432/testdb"
        )

    @pytest.mark.asyncio
    async def test_url_without_asyncpg_unchanged(self) -> None:
        """URLs without ``+asyncpg`` pass through unchanged."""
        saver = _saver()
        with patch(_CHECKPOINT_SAVER_PATH) as mock_cls:
            _wire(mock_cls, saver)
            await create_checkpointer(
                "postgresql://user:pass@localhost:5432/testdb"
            )

        mock_cls.from_conn_string.assert_called_once_with(
            "postgresql://user:pass@localhost:5432/testdb"
        )

    @pytest.mark.asyncio
    async def test_setup_failure_closes_pool_and_raises(self) -> None:
        """A ``setup()`` failure must close the backing pool (CM exit) and raise."""
        saver = _saver()
        saver.setup = AsyncMock(side_effect=RuntimeError("Setup failed"))
        with patch(_CHECKPOINT_SAVER_PATH) as mock_cls:
            cm = _wire(mock_cls, saver)

            with pytest.raises(RuntimeError, match="Setup failed"):
                await create_checkpointer("postgresql://localhost/db")

        assert cm.entered is True
        # Pool must be torn down since the saver is not returned — no leak.
        assert cm.exited is True

    @pytest.mark.asyncio
    async def test_connection_failure_during_enter_raises(self) -> None:
        """Connection errors raised while entering the CM propagate (not None)."""
        saver = _saver()
        with patch(_CHECKPOINT_SAVER_PATH) as mock_cls:
            _wire(mock_cls, saver, enter_exc=ConnectionError("Cannot connect"))

            with pytest.raises(ConnectionError, match="Cannot connect"):
                await create_checkpointer(
                    "postgresql://user:pass@badhost:5432/testdb"
                )

    @pytest.mark.asyncio
    async def test_pool_kept_open_on_success(self) -> None:
        """On success the CM is NOT exited — the pool stays live for reuse."""
        saver = _saver()
        with patch(_CHECKPOINT_SAVER_PATH) as mock_cls:
            cm = _wire(mock_cls, saver)
            await create_checkpointer("postgresql://localhost/db")

        assert cm.entered is True
        assert cm.exited is False
