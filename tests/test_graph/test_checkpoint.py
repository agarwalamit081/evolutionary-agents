"""Tests for src.graph.checkpoint — AsyncPostgresSaver checkpoint factory.

Regression note (two iterations of the same latent bug, both now fixed):

1. ``langgraph-checkpoint-postgres`` 3.x's ``AsyncPostgresSaver.from_conn_string``
   is an ``@asynccontextmanager`` that *yields* the saver — it does NOT return one.
   An early revision called ``.setup()`` on the un-entered CM object
   (``AttributeError``), swallowed by ``_create_checkpointer``'s bare
   ``except Exception:`` so *every* CLI run ran checkpoint-less and ``--resume``
   was a silent no-op.

2. The fix for (1) entered the CM via ``cm.__aenter__()`` and returned the yielded
   saver. But the CM holds the underlying connection in a *generator-local*
   variable; the ``cm`` object is ref-counted to zero the instant
   ``create_checkpointer`` returns, so CPython finalizes the generator, runs the
   ``async with``'s ``__aexit__``, and **closes the connection** — leaving the
   saver holding a dead connection that raised
   ``OperationalError("the connection is closed")`` on the first checkpoint read
   (``aget_tuple`` inside ``AsyncPregelLoop.__aenter__``). This broke BOTH the live
   worker e2e and the production worker (identical ``execute_run`` path); the prior
   tests passed only because their fake CM did not model generator finalization
   (``test_pool_kept_open_on_success`` asserted ``cm.exited is False`` — the broken
   assumption itself).

The current fix opens the connection DIRECTLY (no context manager) so its lifetime
is bound to ``saver.conn`` (not a GC'd generator frame), plus a ``close_checkpointer``
the caller runs in its ``finally`` so a long-lived worker doesn't leak one
connection per run. These tests model that direct-connection shape.
"""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from psycopg.rows import dict_row

from src.graph.checkpoint import close_checkpointer, create_checkpointer

# The checkpoint module imports its deps inside the function body via
# ``from psycopg import AsyncConnection`` / ``from langgraph... import
# AsyncPostgresSaver``, so patching the attribute on the source module is what
# the ``from ... import`` resolves at call time.
_CONN_PATH = "psycopg.AsyncConnection"
_SAVER_PATH = "langgraph.checkpoint.postgres.aio.AsyncPostgresSaver"


@pytest.fixture
def wired() -> Iterator[SimpleNamespace]:
    """Patch ``AsyncConnection`` + ``AsyncPostgresSaver`` with a fake connection.

    Yields the fake ``conn`` (with an ``AsyncMock`` ``close``), the fake ``saver``
    (``conn`` pinned to ``conn``, ``setup`` as an ``AsyncMock``), and the two class
    mocks so tests can assert call args / side-effects. ``connect`` returns the
    fake ``conn`` by default; a test overrides it via ``wired.conn_cls.connect``.
    """
    conn = MagicMock(name="conn")
    conn.close = AsyncMock(name="conn.close")
    saver = MagicMock(name="saver")
    saver.conn = conn
    saver.setup = AsyncMock(name="saver.setup")
    with patch(_CONN_PATH) as mock_conn_cls, patch(_SAVER_PATH) as mock_saver_cls:
        mock_conn_cls.connect = AsyncMock(return_value=conn)
        mock_saver_cls.return_value = saver
        yield SimpleNamespace(
            conn=conn, saver=saver, conn_cls=mock_conn_cls, saver_cls=mock_saver_cls
        )


class TestCreateCheckpointer:
    """Tests for the create_checkpointer factory function."""

    @pytest.mark.asyncio
    async def test_connects_returns_saver_and_setups(self, wired) -> None:
        """Connection opened directly, saver built with it, setup() runs, saver returned."""
        result = await create_checkpointer(
            "postgresql://user:pass@localhost:5432/testdb"
        )

        wired.conn_cls.connect.assert_awaited_once()  # opened directly (no CM)
        assert result is wired.saver
        wired.saver_cls.assert_called_once_with(conn=wired.conn)
        wired.saver.setup.assert_awaited_once()
        # On success the connection is left OPEN for close_checkpointer to release.
        wired.conn.close.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_strips_asyncpg_from_url(self, wired) -> None:
        """``+asyncpg`` is removed from the URL for the psycopg driver."""
        await create_checkpointer(
            "postgresql+asyncpg://user:pass@localhost:5432/testdb"
        )

        wired.conn_cls.connect.assert_called_once_with(
            "postgresql://user:pass@localhost:5432/testdb",
            autocommit=True,
            prepare_threshold=0,
            row_factory=dict_row,
        )

    @pytest.mark.asyncio
    async def test_url_without_asyncpg_unchanged(self, wired) -> None:
        """URLs without ``+asyncpg`` pass through unchanged."""
        await create_checkpointer("postgresql://user:pass@localhost:5432/testdb")

        wired.conn_cls.connect.assert_called_once_with(
            "postgresql://user:pass@localhost:5432/testdb",
            autocommit=True,
            prepare_threshold=0,
            row_factory=dict_row,
        )

    @pytest.mark.asyncio
    async def test_setup_failure_closes_connection_and_raises(self, wired) -> None:
        """A ``setup()`` failure must close the opened connection (no leak) and raise."""
        wired.saver.setup.side_effect = RuntimeError("Setup failed")

        with pytest.raises(RuntimeError, match="Setup failed"):
            await create_checkpointer("postgresql://localhost/db")

        wired.conn_cls.connect.assert_awaited_once()  # connection was opened
        wired.conn.close.assert_awaited_once()  # then released → no leak

    @pytest.mark.asyncio
    async def test_connect_failure_raises_no_saver_built(self, wired) -> None:
        """A connection error during ``connect`` propagates (no saver, no leak)."""
        wired.conn_cls.connect.side_effect = ConnectionError("Cannot connect")

        with pytest.raises(ConnectionError, match="Cannot connect"):
            await create_checkpointer(
                "postgresql://user:pass@badhost:5432/testdb"
            )

        wired.conn_cls.connect.assert_awaited_once()
        wired.saver_cls.assert_not_called()  # no saver built when connect fails

    @pytest.mark.asyncio
    async def test_regression_returns_saver_not_connection(self, wired) -> None:
        """Regression: must return the saver, NOT the raw connection object."""
        result = await create_checkpointer("postgresql://localhost/db")

        assert result is wired.saver
        assert result is not wired.conn


class TestCloseCheckpointer:
    """Tests for the close_checkpointer lifecycle counterpart."""

    @pytest.mark.asyncio
    async def test_closes_saver_connection(self) -> None:
        """Closing a built checkpointer awaits ``saver.conn.close()``."""
        conn = MagicMock()
        conn.close = AsyncMock()
        saver = MagicMock()
        saver.conn = conn

        await close_checkpointer(saver)

        conn.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_none_is_noop(self) -> None:
        """Closing None (checkpointer-less run) must not raise."""
        await close_checkpointer(None)

    @pytest.mark.asyncio
    async def test_no_closable_connection_is_noop(self) -> None:
        """A checkpointer whose ``conn`` has no ``close`` is a safe no-op."""
        saver = MagicMock()
        saver.conn = "not-a-connection"
        await close_checkpointer(saver)  # str has no .close → skipped, no raise

    @pytest.mark.asyncio
    async def test_close_error_is_swallowed(self) -> None:
        """A failing ``close`` never propagates (called from a ``finally``)."""
        conn = MagicMock()
        conn.close = AsyncMock(side_effect=RuntimeError("already closed"))
        saver = MagicMock()
        saver.conn = conn

        await close_checkpointer(saver)  # must not raise
