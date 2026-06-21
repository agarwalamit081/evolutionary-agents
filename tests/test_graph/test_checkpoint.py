"""Tests for src.graph.checkpoint — AsyncPostgresSaver checkpoint factory."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.graph.checkpoint import create_checkpointer

# The checkpoint module imports AsyncPostgresSaver inside the function body
# via: from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
# So we must patch it at its source module path.
_CHECKPOINT_SAVER_PATH = "langgraph.checkpoint.postgres.aio.AsyncPostgresSaver"


class TestCreateCheckpointer:
    """Tests for create_checkpointer factory function."""

    @pytest.mark.asyncio
    async def test_successful_creation(self) -> None:
        """Should create and return an initialized AsyncPostgresSaver."""
        mock_saver = MagicMock()
        mock_saver.setup = AsyncMock()

        with patch(_CHECKPOINT_SAVER_PATH) as mock_cls:
            mock_cls.from_conn_string = MagicMock(return_value=mock_saver)

            result = await create_checkpointer(
                "postgresql://user:pass@localhost:5432/testdb"
            )

            assert result is mock_saver
            mock_saver.setup.assert_called_once()

    @pytest.mark.asyncio
    async def test_strips_asyncpg_from_url(self) -> None:
        """Should remove '+asyncpg' from the connection string for psycopg."""
        mock_saver = MagicMock()
        mock_saver.setup = AsyncMock()

        with patch(_CHECKPOINT_SAVER_PATH) as mock_cls:
            mock_cls.from_conn_string = MagicMock(return_value=mock_saver)

            await create_checkpointer(
                "postgresql+asyncpg://user:pass@localhost:5432/testdb"
            )

            mock_cls.from_conn_string.assert_called_once_with(
                "postgresql://user:pass@localhost:5432/testdb"
            )

    @pytest.mark.asyncio
    async def test_url_without_asyncpg_unchanged(self) -> None:
        """URLs without '+asyncpg' should pass through unchanged."""
        mock_saver = MagicMock()
        mock_saver.setup = AsyncMock()

        with patch(_CHECKPOINT_SAVER_PATH) as mock_cls:
            mock_cls.from_conn_string = MagicMock(return_value=mock_saver)

            await create_checkpointer(
                "postgresql://user:pass@localhost:5432/testdb"
            )

            mock_cls.from_conn_string.assert_called_once_with(
                "postgresql://user:pass@localhost:5432/testdb"
            )

    @pytest.mark.asyncio
    async def test_setup_called_on_checkpointer(self) -> None:
        """setup() must be called on the created checkpointer instance."""
        mock_saver = MagicMock()
        mock_saver.setup = AsyncMock()

        with patch(_CHECKPOINT_SAVER_PATH) as mock_cls:
            mock_cls.from_conn_string = MagicMock(return_value=mock_saver)

            await create_checkpointer("postgresql://localhost/db")

            mock_saver.setup.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_connection_failure_raises(self) -> None:
        """Should propagate connection errors (not return None)."""
        with patch(_CHECKPOINT_SAVER_PATH) as mock_cls:
            mock_cls.from_conn_string = MagicMock(
                side_effect=ConnectionError("Cannot connect to PostgreSQL")
            )

            with pytest.raises(ConnectionError, match="Cannot connect"):
                await create_checkpointer(
                    "postgresql://user:pass@badhost:5432/testdb"
                )

    @pytest.mark.asyncio
    async def test_setup_failure_raises(self) -> None:
        """Should propagate setup errors."""
        mock_saver = MagicMock()
        mock_saver.setup = AsyncMock(side_effect=RuntimeError("Setup failed"))

        with patch(_CHECKPOINT_SAVER_PATH) as mock_cls:
            mock_cls.from_conn_string = MagicMock(return_value=mock_saver)

            with pytest.raises(RuntimeError, match="Setup failed"):
                await create_checkpointer("postgresql://localhost/db")
