"""AsyncPostgresSaver checkpoint factory for LangGraph state persistence.

Provides persistent checkpointing so graph state survives crashes
and can be resumed across runs.
"""

from __future__ import annotations

from typing import Any

from loguru import logger


async def create_checkpointer(database_url: str) -> Any:
    """Create and initialize an AsyncPostgresSaver checkpointer.

    Args:
        database_url: PostgreSQL connection string.

    Returns:
        Initialized AsyncPostgresSaver instance.

    Raises:
        ImportError: If langgraph-checkpoint-postgres is not installed.
        ConnectionError: If PostgreSQL is not reachable.
    """
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    # Convert asyncpg URL to psycopg format if needed
    psycopg_url = database_url.replace("+asyncpg", "")

    checkpointer = AsyncPostgresSaver.from_conn_string(psycopg_url)
    await checkpointer.setup()

    logger.info("AsyncPostgresSaver checkpointer initialized")
    return checkpointer
