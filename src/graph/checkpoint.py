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

    # In langgraph-checkpoint-postgres 3.x, ``from_conn_string`` is an
    # ``@asynccontextmanager`` that yields the saver (owning its connection
    # pool) on enter and closes the pool on exit — it does NOT return a saver
    # directly. Enter it to obtain the saver, call ``setup()`` to create the
    # checkpoint tables, then return the detached saver so its backing pool is
    # reused across checkpoint writes for the process. If ``setup()`` fails,
    # close the pool via the CM's ``__aexit__`` before propagating.
    cm = AsyncPostgresSaver.from_conn_string(psycopg_url)
    checkpointer = await cm.__aenter__()
    try:
        await checkpointer.setup()
    except BaseException:
        await cm.__aexit__(None, None, None)
        raise

    logger.info("AsyncPostgresSaver checkpointer initialized")
    return checkpointer
