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

    The saver owns a single dedicated ``AsyncConnection`` bound to
    ``saver.conn``. The connection's lifetime is therefore bound to the saver,
    NOT to a garbage-collected generator frame — which is the fix for the
    detached-context-manager bug this factory carried for the whole overhaul.

    Why not ``AsyncPostgresSaver.from_conn_string``: in
    ``langgraph-checkpoint-postgres`` 3.x that helper is an
    ``@asynccontextmanager`` that holds the underlying connection in a
    *generator-local* variable::

        async with await AsyncConnection.connect(...) as conn:
            yield cls(conn=conn)

    Entering it via ``cm.__aenter__()`` and returning the yielded saver leaves
    the saver holding ``conn``, but the ``cm`` object is ref-counted to zero the
    instant this factory returns. CPython finalizes the generator, which runs the
    ``async with``'s ``__aexit__`` and **closes the connection** — so the saver is
    left with a dead connection and the very first checkpoint read/write raises
    ``OperationalError("the connection is closed")``. Opening the connection
    directly (no context manager) keeps it alive for as long as the saver is
    referenced. Callers MUST close it via :func:`close_checkpointer` when done so
    a long-lived process (the worker) does not leak one connection per run.
    """
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg import AsyncConnection
    from psycopg.rows import dict_row

    # Convert asyncpg URL to psycopg format if needed.
    psycopg_url = database_url.replace("+asyncpg", "")

    # Open the connection directly (NOT via from_conn_string's async context
    # manager) so it is not closed when a detached CM object is collected. The
    # kwargs mirror from_conn_string exactly: autocommit + prepared-statement
    # caching off (pgvector/psycopg bfloat interop) + dict rows.
    #
    # The two ``type: ignore[arg-type]`` are psycopg-stub covariance limitations,
    # NOT real type errors: ``dict_row`` is the correct async row factory (the
    # row-MAKER protocol is sync even for async cursors) and langgraph's own
    # ``from_conn_string`` makes this identical call. pyright's stubs don't unify
    # ``RowFactory[DictRow]`` with ``AsyncRowFactory[TupleRow]``.
    conn = await AsyncConnection.connect(
        psycopg_url,
        autocommit=True,
        prepare_threshold=0,
        row_factory=dict_row,  # type: ignore[arg-type]
    )
    try:
        saver = AsyncPostgresSaver(conn=conn)  # type: ignore[arg-type]
        await saver.setup()
    except BaseException:
        # Never leak a connection that was opened before setup failed.
        await conn.close()
        raise

    logger.info("AsyncPostgresSaver checkpointer initialized")
    return saver


async def close_checkpointer(checkpointer: Any) -> None:
    """Close the backing connection of a checkpointer built by this factory.

    Idempotent and never raises: safe to call from a ``finally`` block. A no-op
    when ``checkpointer`` is ``None`` or has no closable connection (e.g. a saver
    supplied externally / a mock) so callers don't need to special-case the
    checkpointer-less run.

    This is the lifecycle counterpart to :func:`create_checkpointer`: the worker
    runs many ``execute_run`` calls in its lifetime, and each opens a dedicated
    checkpoint connection that MUST be released here (not left for GC) or the
    process leaks one Postgres connection per run.
    """
    if checkpointer is None:
        return
    conn = getattr(checkpointer, "conn", None)
    close = getattr(conn, "close", None)
    if close is None:
        return
    try:
        await close()
    except Exception as e:  # noqa: BLE001 — best-effort cleanup, never raise out
        logger.debug(f"Checkpointer connection close skipped: {e}")
