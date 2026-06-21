"""Worker entrypoint — drains the run queue (Phase 3 staged role-split).

Runnable as ``python -m src.worker``. Builds the Redis-backed queue + status
store + ``RunConsumer(default_agent_executor)`` and drains ``turing:runs``
at-least-once, XACKing only after each run's checkpoint is durable — so a crash
between claim and ack leaves the entry pending for ``reclaim_stale``
(XAUTOCLAIM) to hand to a peer, which resumes from the last checkpoint
(``thread_id = api-{run_id}`` is stable across redelivery/restart).

Graceful SIGTERM/SIGINT → ``stop_event``: the current job finishes + acks, then
the loop exits (no new jobs taken). Compose can therefore stop/restart a worker
without losing in-flight work; a hard kill (past ``stop_grace_period``) still
resumes from the checkpoint on the next claim.

Staged scope (owner 2026-06-21): this worker KEEPS Docker-socket access so
Phase-2c isolated code-exec (``CODE_EXECUTOR_MODE=docker``) keeps working. The
remote no-DinD runner (``RunnerClient``) is a deferred follow-up (P3b/c).
"""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import sys
from typing import Any

from loguru import logger

from src.config import get_settings
from src.observability.logging import setup_logging
from src.worker.executors import default_agent_executor
from src.worker.queue import RunsQueue
from src.worker.runner import RunConsumer
from src.worker.status import RunStatusStore


def _resolve_consumer_name(settings_worker: Any) -> str:
    """A unique consumer name per process so replica workers don't share a PEL.

    Redis Streams consumer-group semantics require each consumer to have a
    DISTINCT name: a shared name makes multiple processes own the same pending
    list, breaking ``XAUTOCLAIM`` crash-recovery (a peer can't tell a stuck
    entry belongs to a dead sibling).

    With ``deploy.replicas``, each container has a unique hostname — its short
    container id (the SAME id ``docker ps`` shows), e.g. ``363d6e8d1e6e`` — so
    ``worker-{hostname}-{pid}`` is unique across replicas (hostname dimension)
    AND across restarts (pid dimension), yet stable for the process lifetime so
    pending-entry ownership holds for the whole run. The hostname also makes the
    name MEANINGFUL: a Redis consumer seen via ``XINFO CONSUMERS`` ties straight
    back to a real container (``docker ps | grep <hostname>``).

    An explicit ``WorkerSettings.consumer_name`` (env ``WORKER_CONSUMER_NAME``)
    is honored verbatim — the single-worker fixed-name opt-in. The default is
    empty, which selects auto-derivation (the right choice for a replicated
    worker pool). Set a value ONLY for a single, fixed-name worker; setting one
    for replicas makes every replica collide on that single name.
    """
    if settings_worker.consumer_name:
        return settings_worker.consumer_name
    return f"worker-{socket.gethostname()}-{os.getpid()}"


async def _run() -> int:
    """Build the consumer and drain the queue until stopped. Returns exit code."""
    settings = get_settings()
    setup_logging(settings.logging)
    worker_settings = settings.worker.model_copy(
        update={"consumer_name": _resolve_consumer_name(settings.worker)}
    )
    logger.info(
        f"Worker starting — consumer={worker_settings.consumer_name} "
        f"stream={worker_settings.runs_stream} group={worker_settings.group}"
    )

    import redis.asyncio as aioredis

    redis_client = aioredis.from_url(settings.redis.redis_url)
    # Fail fast + exit 1 if Redis is down: a worker without the queue is useless.
    # compose `depends_on: redis: service_healthy` covers cold start; a later
    # outage → exit 1 → `restart: unless-stopped` retries until Redis recovers.
    try:
        await redis_client.ping()  # type: ignore[union-attr]  # redis.asyncio stub returns sync bool
    except Exception as exc:
        logger.error(f"Redis unreachable at {settings.redis.redis_url}: {exc}")
        try:
            await redis_client.aclose()
        except Exception:  # noqa: BLE001 — best-effort close on the error path
            pass
        return 1

    queue = RunsQueue(redis_client, worker_settings)
    status_store = RunStatusStore(redis_client, worker_settings)
    consumer = RunConsumer(queue, status_store, default_agent_executor, worker_settings)

    stop_event = asyncio.Event()

    def _request_stop(*_: object) -> None:
        logger.info("Worker stop signal received; finishing in-flight job, then exiting")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except (NotImplementedError, RuntimeError):
            # add_signal_handler is unavailable on Windows / non-main threads;
            # fall through — CancelledError (asyncio.run teardown) still stops us.
            pass

    try:
        await consumer.serve_forever(stop_event)
    except asyncio.CancelledError:
        logger.info("Worker cancelled; in-flight jobs stay pending for redelivery")
        raise
    finally:
        await redis_client.aclose()
    return 0


def main() -> None:
    """Module entrypoint: ``python -m src.worker``."""
    sys.exit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
