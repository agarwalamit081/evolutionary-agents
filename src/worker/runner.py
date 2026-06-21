"""Worker consumer — drains the run queue with at-least-once delivery (Phase 2b).

``RunConsumer`` claims jobs (new entries, then stale ones left by crashed
workers), executes each via an injected ``executor`` callable, and XACKs ONLY
after the run returns — so a crash between claim and ack leaves the entry
pending for ``reclaim_stale`` (XAUTOCLAIM) to hand to another worker, which
resumes from the last checkpoint (``thread_id = api-{run_id}`` is stable across
redelivery). Status is mirrored to the status store so the API can report it.

The ``executor`` is injected (dependency inversion) so the consumer is unit-
testable with a fake and decoupled from the agent run engine.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.worker.queue import RunsQueue
from src.worker.schema import JobStatus, RunJob
from src.worker.status import RunStatusStore

if TYPE_CHECKING:
    from src.config.settings import WorkerSettings

# A run executor: given a queued job, run the agent and return its final state
# dict (final_output / is_complete / iteration_count). The default production
# executor reuses main._run_agent (see src.worker.executors).
RunExecutor = Callable[[RunJob], Awaitable[dict[str, Any]]]


class RunConsumer:
    """Consumes ``RunJob`` entries from the queue, executing them at-least-once."""

    def __init__(
        self,
        queue: RunsQueue,
        status_store: RunStatusStore,
        executor: RunExecutor,
        settings: WorkerSettings,
    ) -> None:
        self._queue = queue
        self._status = status_store
        self._executor = executor
        self._s = settings

    @staticmethod
    def thread_id_for(run_id: str) -> str:
        """Stable checkpoint thread key — the resume handle across redelivery."""
        return f"api-{run_id}"

    async def _process(self, entry_id: str, job: RunJob) -> bool:
        """Run one job. Returns True iff the entry was acked (terminal).

        On a successful run: mirror the result to the status store, XACK, return
        True. On an executor exception: mark FAILED and DO NOT ack — the entry
        stays pending and is redelivered (reclaim_stale) so the run is retried /
        resumed. (Poison-message/dead-letter capping is a Phase 5 concern.)
        """
        thread_id = self.thread_id_for(job.run_id)
        await self._status.mark(job.run_id, thread_id, JobStatus.RUNNING)
        logger.info(f"Worker claimed run {job.run_id} ({entry_id}): {job.goal[:80]}")
        try:
            result = await self._executor(job)
        except Exception as exc:
            logger.warning(f"Run {job.run_id} executor raised; leaving for redelivery: {exc}")
            await self._status.mark(
                job.run_id, thread_id, JobStatus.FAILED, error=str(exc)
            )
            return False  # NOT acked → redelivered by reclaim_stale

        await self._status.mark(
            job.run_id,
            thread_id,
            JobStatus.COMPLETED,
            final_output=str(result.get("final_output", "")),
            is_complete=bool(result.get("is_complete", False)),
            iteration_count=int(result.get("iteration_count", 0) or 0),
        )
        acked = await self._queue.ack([entry_id])
        logger.info(f"Run {job.run_id} completed (acked={acked})")
        return acked > 0

    async def run_once(self) -> int:
        """One drain pass: recover stale entries, then pull new ones.

        Stale-first so a worker that died mid-run is recovered before new work
        is taken (fairness + faster resume). Returns the number of jobs acked.
        """
        await self._queue.ensure_group()
        acked_total = 0
        # 1. Recover entries orphaned by a crashed worker (XAUTOCLAIM).
        for entry_id, job in await self._queue.reclaim_stale():
            if await self._process(entry_id, job):
                acked_total += 1
        # 2. Pull never-delivered entries (XREADGROUP … >).
        for entry_id, job in await self._queue.read_new():
            if await self._process(entry_id, job):
                acked_total += 1
        return acked_total

    async def serve_forever(self, stop_event: asyncio.Event | None = None) -> None:
        """Drain the queue until ``stop_event`` (or CancelledError) is set.

        Loops ``run_once``; the per-read ``block_ms`` bounds idle wakeups. Every
        iteration also reclaims stale work, so a peer worker crash is recovered
        within one loop regardless of new-traffic.
        """
        stop = stop_event or asyncio.Event()
        await self._queue.ensure_group()
        try:
            while not stop.is_set():
                await self.run_once()
        except asyncio.CancelledError:
            logger.info("Worker serve loop cancelled; in-flight jobs stay pending for redelivery")
            raise
