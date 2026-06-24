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
import contextlib
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from loguru import logger

from src.llm.exceptions import BudgetExhaustedError
from src.runner import RunCancelled
from src.worker.queue import RunsQueue
from src.worker.schema import JobStatus, RunJob
from src.worker.status import RunStatusStore

if TYPE_CHECKING:
    from src.config.settings import WorkerSettings

# A run executor: given a queued job (+ an optional progress callback), run the
# agent and return its final state dict (final_output / is_complete /
# iteration_count). The default production executor calls src.runner.execute_run
# (see src.worker.executors). The progress callback lets the executor stream the
# live iteration_count back mid-run so the run-status hash reflects progress
# (#255); it may be ignored (fakes) or None (no progress reporting).
RunExecutor = Callable[
    [RunJob, Callable[[int], Awaitable[None]] | None], Awaitable[dict[str, Any]]
]


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

    def _resolve_timeout(self, job: RunJob) -> float:
        """Resolve the per-run wall-clock timeout (seconds); ``0`` = disabled.

        ``RunJob.run_timeout_s`` (the API per-run override) wins over the worker
        default ``WorkerSettings.run_timeout_s``. An explicit ``0`` disables even
        when the worker default is set; ``None`` falls back to the worker default
        (which itself defaults to ``0`` — no timeout, the current behavior). A
        negative value is treated as disabled.
        """
        per_run = job.run_timeout_s
        timeout = (
            float(per_run) if per_run is not None else float(self._s.run_timeout_s or 0.0)
        )
        return timeout if timeout > 0 else 0.0

    async def _process(self, entry_id: str, job: RunJob) -> bool:
        """Run one job. Returns True iff the entry was acked (terminal).

        Lease (Bug C): acquires the per-run lock before any work; if another
        worker already holds it, SKIP (return False WITHOUT acking — the rightful
        owner's XACK removes the entry group-wide, so a skipped entry does not
        accumulate). Renewed every ttl/3 while the run is live and released in
        ``finally`` on every exit path.

        On a successful run: mirror the result to the status store, XACK, return
        True. On an executor exception: count the attempt; below
        ``dead_letter_max_attempts`` mark FAILED and DO NOT ack — the entry stays
        pending and is redelivered (reclaim_stale) so a TRANSIENT failure is
        retried / resumed from the last checkpoint. AT the cap (a deterministic /
        poison failure) XACK + mark FAILED permanently so it is NOT retried
        infinitely (Bug B: without this a deterministic crash — e.g. a missing dep
        at graph build — was redelivered every ``reclaim_min_idle_ms`` forever).
        """
        run_id = job.run_id
        thread_id = self.thread_id_for(run_id)

        # ── Per-run lease lock (Bug C) ────────────────────────────────────────
        # ``reclaim_min_idle_ms`` (XAUTOCLAIM) is shorter than a normal run, so a
        # peer worker would otherwise steal this still-healthy in-flight entry and
        # run the SAME goal a second time. Acquire SET-NX first; if held, skip.
        token = uuid4().hex
        if not await self._queue.try_lock(run_id, token, self._s.lock_ttl_s):
            logger.info(
                f"Run {run_id} ({entry_id}) already in-flight on another worker "
                f"(lease held); skipping duplicate claim"
            )
            return False  # NOT acked — the rightful owner acks group-wide

        renewal = asyncio.create_task(
            self._renew_lease(run_id, token, max(1.0, self._s.lock_ttl_s / 3.0))
        )
        try:
            await self._status.mark(run_id, thread_id, JobStatus.RUNNING)
            logger.info(f"Worker claimed run {run_id} ({entry_id}): {job.goal[:80]}")

            # #255: report the live iteration_count into the status hash mid-run so
            # GET /runs/<id> reflects progress (previously 0 until completion). The
            # callback mirrors each increment onto the RUNNING record; the executor
            # may ignore it (fakes), and the COMPLETED mark below still stamps the
            # authoritative final count. execute_run streams only while a callback
            # is passed, so the CLI path (callback=None) is unchanged.
            async def _report_progress(iteration: int) -> None:
                await self._status.mark(
                    run_id, thread_id, JobStatus.RUNNING, iteration_count=iteration
                )
                # E (cancel checkpoint): poll the Redis cancel flag each
                # iteration and raise ``RunCancelled`` when set → propagates
                # through execute_run's progress-callback path (src/runner.py
                # re-raises it rather than swallowing as observability-only) to
                # this run's ``except RunCancelled`` handler → terminal CANCELLED
                # + acked. ~1-iteration latency (fast on q09-style cap-loop
                # churn). ``is_cancelled`` fails open on Redis error, so a blip
                # never spuriously kills a healthy run.
                if await self._status.is_cancelled(run_id):
                    raise RunCancelled(f"cancelled via POST /runs/{run_id}/cancel")

            try:
                # ── Run-level wall-clock timeout (A, opt-in) ──────────────────
                # When ``_resolve_timeout`` returns >0, wrap the executor in
                # ``asyncio.timeout``; the inner ``except TimeoutError`` then
                # marks a terminal TIMEOUT (acked, NOT redelivered — the
                # per-iteration AsyncPostgresSaver checkpoint lets ``--resume``
                # continue later). When unarmed (timeout_s == 0) we still enter
                # the inner ``except`` on a *stray* downstream TimeoutError, but
                # ``re-raise`` it so it falls through to the typed/dead-letter
                # handlers below — preserving the prior default-path behavior.
                timeout_s = self._resolve_timeout(job)
                timeout_ctx = (
                    asyncio.timeout(timeout_s) if timeout_s > 0 else contextlib.nullcontext()
                )
                try:
                    # ``asyncio.timeout`` (a ``Timeout``) and ``nullcontext`` both
                    # implement the async CM protocol, so ``async with`` is the
                    # correct unifier (``with`` is rejected by the ``Timeout``
                    # stub, which only exposes ``__aenter__``/``__aexit__``).
                    async with timeout_ctx:
                        result = await self._executor(job, _report_progress)
                except TimeoutError:
                    if timeout_s <= 0:
                        raise  # stray downstream TimeoutError — dead-letter, not ours
                    logger.warning(
                        f"Run {run_id} timed out after {timeout_s}s (terminal, "
                        f"resumable via --resume)"
                    )
                    await self._status.mark(
                        run_id,
                        thread_id,
                        JobStatus.TIMEOUT,
                        error=f"Run timeout after {timeout_s}s",
                    )
                    acked = await self._queue.ack([entry_id])
                    return acked > 0  # terminal — resumable via checkpoint
                except RunCancelled as exc:
                    # Graceful cancel (E): a Redis flag set via POST
                    # /runs/{run_id}/cancel, polled at the per-iteration progress
                    # callback inside execute_run → raises RunCancelled with
                    # ~1-iteration latency. Terminal + acked (NOT redelivered).
                    logger.info(f"Run {run_id} cancelled (terminal): {exc}")
                    await self._status.mark(
                        run_id,
                        thread_id,
                        JobStatus.CANCELLED,
                        error=str(exc),
                    )
                    acked = await self._queue.ack([entry_id])
                    return acked > 0  # terminal
                except BudgetExhaustedError as exc:
                    # Opt-in budget hard-stop (D): the gateway raised instead of
                    # downgrading to a cheaper fallback. Terminal + acked
                    # (resumable via checkpoint — caveat: cumulative budget
                    # re-trips on resume; deferred follow-up).
                    logger.warning(
                        f"Run {run_id} budget-exhausted (terminal, opt-in "
                        f"hard_stop): {exc}"
                    )
                    await self._status.mark(
                        run_id,
                        thread_id,
                        JobStatus.BUDGET_EXHAUSTED,
                        error=str(exc),
                    )
                    acked = await self._queue.ack([entry_id])
                    return acked > 0  # terminal — resumable
            except Exception as exc:
                # Dead-letter cap (Bug B). ``record_attempt`` is keyed by run_id, so
                # the count is stable across XAUTOCLAIM redelivery (same consumer or a
                # reclaimed peer). At the cap the entry is acked (removed from the PEL)
                # so reclaim_stale can never hand it out again.
                attempts = await self._queue.record_attempt(run_id)
                if attempts >= self._s.dead_letter_max_attempts:
                    logger.error(
                        f"Run {run_id} dead-lettered after {attempts} failed "
                        f"attempts (cap={self._s.dead_letter_max_attempts}); acking "
                        f"to stop redelivery: {exc}"
                    )
                    await self._status.mark(
                        run_id,
                        thread_id,
                        JobStatus.FAILED,
                        error=f"{exc} (dead-lettered after {attempts} attempts)",
                    )
                    acked = await self._queue.ack([entry_id])
                    return acked > 0  # terminal — NOT redelivered
                logger.warning(
                    f"Run {run_id} executor raised (attempt {attempts}/"
                    f"{self._s.dead_letter_max_attempts}); leaving for redelivery: {exc}"
                )
                await self._status.mark(
                    run_id, thread_id, JobStatus.FAILED, error=str(exc)
                )
                return False  # NOT acked → redelivered by reclaim_stale

            await self._status.mark(
                run_id,
                thread_id,
                JobStatus.COMPLETED,
                final_output=str(result.get("final_output", "")),
                is_complete=bool(result.get("is_complete", False)),
                iteration_count=int(result.get("iteration_count", 0) or 0),
            )
            acked = await self._queue.ack([entry_id])
            logger.info(f"Run {run_id} completed (acked={acked})")
            return acked > 0
        finally:
            renewal.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await renewal
            # Compare-and-del: a stale holder (TTL expired + peer reacquired) must
            # NOT evict the live owner. Best-effort; the TTL is the ultimate net.
            try:
                await self._queue.release_lock(run_id, token)
            except Exception as exc:  # noqa: BLE001 — release must never mask the result
                logger.warning(f"Run {run_id} lease release failed: {exc}")

    async def _renew_lease(self, run_id: str, token: str, interval: float) -> None:
        """Background lease renewal — extends the TTL every ``interval`` while the
        run is live, so a long run never lets the lease expire (which would let a
        peer steal it via reclaim). Exits if the lock is lost (renew returned
        False): logged WARNING; the run is NOT aborted mid-flight, but another
        worker may then take it over once it ends. Cancelled by ``_process``'s
        finally on completion.
        """
        try:
            while True:
                await asyncio.sleep(interval)
                try:
                    held = await self._queue.renew_lock(
                        run_id, token, self._s.lock_ttl_s
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 — best-effort; keep looping
                    logger.warning(
                        f"Run {run_id} lease renewal error (retrying next tick): {exc}"
                    )
                    continue
                if not held:
                    logger.warning(
                        f"Run {run_id} lease lost mid-run (expired/evicted) — a peer "
                        f"worker may take it over"
                    )
                    return
        except asyncio.CancelledError:
            raise

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
