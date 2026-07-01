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
from typing import TYPE_CHECKING, Any, Coroutine, cast
from uuid import uuid4

from loguru import logger

from src.llm.exceptions import BudgetExhaustedError
from src.runner import RunCancelled
from src.tools._paths import run_subdir_path
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

# Grace given to a cancelled ``exec_task`` to unwind before it is abandoned. The
# out-of-band watchdog enforces the terminal TIMEOUT at the deadline REGARDLESS
# (it never waits on the executor); this grace only bounds how long ``_process``
# blocks on best-effort cleanup of a cancellation-absorbing executor. Module
# constant (not a setting) so tests can shrink it for speed; the leaked task is
# reaped at worker restart if it never honors cancellation within the grace.
_EXEC_CANCEL_GRACE_S: float = 5.0


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
            # Hoisted so the outer ``finally`` can cancel a still-running
            # ``exec_task`` if ``_process`` returns early (terminal guard below)
            # or is cancelled while parked in ``asyncio.wait``.
            exec_task: asyncio.Task[dict[str, Any]] | None = None

            # Idempotent terminal guard (Fix A1): a DUPLICATE stream entry for a
            # run that ALREADY reached a terminal state on a prior delivery must
            # NOT re-execute the goal. Without this, a duplicate entry re-ran a
            # FINISHED run from its checkpoint on every redelivery (the q04/q06
            # pathology: a completed run re-executed for minutes per duplicate,
            # accumulating cost on a goal already done). ``FAILED`` is
            # intentionally excluded — a transient failure stays retryable up to
            # the dead-letter cap. Best-effort: a status-store miss (None / Redis
            # hiccup) falls through to normal processing.
            #
            # Gated behind the per-run lease (``lock_ttl_s > 0``): the lease and
            # this guard are BOTH claim-eligibility dedup gates, so the operator's
            # ``lock_ttl_s <= 0`` opt-out (legacy single-worker mode) disables BOTH
            # — preserving the documented "no skip" contract of lease-off mode.
            # Production always runs with the lease on, so the redelivery-forever
            # fix (q04/q06) is active where it matters.
            if self._s.lock_ttl_s > 0:
                prior = await self._status.get(run_id)
                if prior is not None and prior.status in (
                    JobStatus.COMPLETED,
                    JobStatus.CANCELLED,
                    JobStatus.BUDGET_EXHAUSTED,
                    JobStatus.TIMEOUT,
                ):
                    logger.info(
                        f"Run {run_id} ({entry_id}) already terminal "
                        f"({prior.status.value}); acking duplicate entry without "
                        f"re-running"
                    )
                    acked = await self._queue.ack_and_delete([entry_id])
                    return acked > 0  # terminal — entry removed; lease released in finally

            # Surface the per-run output folder through the status hash so a
            # caller discovers the artifact location (``results/<run_id>/``)
            # without guessing — deliverables live in a per-run subdir, not the
            # flat results/ root. Observability-only: a bad run_id / settings
            # hiccup must never break the run (empty string is a safe no-op).
            results_dir = ""
            try:
                results_dir = str(run_subdir_path(run_id))
            except (ValueError, OSError) as exc:
                logger.debug("results_dir resolution skipped for {}: {}", run_id, exc)
            await self._status.mark(
                run_id, thread_id, JobStatus.RUNNING, results_dir=results_dir
            )
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

            # ``exec_task`` was hoisted to the top of this ``try`` (above the
            # terminal guard) so both the guard's early ``return`` and a
            # serve_forever shutdown/cancel can be cleaned up by the outer
            # ``finally``. Do NOT re-declare it here.
            try:
                # ── Out-of-band run-timeout watchdog (Phase 1) ──────────────────
                # The run is RACED: ``exec_task`` vs ``_watchdog``. ``_watchdog``
                # sleeps to the deadline then enforces a terminal TIMEOUT with
                # PLAIN REDIS CALLS — it does NOT depend on ``exec_task`` honoring
                # cancellation. This closes the q04 3h-stall: a stalled run
                # absorbed the in-band ``asyncio.timeout`` cancellation somewhere
                # in the execute_run↔gateway stack, so no terminal mark ever fired
                # and ``_renew_lease`` (created OUTSIDE the old timeout ctx) kept
                # the lease alive ~1h28m past the ceiling. The watchdog cancels the
                # renewal on the deadline so the zombie stops renewing → a peer can
                # reclaim the entry. When unarmed (``timeout_s == 0``, the shipped
                # default) there is no watchdog and ``exec_task`` resolves on its
                # own (prior behavior). The gateway already bounds each socket read
                # at ``request_timeout``; the watchdog covers the case where even
                # that absorption hangs.
                timeout_s = self._resolve_timeout(job)
                # The injected executor is typed ``Awaitable`` (so fakes may be a
                # bare awaitable), but the production executor is an ``async def``
                # → a coroutine. ``asyncio.create_task`` requires a coroutine, so
                # cast at the single call site rather than narrowing the executor
                # type alias (which would forbid legitimate bare-awaitable fakes).
                exec_task = asyncio.create_task(
                    cast(
                        Coroutine[Any, Any, dict[str, Any]],
                        self._executor(job, _report_progress),
                    )
                )
                watchdog: asyncio.Task[bool] | None = None
                if timeout_s > 0:
                    deadline = asyncio.get_running_loop().time() + timeout_s
                    watchdog = asyncio.create_task(
                        self._watchdog(
                            run_id,
                            thread_id,
                            entry_id,
                            token,
                            timeout_s,
                            deadline,
                            exec_task,
                            renewal,
                        )
                    )

                wait_set: set[asyncio.Task[Any]] = {exec_task}
                if watchdog is not None:
                    wait_set.add(watchdog)
                await asyncio.wait(wait_set, return_when=asyncio.FIRST_COMPLETED)

                # Watchdog priority: if it fired AND enforced, the terminal TIMEOUT
                # work (mark + ack + release + cancel renewal) is already done
                # out-of-band. ``exec_task`` is still hung, so best-effort cancel
                # it (abandon if it won't honor cancellation within a short grace).
                if watchdog is not None and watchdog.done():
                    enforced = False
                    try:
                        enforced = watchdog.result()
                    except Exception as exc:  # noqa: BLE001 — never mask the run
                        logger.error(
                            f"Run {run_id} watchdog crashed (timeout NOT enforced): {exc}"
                        )
                    if enforced:
                        if not exec_task.done():
                            exec_task.cancel()
                            # BOUNDED best-effort unwind — NOT ``wait_for``:
                            # ``wait_for`` internally ``await``s the task after its
                            # own timeout (``_cancel_and_wait``), so a cancellation-
                            # ABSORBING executor would hang ``_process`` here — the
                            # exact q04 stall the watchdog exists to end. ``asyncio.wait``
                            # returns at ``timeout`` WITHOUT awaiting pending tasks, so
                            # an absorbing task is abandoned (reaped at worker restart);
                            # the run is already terminal (TIMEOUT + acked + released).
                            await asyncio.wait(
                                {exec_task}, timeout=_EXEC_CANCEL_GRACE_S
                            )
                        exec_task = None  # handled — skip double-cancel in finally
                        return True  # watchdog already acked + released the lease

                # ``exec_task`` resolved first (normal completion, a typed
                # exception, or a stray). Cancel the deadline watchdog — the outer
                # ``finally`` cancels the lease renewal.
                if watchdog is not None and not watchdog.done():
                    watchdog.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await watchdog

                # Unwrap the executor outcome via the typed handlers (the executor
                # returned a result OR raised). Only the watchdog path cancels
                # ``exec_task``, so it is not cancelled here.
                if exec_task.cancelled():  # defensive — treat as a generic failure
                    raise asyncio.CancelledError("exec_task cancelled unexpectedly")
                exc = exec_task.exception()
                if exc is not None:
                    if isinstance(exc, RunCancelled):
                        # Graceful cancel (E): a Redis flag polled at the
                        # per-iteration progress callback → RunCancelled with
                        # ~1-iteration latency. Terminal + acked (NOT redelivered).
                        logger.info(f"Run {run_id} cancelled (terminal): {exc}")
                        await self._status.mark(
                            run_id, thread_id, JobStatus.CANCELLED, error=str(exc)
                        )
                        acked = await self._queue.ack_and_delete([entry_id])
                        return acked > 0  # terminal
                    if isinstance(exc, BudgetExhaustedError):
                        # Opt-in budget hard-stop (D): the gateway raised instead
                        # of downgrading. Terminal + acked (resumable via
                        # checkpoint — caveat: cumulative cap re-trips on resume).
                        logger.warning(
                            f"Run {run_id} budget-exhausted (terminal, opt-in "
                            f"hard_stop): {exc}"
                        )
                        await self._status.mark(
                            run_id, thread_id, JobStatus.BUDGET_EXHAUSTED, error=str(exc)
                        )
                        acked = await self._queue.ack_and_delete([entry_id])
                        return acked > 0  # terminal — resumable
                    if isinstance(exc, TimeoutError) and timeout_s <= 0:
                        # F (fallthrough): a STRAY downstream TimeoutError (e.g. an
                        # httpx read timeout) while the wall-clock bound is UNARMED
                        # → dead-letter (re-raise), NOT mislabeled TIMEOUT.
                        raise exc
                    raise exc  # generic exception → dead-letter handler below
                result = exec_task.result()
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
                    acked = await self._queue.ack_and_delete([entry_id])
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
            acked = await self._queue.ack_and_delete([entry_id])
            logger.info(f"Run {run_id} completed (acked={acked})")
            return acked > 0
        finally:
            # If ``_process`` is exiting with the executor still running (e.g. a
            # serve_forever shutdown cancelled the supervisor while it was parked
            # in ``asyncio.wait``), cancel the task so it does not leak. The
            # watchdog-enforced path sets ``exec_task = None`` after its own
            # best-effort cancel, so this is a no-op there. A truly stuck executor
            # (absorbs cancellation) is abandoned after the grace — reaped at
            # worker restart — the lease/status are already terminal by then.
            if exec_task is not None and not exec_task.done():
                exec_task.cancel()
                # Bounded cleanup, same reason as the enforced path: ``wait_for``
                # would hang on an absorbing executor. ``asyncio.wait`` returns at
                # the timeout; a still-pending task is abandoned + reaped at restart.
                await asyncio.wait({exec_task}, timeout=_EXEC_CANCEL_GRACE_S)
            renewal.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await renewal
            # Compare-and-del: a stale holder (TTL expired + peer reacquired) must
            # NOT evict the live owner. Best-effort; the TTL is the ultimate net.
            try:
                await self._queue.release_lock(run_id, token)
            except Exception as exc:  # noqa: BLE001 — release must never mask the result
                logger.warning(f"Run {run_id} lease release failed: {exc}")

    async def _watchdog(
        self,
        run_id: str,
        thread_id: str,
        entry_id: str,
        token: str,
        timeout_s: float,
        deadline: float,
        exec_task: asyncio.Task[dict[str, Any]],
        renewal: asyncio.Task[None],
    ) -> bool:
        """Out-of-band run-timeout enforcement. Returns True iff it enforced a
        terminal TIMEOUT (so ``_process`` knows the run is already acked + the
        lease released and must not double-handle).

        Sleeps until ``deadline``; if the executor is STILL running then (the run
        exceeded its wall-clock bound), enforces a terminal TIMEOUT with PLAIN
        REDIS CALLS — mark + ack + release + cancel the lease renewal — INDEPENDENT
        of whether ``exec_task`` honors cancellation. This is the q04 3h-stall fix:
        a stalled run absorbed the in-band ``asyncio.timeout`` cancellation
        somewhere in the execute_run↔gateway stack, so no terminal mark ever fired
        and ``_renew_lease`` (created outside the old timeout ctx) kept the lease
        alive ~1h28m past the ceiling. The watchdog cancels the renewal on the
        deadline so the zombie stops renewing → a peer can reclaim the entry.

        Race guard: if the executor finished (normally OR via the cooperative
        timeout) by the wake instant, it wins — return False and do NOT clobber its
        terminal outcome (likely COMPLETED / a typed exception). Plain Redis calls
        only — no LLM, no gateway — so the watchdog itself can never hang; each
        step is best-effort so a partial Redis failure cannot leave it half-done.
        """
        loop = asyncio.get_running_loop()
        await asyncio.sleep(max(0.0, deadline - loop.time()))
        # Race guard: exec finished while we slept / at the wake instant → it wins.
        if exec_task.done():
            return False
        logger.warning(
            f"Run {run_id} exceeded its {timeout_s}s wall-clock bound (still "
            f"running) — enforcing terminal TIMEOUT out-of-band (resumable via "
            f"--resume)"
        )
        # Stop the zombie renewing the lease so a peer can take the entry over.
        renewal.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await renewal
        try:
            await self._status.mark(
                run_id,
                thread_id,
                JobStatus.TIMEOUT,
                error=f"Run timeout after {timeout_s}s",
            )
        except Exception as exc:  # noqa: BLE001 — observability-only, never mask
            logger.warning(f"Run {run_id} watchdog status-mark failed: {exc}")
        try:
            await self._queue.ack_and_delete([entry_id])
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Run {run_id} watchdog ack failed: {exc}")
        try:
            await self._queue.release_lock(run_id, token)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Run {run_id} watchdog lease release failed: {exc}")
        return True

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
