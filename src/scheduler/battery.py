"""Nightly capability-curve battery enqueue logic (#197 Phase 5).

Pure, importable WITHOUT apscheduler: builds the date-suffixed ``RunJob``
payloads for the golden ``BATTERY04_GOALS`` and enqueues them through the SAME
``RunsQueue.enqueue`` seam the API uses. The APScheduler daemon wiring lives in
``make_battery_scheduler`` (lazy import so this module stays import-light and
unit-testable). The process entrypoint is ``__main__.py``.

Design: reusing ``RunsQueue.enqueue`` (not a hand-rolled XADD) means the
scheduler produces byte-identical stream entries to the API path — the worker's
lease-lock, dead-letter, checkpoint, and eval-resolution machinery all apply
unchanged. Each spec runs under ``run_id = f"{spec_id}-{YYYYMMDD}"``; the verify
node's ``runner._resolve_eval_spec_id`` strips that suffix to score against the
spec.

Cross-query DAG (#575): the battery-04 goals are a DEPENDENCY GRAPH, not 9
independent queries — q02/q03 read q01's output, q06 reads q05's, q04 reads
q1-q3. A goal hardcodes a flat path (``results/q01/normalized.csv``), so it must
run AFTER its upstream and against the upstream's REAL (not stale cross-night)
output. Two changes fix the confound: (1) battery jobs set
``results_per_run_subdir=False`` + ``clear_flat_subdirs=[<qNN>]`` so they share
the flat results root their hardcoded paths expect AND each self-clears its own
write-dir of a prior night's leftovers; (2) ``enqueue_battery`` does a
topological DAG-release — roots enqueue immediately (parallel on the worker
pool), a dependent enqueues only once every upstream reaches a terminal status
(polling the run-status hash), bounded by an overall deadline. This keeps full
worker parallelism (the battery still finishes well before the 05:00 curve-gate)
while guaranteeing a dependent never races its upstream. Per-run results
isolation (the #574 contamination fix) stays active for every non-battery run.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from loguru import logger

from src.config.settings import SchedulerSettings
from src.eval.golden import BATTERY04_GOALS
from src.eval.models import GoalSpec
from src.worker.queue import RunsQueue
from src.worker.schema import JobStatus, RunJob
from src.worker.status import RunStatusStore

if TYPE_CHECKING:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler


# The battery-04 cross-query dependency graph (#575). A key's goal may only run
# once every value (its upstream spec ids) has reached a terminal status — the
# upstream writes the flat ``results/<qNN>/`` output the dependent hardcodes a
# read of. Absent keys are roots (no upstream): q01, q05, q07, q08, q09.
BATTERY_DEPENDENCIES: dict[str, list[str]] = {
    "battery04_q02": ["battery04_q01"],
    "battery04_q03": ["battery04_q01"],
    "battery04_q04": ["battery04_q01", "battery04_q02", "battery04_q03"],
    "battery04_q06": ["battery04_q05"],
}

# Status-hash values that mean a run is DONE (the release loop may enqueue a
# dependent once its upstream is here). Mirrors ``RunStatusStore.mark``'s
# finished-at set; QUEUED/RUNNING are NOT terminal.
_TERMINAL_STATUSES: frozenset[str] = frozenset(
    s.value
    for s in (
        JobStatus.COMPLETED,
        JobStatus.FAILED,
        JobStatus.TIMEOUT,
        JobStatus.BUDGET_EXHAUSTED,
        JobStatus.CANCELLED,
    )
)


def build_run_id(spec_id: str, date_str: str) -> str:
    """The scheduler's per-night run_id: ``{spec_id}-{YYYYMMDD}``.

    The ``-YYYYMMDD`` suffix is stripped by ``runner._resolve_eval_spec_id`` to
    recover the spec id for eval scoring; it also isolates nightly deliverables
    under ``results/<spec>-<date>/`` so a night's artifacts never overwrite a
    prior night's. The capability curve is unaffected — ``eval_results`` is keyed
    per ``eval_attempt_id`` (fresh per invocation), not per ``run_id``.
    """
    return f"{spec_id}-{date_str}"


def _today_date_str(settings_s: SchedulerSettings) -> str:
    """Today's date (UTC) formatted as the configured suffix (default ``YYYYMMDD``)."""
    return datetime.now(timezone.utc).strftime(settings_s.date_suffix_format)


def _spec_id_from_run(run_id: str) -> str:
    """Strip the trailing ``-<date>`` suffix from a battery ``run_id`` → spec id.

    ``build_run_id`` joins ``f"{spec_id}-{YYYYMMDD}"`` and the date suffix is
    compact (no embedded hyphens), so the spec id is everything before the last
    hyphen. Used to look a job's dependencies up in ``BATTERY_DEPENDENCIES``.
    """
    return run_id.rsplit("-", 1)[0] if "-" in run_id else run_id


def _flat_clear_subdirs(spec: GoalSpec) -> list[str]:
    """The flat ``results/<qNN>/`` dir(s) this goal writes — to self-clear pre-run.

    Each battery goal's ``expected_deliverables`` share a ``results/<qNN>/<file>``
    prefix; the first path component after ``results/`` is the goal's single
    flat write-dir. The worker clears it before a fresh flat-root run (#575) so a
    prior night's DIFFERING file set (e.g. an extra file no longer produced)
    can't linger and be read by a dependent or scored as this run's output.
    Empty for a spec with no ``results/``-prefixed deliverable (cleared nowhere).
    """
    subs: set[str] = set()
    for deliverable in spec.expected_deliverables:
        if not deliverable.startswith("results/"):
            continue
        parts = deliverable.split("/")
        if len(parts) > 2:  # results/<qNN>/<file>
            subs.add(parts[1])
    return sorted(subs)


def build_battery_jobs(
    specs: list[GoalSpec], settings_s: SchedulerSettings, date_str: str
) -> list[RunJob]:
    """Map the battery ``GoalSpec`` list to date-suffixed ``RunJob`` payloads.

    ``spec_limit`` bounds the batch: 0 = every spec (the production nightly
    curve); >0 = only the first N (set 1 for a cheap one-spec plumbing smoke).
    Each spec's ``max_iterations`` comes from the spec itself; ``no_evolution`` /
    ``model`` come from scheduler settings (model empty → the run uses its
    complexity-tiered default, mirroring ``--eval``).
    """
    limit = settings_s.spec_limit
    selected = specs[:limit] if limit > 0 else specs
    model = settings_s.model or None
    return [
        RunJob(
            run_id=build_run_id(spec.spec_id, date_str),
            goal=spec.goal_text,
            max_iterations=spec.max_iterations,
            no_evolution=settings_s.no_evolution,
            model=model,
            # Battery flat-root mode (#575): cross-dependent goals hardcode flat
            # reads (``cat results/q01/...``) of a sibling goal's output, so the
            # whole battery shares the flat results root instead of per-run
            # isolation. Integrity comes from the pre-run self-clear + the
            # topological DAG release (enqueue_battery), not per-run fencing.
            results_per_run_subdir=False,
            clear_flat_subdirs=_flat_clear_subdirs(spec),
        )
        for spec in selected
    ]


class BatteryEnqueuer:
    """Enqueue the full golden battery into ``turing:runs`` on schedule.

    Thin over ``RunsQueue``: builds the date-suffixed jobs and enqueues them. A
    test injects a fake queue and asserts the exact XADD payloads without a
    broker. One per-job failure is logged + skipped (non-fatal): a single bad
    enqueue must not abort the batch, and the stream/lease machinery already
    makes redelivery safe.

    With a ``status_store`` (production), ``enqueue_battery`` does a topological
    DAG-release (#575): roots enqueue immediately, a dependent only once every
    upstream reaches a terminal status. Without one (tests / legacy host runs)
    it falls back to all-at-once enqueue — the original behavior.
    """

    def __init__(
        self,
        queue: RunsQueue,
        settings_s: SchedulerSettings,
        status_store: RunStatusStore | None = None,
    ) -> None:
        self._queue = queue
        self._settings = settings_s
        self._status_store = status_store

    async def _is_terminal(self, run_id: str) -> bool:
        """True iff ``run_id``'s status hash shows a terminal status.

        Best-effort: a missing hash (not yet claimed / unknown / expired) or a
        Redis error reads as NOT terminal, so the release loop keeps waiting for
        a real upstream rather than releasing on a transient blip. The overall
        deadline (``enqueue_battery``) is the backstop if a status never lands.
        """
        if self._status_store is None:
            return False
        try:
            status = await self._status_store.get(run_id)
        except Exception:  # noqa: BLE001 — best-effort poll; never break the loop
            return False
        return status is not None and status.status.value in _TERMINAL_STATUSES

    async def _deps_satisfied(self, spec_id: str, date_str: str) -> bool:
        """True iff every cross-query upstream of ``spec_id`` is terminal."""
        for upstream in BATTERY_DEPENDENCIES.get(spec_id, ()):
            if not await self._is_terminal(build_run_id(upstream, date_str)):
                return False
        return True

    async def enqueue_battery(self, date_str: str | None = None) -> list[str]:
        """Enqueue every battery spec for ``date_str`` (default: today UTC).

        Returns the list of stream entry ids (one per spec). A per-spec enqueue
        failure is logged at WARNING and recorded as ``""`` — the batch
        continues so one transient Redis hiccup on spec N does not drop the rest.

        Release order (#575): with a status store, roots (no upstream) enqueue
        immediately and run in parallel on the worker pool; a dependent is held
        until its upstreams are terminal (polled every ``release_poll_s``), so a
        cross-query goal never races the upstream whose flat output it reads. An
        overall ``release_wait_s`` deadline bounds the loop: if it hits (both
        workers down mid-battery, a status that never lands) the remaining
        dependents are enqueued anyway — a missing upstream makes them honestly
        fail, which is exactly the degraded curve point a broken night records.
        Without a status store this is the original all-at-once enqueue.
        """
        suffix = date_str or _today_date_str(self._settings)
        jobs = build_battery_jobs(BATTERY04_GOALS, self._settings, suffix)
        logger.info(
            f"Battery scheduler firing — {len(jobs)} specs under date suffix "
            f"-{suffix} (mode={'DAG-release' if self._status_store else 'all-at-once'})"
        )
        by_spec = {_spec_id_from_run(job.run_id): job for job in jobs}

        async def _enqueue(job: RunJob) -> str:
            try:
                entry_id = await self._queue.enqueue(job)
                logger.info(f"Enqueued {job.run_id} → {entry_id}")
                return entry_id
            except Exception as exc:  # noqa: BLE001 — non-fatal: keep the batch going
                logger.warning(f"Failed to enqueue {job.run_id}: {exc}")
                return ""

        entry_ids: list[str] = []
        enqueued: set[str] = set()

        # No status store → original all-at-once behavior (tests / legacy host).
        if self._status_store is None:
            for job in jobs:
                entry_ids.append(await _enqueue(job))
                enqueued.add(_spec_id_from_run(job.run_id))
            return entry_ids

        # Phase 1: release every goal whose upstreams are already terminal. On
        # the first pass only roots qualify; later passes release dependents.
        async def _release_ready() -> None:
            for spec_id, job in by_spec.items():
                if spec_id in enqueued:
                    continue
                if await self._deps_satisfied(spec_id, suffix):
                    entry_ids.append(await _enqueue(job))
                    enqueued.add(spec_id)

        await _release_ready()

        # Phase 2: poll until every goal is released, bounded by the deadline.
        deadline = time.monotonic() + self._settings.release_wait_s
        while len(enqueued) < len(jobs):
            if time.monotonic() >= deadline:
                pending = [sid for sid in by_spec if sid not in enqueued]
                logger.warning(
                    f"Battery release deadline ({self._settings.release_wait_s}s) "
                    f"hit with {len(pending)} goal(s) unenqueued {pending}; "
                    "enqueuing anyway (dependents may score low on missing upstreams)."
                )
                for spec_id in pending:
                    entry_ids.append(await _enqueue(by_spec[spec_id]))
                    enqueued.add(spec_id)
                break
            await _release_ready()
            if len(enqueued) >= len(jobs):
                break
            await asyncio.sleep(self._settings.release_poll_s)
        return entry_ids


def make_battery_scheduler(
    enqueuer: BatteryEnqueuer, settings_s: SchedulerSettings
) -> AsyncIOScheduler:
    """Build an APScheduler that fires ``enqueuer.enqueue_battery`` on ``cron``.

    apscheduler is imported LAZILY (inside this function) so importing
    ``src.scheduler.battery`` never requires the dep — the pure logic above
    (``build_run_id`` / ``build_battery_jobs`` / ``BatteryEnqueuer``) is unit
    -tested without it. The job recomputes today's date at fire time (not
    scheduler-start time) so a long-lived daemon enqueues under the right date
    each night.

    Returns the (un-started) ``AsyncIOScheduler``; the caller ``.start()``s it.
    """
    from apscheduler.schedulers.asyncio import AsyncIOScheduler  # noqa: PLC0415
    from apscheduler.triggers.cron import CronTrigger  # noqa: PLC0415

    async def _fire() -> None:
        await enqueuer.enqueue_battery()

    scheduler = AsyncIOScheduler(timezone=settings_s.timezone)
    scheduler.add_job(
        _fire,
        CronTrigger.from_crontab(settings_s.cron, timezone=settings_s.timezone),
        id="turing-battery",
        # A battery takes minutes; never overlap two fires, and coalesce missed
        # fires into one. misfire_grace_time lets a fire deferred by a brief
        # restart still run (a daemon down across 02:00 recovers within the hour).
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3600,
    )
    return scheduler
