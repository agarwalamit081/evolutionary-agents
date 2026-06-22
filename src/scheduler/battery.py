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
spec, while per-run results isolate under ``results/<spec>-<date>/``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from loguru import logger

from src.config.settings import SchedulerSettings
from src.eval.golden import BATTERY04_GOALS
from src.eval.models import GoalSpec
from src.worker.queue import RunsQueue
from src.worker.schema import RunJob

if TYPE_CHECKING:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler


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


def build_battery_jobs(
    specs: list[GoalSpec], settings_s: SchedulerSettings, date_str: str
) -> list[RunJob]:
    """Map every battery ``GoalSpec`` to a date-suffixed ``RunJob``.

    ``max_iterations`` comes from the spec (each query's own cap);
    ``no_evolution`` / ``model`` come from scheduler settings (model empty → the
    run uses its complexity-tiered default, mirroring ``--eval``).
    """
    model = settings_s.model or None
    return [
        RunJob(
            run_id=build_run_id(spec.spec_id, date_str),
            goal=spec.goal_text,
            max_iterations=spec.max_iterations,
            no_evolution=settings_s.no_evolution,
            model=model,
        )
        for spec in specs
    ]


class BatteryEnqueuer:
    """Enqueue the full golden battery into ``turing:runs`` on schedule.

    Thin over ``RunsQueue``: builds the date-suffixed jobs and enqueues each.
    Holds no Redis client of its own — the queue does — so a test injects a fake
    queue and asserts the exact XADD payloads without a broker. One per-job
    failure is logged + skipped (non-fatal): a single bad enqueue must not abort
    the batch, and the stream/lease machinery already makes redelivery safe.
    """

    def __init__(self, queue: RunsQueue, settings_s: SchedulerSettings) -> None:
        self._queue = queue
        self._settings = settings_s

    async def enqueue_battery(self, date_str: str | None = None) -> list[str]:
        """Enqueue every battery spec for ``date_str`` (default: today UTC).

        Returns the list of stream entry ids (one per spec), in spec order. A
        per-spec enqueue failure is logged at WARNING and recorded as ``""`` in
        the result list — the batch continues so one transient Redis hiccup on
        spec N does not drop specs N+1..end.
        """
        suffix = date_str or _today_date_str(self._settings)
        jobs = build_battery_jobs(BATTERY04_GOALS, self._settings, suffix)
        entry_ids: list[str] = []
        logger.info(
            f"Battery scheduler firing — {len(jobs)} specs under date suffix -{suffix}"
        )
        for job in jobs:
            try:
                entry_id = await self._queue.enqueue(job)
                entry_ids.append(entry_id)
                logger.info(f"Enqueued {job.run_id} → {entry_id}")
            except Exception as exc:  # noqa: BLE001 — non-fatal: keep the batch going
                logger.warning(f"Failed to enqueue {job.run_id}: {exc}")
                entry_ids.append("")
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
