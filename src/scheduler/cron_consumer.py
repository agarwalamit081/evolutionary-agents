"""Agent-cron consumer — fires durable ``scheduled_tasks`` rows into the run queue (I1).

Companion to the ``create_scheduled_task`` builtin (Phase 5 I1). The agent authors
future work into the ``scheduled_tasks`` table; THIS module turns those rows into
real runs. It reconciles the table against an APScheduler instance (add/refresh
a per-task ``CronTrigger`` job, remove disabled/deleted ones) and, on each task's
cron tick, enqueues a ``RunJob`` through the SAME ``RunsQueue.enqueue`` seam the
API + battery enqueuer use — so a scheduled run flows through the real deployed
worker stack (lease-lock, checkpoint, eval-resolution all apply unchanged).

Pure + importable WITHOUT apscheduler: ``AgentCronEnqueuer`` (load rows / build a
fire job / enqueue) and the reconcile logic are unit-testable with a fake queue
and an un-started ``AsyncIOScheduler``; APScheduler is imported lazily inside the
methods that need it. ``make_agent_cron_sync_job`` registers an IntervalTrigger
job that re-runs reconcile every ``sync_interval_s`` so a task the agent creates
mid-flight is picked up without a scheduler restart. Opt-in (``AGENT_CRON_ENABLED``).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from loguru import logger

from src.config.settings import AgentCronSettings
from src.worker.queue import RunsQueue
from src.worker.schema import RunJob

if TYPE_CHECKING:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    from src.db.models import ScheduledTask


#: Prefix for per-task cron jobs on the shared scheduler. Used by reconcile to
#: distinguish agent-cron jobs from the battery / curve-gate / prune / optimizer
#: jobs that share the same scheduler instance.
CRON_JOB_PREFIX = "turing-cron-"
#: The reconcile job id (IntervalTrigger). Stable so ``replace_existing`` is a
#: no-op when the daemon re-registers it across a reconnect.
SYNC_JOB_ID = "turing-agent-cron-sync"


class AgentCronEnqueuer:
    """Turn ``scheduled_tasks`` rows into cron-fired ``RunJob`` enqueues.

    Thin over ``RunsQueue``: holds no Redis client of its own (the queue does),
    so a test injects a fake queue and asserts the exact enqueued ``RunJob``
    without a broker. A per-task enqueue failure is logged + returns ``""``
    (non-fatal) so one transient Redis hiccup never aborts a sibling fire; the
    lease/redelivery machinery makes a dropped fire safe to miss once.
    """

    def __init__(self, queue: RunsQueue, settings: AgentCronSettings) -> None:
        self._queue = queue
        self._settings = settings

    def job_id_for(self, name: str) -> str:
        """The stable APScheduler job id for a task (keyed by its unique name)."""
        return f"{CRON_JOB_PREFIX}{name}"

    async def load_enabled_tasks(self) -> list[ScheduledTask]:
        """Load every enabled ``scheduled_tasks`` row, name-sorted (stable reconcile)."""
        from sqlalchemy import select

        from src.db.models import ScheduledTask
        from src.db.session import get_session

        async with get_session() as session:
            result = await session.execute(
                select(ScheduledTask)
                .where(ScheduledTask.enabled.is_(True))
                .order_by(ScheduledTask.name)
            )
            return list(result.scalars().all())

    def build_run_id(self, task: ScheduledTask, now: datetime) -> str:
        """Per-fire run id: ``cron-<name>-<UTC compact stamp>``.

        ``max_instances=1`` per task guarantees no two fires of the SAME task run
        concurrently, so the second-granularity stamp is collision-free in
        practice (cron ticks are >=1 min apart in production). The ``thread_id``
        (``api-{run_id}``) and per-run results subdir isolate the fired run from
        every other — same join-key convention as the API/battery paths.
        """
        return f"cron-{task.name}-{now:%Y%m%d%H%M%S}"

    async def fire(
        self, task: ScheduledTask, now: datetime | None = None
    ) -> str:
        """Enqueue one ``RunJob`` for ``task``'s cron tick. Returns the entry id.

        ``now`` is injectable so a test fixes the run-id stamp deterministically;
        the production fire reads the wall clock. A model pin flows straight into
        the job (empty/None → the run uses its complexity-tiered default routing).
        """
        fire_time = now or datetime.now(timezone.utc)
        job = RunJob(
            run_id=self.build_run_id(task, fire_time),
            goal=task.goal,
            model=task.model or None,
        )
        try:
            entry_id = await self._queue.enqueue(job)
            logger.info(
                f"Agent cron fired task={task.name!r} → run_id={job.run_id} entry={entry_id}"
            )
            return entry_id
        except Exception as exc:  # noqa: BLE001 — non-fatal: one dropped fire must not abort siblings
            logger.warning(f"Agent cron enqueue failed for task={task.name!r}: {exc}")
            return ""

    def make_fire_callback(self, task: ScheduledTask) -> Callable[[], Awaitable[None]]:
        """Build the no-arg async closure APScheduler invokes on each cron tick."""
        # Bind the current task row in the closure; reconcile replaces the job
        # (replace_existing=True) when the agent revises the row, so a stale
        # closure never fires a superseded schedule.
        async def _fire() -> None:
            await self.fire(task)

        return _fire

    async def reconcile(self, scheduler: AsyncIOScheduler) -> int:
        """Mirror enabled tasks onto ``scheduler``; prune disabled/deleted ones.

        Idempotent: safe to call every ``sync_interval_s``. Adds a fresh
        ``CronTrigger`` job per enabled task (replace_existing), removes any
        agent-cron job whose task was disabled/deleted, and skips a row whose
        cron/timezone is unparseable (logged, never raised — one bad row can't
        abort the reconcile). Returns the count of tasks registered this pass.
        """
        from apscheduler.triggers.cron import CronTrigger  # noqa: PLC0415

        tasks = await self.load_enabled_tasks()
        desired = {self.job_id_for(t.name): t for t in tasks}

        # Drop jobs whose task disappeared (disabled by re-call, or deleted). Only
        # touch OUR jobs (the prefix) — the battery/curve-gate/prune/optimizer
        # jobs share this scheduler and must be left alone.
        for job in list(scheduler.get_jobs()):
            if isinstance(job.id, str) and job.id.startswith(CRON_JOB_PREFIX) and job.id not in desired:
                scheduler.remove_job(job.id)
                logger.info(f"Agent cron removed stale job {job.id!r}")

        from apscheduler.jobstores.base import JobLookupError  # noqa: PLC0415

        registered = 0
        for jid, task in desired.items():
            tz = task.timezone or "UTC"
            try:
                trigger = CronTrigger.from_crontab(task.cron, timezone=tz)
            except Exception as exc:  # noqa: BLE001 — skip one bad row, keep reconciling
                logger.warning(
                    f"Agent cron skipping task={task.name!r}: unparseable cron "
                    f"{task.cron!r} tz={tz!r}: {exc}"
                )
                continue
            # ``replace_existing`` only dedups once the scheduler is STARTED; before
            # ``start()`` APScheduler appends to ``_pending_jobs`` verbatim, so an
            # un-started reconcile would stack duplicate ids (and a started one still
            # benefits from tearing the old job down so the fresh row's closure +
            # trigger replace a possibly-revised schedule). JobLookupError-safe ⇒ a
            # first registration is a clean add.
            try:
                scheduler.remove_job(jid)
            except JobLookupError:
                pass
            scheduler.add_job(
                self.make_fire_callback(task),
                trigger,
                id=jid,
                replace_existing=True,
                # Never overlap two fires of the same task; coalesce missed ticks
                # into one; a deferred fire (brief restart) still runs within the hour.
                max_instances=1,
                coalesce=True,
                misfire_grace_time=3600,
            )
            registered += 1

        logger.info(
            f"Agent-cron reconcile: {registered}/{len(desired)} task(s) registered"
        )
        return registered


def make_agent_cron_sync_job(
    scheduler: AsyncIOScheduler,
    enqueuer: AgentCronEnqueuer,
    settings: AgentCronSettings,
) -> None:
    """Register the periodic reconcile job on ``scheduler`` (IntervalTrigger).

    The agent can create/revise/disable tasks mid-flight (via the builtin, while
    the scheduler daemon runs); the sync job re-runs ``reconcile`` every
    ``sync_interval_s`` so a new task is armed within one tick of its row landing.
    The daemon ALSO reconciles once immediately at startup (see ``__main__.py``)
    so pre-existing tasks fire without waiting for the first tick.
    """
    from apscheduler.triggers.interval import IntervalTrigger  # noqa: PLC0415

    async def _sync() -> None:
        await enqueuer.reconcile(scheduler)

    scheduler.add_job(
        _sync,
        IntervalTrigger(seconds=settings.sync_interval_s, timezone=settings.timezone),
        id=SYNC_JOB_ID,
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3600,
    )
