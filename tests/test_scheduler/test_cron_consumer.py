"""AgentCronEnqueuer + sync job (Phase 5 I1): fire→RunJob + reconcile.

The fire path is verified with a capturing fake queue (never a broker); the
reconcile + sync-job paths run against an UN-STARTED ``AsyncIOScheduler`` (its
in-memory jobstore accepts ``add_job``/``get_jobs``/``remove_job`` before
``.start()``) with ``load_enabled_tasks`` monkeypatched so no DB is touched. This
locks the DoD — a task row ⇒ an armed cron job ⇒ an enqueued ``RunJob`` on fire
— deterministically, with no wall-clock dependency.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from src.db.models import ScheduledTask
from src.scheduler.cron_consumer import (
    CRON_JOB_PREFIX,
    SYNC_JOB_ID,
    AgentCronEnqueuer,
    make_agent_cron_sync_job,
)
from src.worker.schema import RunJob


def _task(
    *,
    name: str = "weekday-report",
    cron: str = "0 9 * * 1-5",
    goal: str = "refresh the daily report",
    model: str | None = None,
    timezone_: str = "UTC",
) -> ScheduledTask:
    return ScheduledTask(
        name=name, cron=cron, goal=goal, model=model, timezone=timezone_, enabled=True
    )


class _FakeQueue:
    def __init__(self, *, raise_on_enqueue: bool = False) -> None:
        self.jobs: list[RunJob] = []
        self._raise = raise_on_enqueue

    async def enqueue(self, job: RunJob) -> str:
        if self._raise:
            raise RuntimeError("redis down")
        self.jobs.append(job)
        return f"entry-{len(self.jobs)}"


def _settings(sync_interval_s: int = 60) -> Any:
    return SimpleNamespace(sync_interval_s=sync_interval_s, timezone="UTC")


class TestFire:
    @pytest.mark.asyncio
    async def test_build_run_id_format(self) -> None:
        enq = AgentCronEnqueuer(_FakeQueue(), _settings())  # type: ignore[arg-type]
        now = datetime(2026, 6, 27, 9, 0, 5, tzinfo=timezone.utc)
        assert enq.build_run_id(_task(), now) == "cron-weekday-report-20260627090005"

    @pytest.mark.asyncio
    async def test_fire_enqueues_correct_runjob(self) -> None:
        queue = _FakeQueue()
        enq = AgentCronEnqueuer(queue, _settings())  # type: ignore[arg-type]
        now = datetime(2026, 6, 27, 9, 30, 0, tzinfo=timezone.utc)
        task = _task(goal="refresh report", model="glm-4.7")

        entry = await enq.fire(task, now=now)

        assert entry == "entry-1"
        assert len(queue.jobs) == 1
        job = queue.jobs[0]
        assert job.run_id == "cron-weekday-report-20260627093000"
        assert job.goal == "refresh report"
        assert job.model == "glm-4.7"
        # A cron-fired run uses the default iteration cap + allows evolution.
        assert job.max_iterations is None
        assert job.no_evolution is False

    @pytest.mark.asyncio
    async def test_fire_default_model_is_none(self) -> None:
        queue = _FakeQueue()
        enq = AgentCronEnqueuer(queue, _settings())  # type: ignore[arg-type]
        await enq.fire(_task(model=None), now=datetime(2026, 1, 1, tzinfo=timezone.utc))
        assert queue.jobs[0].model is None

    @pytest.mark.asyncio
    async def test_fire_enqueue_failure_is_non_fatal(self) -> None:
        queue = _FakeQueue(raise_on_enqueue=True)
        enq = AgentCronEnqueuer(queue, _settings())  # type: ignore[arg-type]
        # Must not raise — a dropped fire returns "" (sibling fires continue).
        entry = await enq.fire(_task(), now=datetime(2026, 1, 1, tzinfo=timezone.utc))
        assert entry == ""


@pytest.fixture()
def scheduler() -> AsyncIOScheduler:
    return AsyncIOScheduler()


class TestReconcile:
    @pytest.mark.asyncio
    async def test_registers_cron_job_per_task(
        self, scheduler: AsyncIOScheduler
    ) -> None:
        enq = AgentCronEnqueuer(_FakeQueue(), _settings())  # type: ignore[arg-type]
        tasks = [_task(name="a", cron="0 9 * * *"), _task(name="b", cron="*/30 * * * *")]
        with patch.object(enq, "load_enabled_tasks", return_value=tasks):
            registered = await enq.reconcile(scheduler)

        assert registered == 2
        ids = {j.id for j in scheduler.get_jobs()}
        assert ids == {f"{CRON_JOB_PREFIX}a", f"{CRON_JOB_PREFIX}b"}
        # Each registered job carries a real CronTrigger.
        for job in scheduler.get_jobs():
            assert isinstance(job.trigger, CronTrigger)

    @pytest.mark.asyncio
    async def test_removes_stale_jobs(self, scheduler: AsyncIOScheduler) -> None:
        enq = AgentCronEnqueuer(_FakeQueue(), _settings())  # type: ignore[arg-type]
        # Pre-arm a task, then reconcile it away (no enabled rows).
        async def _fake_load() -> list[ScheduledTask]:
            return [_task(name="keep", cron="0 9 * * *")]

        with patch.object(enq, "load_enabled_tasks", _fake_load):
            await enq.reconcile(scheduler)
        assert f"{CRON_JOB_PREFIX}keep" in {j.id for j in scheduler.get_jobs()}

        # Second reconcile: the task is gone → its job is pruned (only OUR jobs).
        with patch.object(enq, "load_enabled_tasks", return_value=[]):
            await enq.reconcile(scheduler)
        assert scheduler.get_jobs() == []

    @pytest.mark.asyncio
    async def test_skips_unparseable_cron(self, scheduler: AsyncIOScheduler) -> None:
        enq = AgentCronEnqueuer(_FakeQueue(), _settings())  # type: ignore[arg-type]
        tasks = [_task(name="good", cron="0 9 * * *"), _task(name="bad", cron="not cron")]
        with patch.object(enq, "load_enabled_tasks", return_value=tasks):
            registered = await enq.reconcile(scheduler)

        # Only the parseable task armed; the bad row skipped (non-fatal).
        assert registered == 1
        ids = {j.id for j in scheduler.get_jobs()}
        assert ids == {f"{CRON_JOB_PREFIX}good"}

    @pytest.mark.asyncio
    async def test_reconcile_is_idempotent(
        self, scheduler: AsyncIOScheduler
    ) -> None:
        enq = AgentCronEnqueuer(_FakeQueue(), _settings())  # type: ignore[arg-type]
        tasks = [_task(name="a", cron="0 9 * * *")]
        with patch.object(enq, "load_enabled_tasks", return_value=tasks):
            await enq.reconcile(scheduler)
            await enq.reconcile(scheduler)  # replace_existing=True → no duplicate
        cron_jobs = [j for j in scheduler.get_jobs() if j.id.startswith(CRON_JOB_PREFIX)]
        assert len(cron_jobs) == 1


class TestSyncJob:
    def test_registers_interval_job(self, scheduler: AsyncIOScheduler) -> None:
        enq = AgentCronEnqueuer(_FakeQueue(), _settings(sync_interval_s=30))  # type: ignore[arg-type]
        make_agent_cron_sync_job(scheduler, enq, _settings(sync_interval_s=30))  # type: ignore[arg-type]
        jobs = {j.id: j for j in scheduler.get_jobs()}
        assert SYNC_JOB_ID in jobs
        assert isinstance(jobs[SYNC_JOB_ID].trigger, IntervalTrigger)
