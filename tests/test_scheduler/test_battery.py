"""Tests for the nightly capability-curve battery scheduler (#197 Phase 5).

Covers the pure enqueue logic (no broker, no LLM, no apscheduler needed for the
core): the date-suffixed ``run_id`` builder, the spec→``RunJob`` mapping, the
``BatteryEnqueuer`` batch (success + non-fatal per-job failure), the
``_resolve_eval_spec_id`` date-suffix strip that lets scheduled runs be scored,
and that the configured crontab is parseable + the APScheduler wiring registers
exactly one cron job. The worker→execute_run run_id seam is locked separately
(``tests/test_worker/test_executors.py``).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.config.settings import SchedulerSettings
from src.eval.golden import BATTERY04_GOALS
from src.runner import _resolve_eval_spec_id, _strip_date_suffix
from src.scheduler.battery import (
    BatteryEnqueuer,
    build_battery_jobs,
    build_run_id,
    make_battery_scheduler,
)
from src.worker.schema import RunJob


# ── Pure builders ─────────────────────────────────────────────────────


def test_build_run_id_appends_compact_date_suffix() -> None:
    assert build_run_id("battery04_q01", "20260622") == "battery04_q01-20260622"


class TestStripDateSuffix:
    """The resolver relies on this exact contract: ``-YYYYMMDD`` (8 digits)."""

    def test_recovers_spec_id(self) -> None:
        assert _strip_date_suffix("battery04_q01-20260622") == "battery04_q01"

    def test_none_when_no_suffix(self) -> None:
        assert _strip_date_suffix("battery04_q01") is None

    def test_none_when_not_eight_digits(self) -> None:
        assert _strip_date_suffix("deploy-123") is None  # 3 digits

    def test_none_when_embedded_hyphens(self) -> None:
        # %Y-%m-%d would break the strip — the compact %Y%m%d format is required.
        assert _strip_date_suffix("battery04_q01-2026-06-22") is None


class TestResolveEvalSpecId:
    """The scheduler's date-suffixed run_id must still resolve to its spec."""

    def test_date_suffixed_long_form(self) -> None:
        assert _resolve_eval_spec_id("battery04_q01-20260622") == "battery04_q01"

    def test_existing_long_form_unchanged(self) -> None:
        assert _resolve_eval_spec_id("battery04_q01") == "battery04_q01"

    def test_existing_short_form_unchanged(self) -> None:
        assert _resolve_eval_spec_id("q01") == "battery04_q01"

    def test_ordinary_run_id_unaffected(self) -> None:
        assert _resolve_eval_spec_id("deploy-run-42") is None

    def test_none_run_id(self) -> None:
        assert _resolve_eval_spec_id(None) is None


# ── Spec → RunJob mapping ─────────────────────────────────────────────


class TestBuildBatteryJobs:
    def test_maps_every_spec_to_a_date_suffixed_runjob(self) -> None:
        settings = SchedulerSettings(_env_file=None, model="glm-4.7")
        specs = BATTERY04_GOALS[:2]
        jobs = build_battery_jobs(specs, settings, "20260622")

        assert len(jobs) == 2
        assert all(isinstance(j, RunJob) for j in jobs)
        for spec, job in zip(specs, jobs, strict=True):
            assert job.run_id == f"{spec.spec_id}-20260622"
            assert job.goal == spec.goal_text
            assert job.max_iterations == spec.max_iterations
            assert job.model == "glm-4.7"
            assert job.no_evolution is settings.no_evolution

    def test_empty_model_becomes_none_so_run_uses_tiered_default(self) -> None:
        settings = SchedulerSettings(_env_file=None)  # model="" default
        jobs = build_battery_jobs(BATTERY04_GOALS[:1], settings, "20260622")
        assert jobs[0].model is None

    def test_no_evolution_threads_through(self) -> None:
        settings = SchedulerSettings(_env_file=None, no_evolution=True)
        jobs = build_battery_jobs(BATTERY04_GOALS[:1], settings, "20260622")
        assert jobs[0].no_evolution is True


# ── BatteryEnqueuer (fake queue — no broker) ──────────────────────────


class _FakeQueue:
    """Records every enqueue call + returns sequential entry ids."""

    def __init__(self) -> None:
        self.enqueued: list[RunJob] = []

    async def enqueue(self, job: RunJob) -> str:
        self.enqueued.append(job)
        return f"id-{len(self.enqueued)}"


class _FlakyQueue:
    """Fails exactly the 2nd enqueue, then succeeds — exercises non-fatal batch."""

    def __init__(self) -> None:
        self.calls = 0

    async def enqueue(self, _job: RunJob) -> str:
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("transient redis hiccup")
        return f"ok-{self.calls}"


class TestBatteryEnqueuer:
    @pytest.mark.asyncio
    async def test_enqueues_every_spec_with_date_suffix(self) -> None:
        fake = _FakeQueue()
        settings = SchedulerSettings(_env_file=None)
        enqueuer = BatteryEnqueuer(fake, settings)  # type: ignore[arg-type]

        ids = await enqueuer.enqueue_battery("20260622")

        assert len(ids) == len(BATTERY04_GOALS)
        assert len(fake.enqueued) == len(BATTERY04_GOALS)
        # each enqueued run_id is date-suffixed AND resolves back to a real spec
        for job in fake.enqueued:
            assert job.run_id.endswith("-20260622")
            assert _resolve_eval_spec_id(job.run_id) is not None
        # one entry id per spec, in order
        assert ids == [f"id-{i}" for i in range(1, len(BATTERY04_GOALS) + 1)]

    @pytest.mark.asyncio
    async def test_batch_continues_past_a_failed_enqueue(self) -> None:
        fake = _FlakyQueue()
        settings = SchedulerSettings(_env_file=None)
        enqueuer = BatteryEnqueuer(fake, settings)  # type: ignore[arg-type]

        ids = await enqueuer.enqueue_battery("20260622")

        # every spec was ATTEMPTED (the 2nd failed but the batch did not abort)
        assert fake.calls == len(BATTERY04_GOALS)
        assert ids[1] == ""  # the failed one recorded as empty
        assert all(eid for eid in ids[:1] + ids[2:])  # the rest succeeded


# ── APScheduler wiring (apscheduler IS installed: requirements.txt) ───


def test_default_cron_is_a_valid_crontab() -> None:
    """The shipped default ``0 2 * * *`` must parse + yield a future fire time."""
    from apscheduler.triggers.cron import CronTrigger

    settings = SchedulerSettings(_env_file=None)
    trigger = CronTrigger.from_crontab(settings.cron, timezone=settings.timezone)
    next_fire = trigger.get_next_fire_time(None, datetime.now(timezone.utc))
    assert next_fire is not None


def test_make_battery_scheduler_registers_single_cron_job() -> None:
    """The daemon wiring adds exactly one job (id=turing-battery) on the cron."""
    from apscheduler.triggers.cron import CronTrigger

    fake = _FakeQueue()
    settings = SchedulerSettings(_env_file=None, cron="0 2 * * *")
    enqueuer = BatteryEnqueuer(fake, settings)  # type: ignore[arg-type]

    scheduler = make_battery_scheduler(enqueuer, settings)
    jobs = scheduler.get_jobs()

    assert len(jobs) == 1
    job = jobs[0]
    assert job.id == "turing-battery"
    assert isinstance(job.trigger, CronTrigger)
