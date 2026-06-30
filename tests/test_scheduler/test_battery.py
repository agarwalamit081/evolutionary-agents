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
    BATTERY_DEPENDENCIES,
    BatteryEnqueuer,
    _flat_clear_subdirs,
    _spec_id_from_run,
    build_battery_jobs,
    build_run_id,
    make_battery_scheduler,
)
from src.worker.schema import JobStatus, RunJob, RunStatus


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

    def test_spec_limit_zero_enqueues_every_spec(self) -> None:
        """spec_limit=0 (the default) is the production nightly curve — every spec."""
        settings = SchedulerSettings(_env_file=None)  # spec_limit defaults 0
        jobs = build_battery_jobs(BATTERY04_GOALS, settings, "20260622")
        assert len(jobs) == len(BATTERY04_GOALS)

    def test_spec_limit_positive_caps_to_first_n(self) -> None:
        """spec_limit>0 enqueues only the first N specs — the cheap smoke cap."""
        settings = SchedulerSettings(_env_file=None, spec_limit=1)
        jobs = build_battery_jobs(BATTERY04_GOALS, settings, "20260622")
        assert len(jobs) == 1
        assert jobs[0].run_id == f"{BATTERY04_GOALS[0].spec_id}-20260622"


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


# ── Cross-query DAG release (#575) ───────────────────────────────────


class TestFlatClearSubdir:
    """_flat_clear_subdir: derive a goal's flat results/<qNN>/ write-dir."""

    def test_derives_single_qnn_from_real_spec(self) -> None:
        q01 = next(s for s in BATTERY04_GOALS if s.spec_id == "battery04_q01")
        # q01 deliverables are results/q01/normalized.csv + results/q01/summary.json
        assert _flat_clear_subdirs(q01) == ["q01"]

    def test_derives_for_every_battery_goal(self) -> None:
        """Every battery goal writes under exactly one results/<qNN>/ dir."""
        for spec in BATTERY04_GOALS:
            subs = _flat_clear_subdirs(spec)
            assert len(subs) == 1, f"{spec.spec_id} should clear one dir, got {subs}"
            assert subs[0].startswith("q")

    def test_empty_when_no_results_deliverable(self) -> None:
        from src.eval.models import GoalSpec

        spec = GoalSpec(
            spec_id="x", name="x", goal_text="g", expected_deliverables=["workspace/f.txt"]
        )
        assert _flat_clear_subdirs(spec) == []


class TestSpecIdFromRun:
    def test_strips_compact_date_suffix(self) -> None:
        assert _spec_id_from_run("battery04_q02-20260622") == "battery04_q02"

    def test_no_suffix_returns_as_is(self) -> None:
        assert _spec_id_from_run("battery04_q02") == "battery04_q02"


class TestBatteryJobFlatRoot:
    """build_battery_jobs sets the flat-root + self-clear fields (#575)."""

    def test_jobs_use_flat_root_and_self_clear(self) -> None:
        settings = SchedulerSettings(_env_file=None)
        jobs = build_battery_jobs(BATTERY04_GOALS, settings, "20260622")
        assert len(jobs) == len(BATTERY04_GOALS)
        for spec, job in zip(BATTERY04_GOALS, jobs, strict=True):
            # Battery shares the flat results root so cross-query reads resolve.
            assert job.results_per_run_subdir is False
            # Each goal self-clears its own flat write-dir pre-run.
            assert job.clear_flat_subdirs == _flat_clear_subdirs(spec)


class _ProgressiveStatusStore:
    """Fake RunStatusStore: a run flips to COMPLETED after ``flip_after`` polls.

    Models a real worker finishing an upstream some polls after it was enqueued,
    so a dependent that polls it early sees NOT-terminal and waits — the exact
    race the DAG release exists to prevent.
    """

    def __init__(self, flip_after: dict[str, int]) -> None:
        self._flip_after = dict(flip_after)
        self.polls: dict[str, int] = {}

    async def get(self, run_id: str) -> RunStatus | None:
        n = self.polls.get(run_id, 0) + 1
        self.polls[run_id] = n
        if run_id in self._flip_after and n >= self._flip_after[run_id]:
            return RunStatus(run_id=run_id, thread_id=f"api-{run_id}", status=JobStatus.COMPLETED)
        return None  # queued / running / unknown → not terminal


class _NeverTerminalStore:
    """Fake store: nothing ever reaches terminal (drives the deadline fallback)."""

    async def get(self, _run_id: str) -> RunStatus | None:
        return None


class TestBatteryDagRelease:
    """enqueue_battery: topological release + deadline fallback (#575)."""

    @pytest.mark.asyncio
    async def test_dependents_release_only_after_upstream_terminal(self) -> None:
        fake = _FakeQueue()
        settings = SchedulerSettings(
            _env_file=None, release_poll_s=0.0, release_wait_s=5.0
        )
        # Upstreams that something waits on flip terminal after 2 polls.
        store = _ProgressiveStatusStore(
            {
                "battery04_q01-20260622": 2,
                "battery04_q02-20260622": 2,
                "battery04_q03-20260622": 2,
                "battery04_q05-20260622": 2,
            }
        )
        enqueuer = BatteryEnqueuer(fake, settings, status_store=store)  # type: ignore[arg-type]

        ids = await enqueuer.enqueue_battery("20260622")

        # Every goal eventually enqueued, one entry id each.
        assert len(fake.enqueued) == len(BATTERY04_GOALS)
        assert all(eid for eid in ids)
        run_ids = [j.run_id for j in fake.enqueued]

        def idx(spec_id: str) -> int:
            return run_ids.index(f"{spec_id}-20260622")

        # A dependent is enqueued AFTER every upstream it reads.
        for dependent, upstreams in BATTERY_DEPENDENCIES.items():
            for up in upstreams:
                assert idx(up) < idx(dependent), (
                    f"{dependent} (idx {idx(dependent)}) must follow {up} "
                    f"(idx {idx(up)}); order was {run_ids}"
                )
        # q01 was polled repeatedly (the release WAITED for it), not once.
        assert store.polls["battery04_q01-20260622"] >= 2

    @pytest.mark.asyncio
    async def test_deadline_enqueues_remaining_when_upstream_never_lands(self) -> None:
        """If an upstream status never lands (workers down), the deadline still
        enqueues the dependents — they honestly fail on missing data, the
        correct degraded curve point rather than a hung nightly fire."""
        fake = _FakeQueue()
        settings = SchedulerSettings(
            _env_file=None, release_poll_s=0.0, release_wait_s=0.0
        )
        enqueuer = BatteryEnqueuer(
            fake, settings, status_store=_NeverTerminalStore()  # type: ignore[arg-type]
        )

        ids = await enqueuer.enqueue_battery("20260622")

        # Roots enqueue in phase 1; dependents enqueue via the deadline fallback.
        assert len(fake.enqueued) == len(BATTERY04_GOALS)
        assert all(eid for eid in ids)
        run_ids = {j.run_id for j in fake.enqueued}
        # A cross-query dependent that would otherwise wait forever is present.
        assert "battery04_q04-20260622" in run_ids

    @pytest.mark.asyncio
    async def test_no_status_store_falls_back_to_all_at_once(self) -> None:
        """Legacy/test path (no store): original all-at-once enqueue, in order."""
        fake = _FakeQueue()
        settings = SchedulerSettings(_env_file=None)
        enqueuer = BatteryEnqueuer(fake, settings)  # type: ignore[arg-type] — no store

        ids = await enqueuer.enqueue_battery("20260622")

        assert len(fake.enqueued) == len(BATTERY04_GOALS)
        # All-at-once preserves spec order (the pre-#575 behavior).
        assert [j.run_id for j in fake.enqueued] == [
            f"{s.spec_id}-20260622" for s in BATTERY04_GOALS
        ]
        assert ids == [f"id-{i}" for i in range(1, len(BATTERY04_GOALS) + 1)]
