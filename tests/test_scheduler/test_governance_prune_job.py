"""Scheduler wiring for the periodic governance prune (battery-04 q09 fix C).

Mirrors ``test_curve_gate_job.py``: the shipped default cron parses + yields a future
fire time; ``add_governance_prune_job`` registers exactly one ``turing-governance-prune``
CronTrigger job on the scheduler; and the prune is OFF by default so
``src/scheduler/__main__.py``'s ``if prune_settings.enabled`` branch skips registration
on a clean host run (the same opt-in convention as the battery + curve-gate jobs).
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.config.settings import GovernancePruneSettings
from src.governance.prune import GovernancePruner, add_governance_prune_job


def test_default_prune_cron_is_a_valid_crontab() -> None:
    """The shipped default ``0 4 * * *`` must parse + yield a future fire time."""
    from apscheduler.triggers.cron import CronTrigger

    settings = GovernancePruneSettings(_env_file=None)
    trigger = CronTrigger.from_crontab(settings.cron, timezone=settings.timezone)
    next_fire = trigger.get_next_fire_time(None, datetime.now(timezone.utc))
    assert next_fire is not None


def test_governance_prune_disabled_by_default() -> None:
    """``GOVERNANCE_PRUNE_ENABLED`` defaults False — the daemon registers nothing unless opted in."""
    assert GovernancePruneSettings(_env_file=None).enabled is False


def test_add_governance_prune_job_registers_single_cron_job() -> None:
    """The wiring adds exactly one job (id=turing-governance-prune) on the prune cron."""
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger

    scheduler = AsyncIOScheduler()
    settings = GovernancePruneSettings(_env_file=None, cron="0 4 * * *")
    pruner = GovernancePruner()

    add_governance_prune_job(scheduler, pruner, settings)

    jobs = scheduler.get_jobs()
    assert len(jobs) == 1
    job = jobs[0]
    assert job.id == "turing-governance-prune"
    assert isinstance(job.trigger, CronTrigger)
    assert job.max_instances == 1
    assert job.coalesce is True
