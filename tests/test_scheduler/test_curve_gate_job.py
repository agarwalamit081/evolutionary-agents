"""Scheduler wiring for the curve regression→rollback gate (Phase 2 C1f).

Mirrors the battery-scheduler wiring tests (``test_battery.py``): the shipped
default cron parses + yields a future fire time; ``add_curve_gate_job`` registers
exactly one ``turing-curve-gate`` CronTrigger job on the scheduler; and the gate
is OFF by default so ``src/scheduler/__main__.py``'s ``if gate_enabled`` branch
skips registration on a clean host run (the same opt-in convention as the
battery scheduler).
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.config.settings import CapabilityCurveSettings
from src.eval.curve import CapabilityCurve
from src.eval.store import EvalStore
from src.evolution.curve_gate import CurveRegressionGate, add_curve_gate_job
from src.evolution.promote import PromotionGate


def test_default_curve_cron_is_a_valid_crontab() -> None:
    """The shipped default ``0 5 * * *`` must parse + yield a future fire time."""
    from apscheduler.triggers.cron import CronTrigger

    settings = CapabilityCurveSettings(_env_file=None)
    trigger = CronTrigger.from_crontab(settings.curve_cron, timezone=settings.timezone)
    next_fire = trigger.get_next_fire_time(None, datetime.now(timezone.utc))
    assert next_fire is not None


def test_curve_gate_disabled_by_default() -> None:
    """``CAPABILITY_CURVE_GATE_ENABLED`` defaults False — the daemon registers nothing unless opted in."""
    assert CapabilityCurveSettings(_env_file=None).gate_enabled is False


def test_add_curve_gate_job_registers_single_cron_job() -> None:
    """The wiring adds exactly one job (id=turing-curve-gate) on the curve cron."""
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger

    scheduler = AsyncIOScheduler()
    settings = CapabilityCurveSettings(_env_file=None, curve_cron="0 5 * * *")
    gate = CurveRegressionGate(CapabilityCurve(EvalStore()), PromotionGate(), settings=settings)

    add_curve_gate_job(scheduler, gate, settings)

    jobs = scheduler.get_jobs()
    assert len(jobs) == 1
    job = jobs[0]
    assert job.id == "turing-curve-gate"
    assert isinstance(job.trigger, CronTrigger)
    assert job.max_instances == 1
    assert job.coalesce is True
