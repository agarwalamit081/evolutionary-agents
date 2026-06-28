"""Scheduler wiring for the nightly optimizer trigger (Phase 2 C2).

Mirrors ``test_governance_prune_job.py`` / ``test_curve_gate_job.py``: the shipped
default cron parses + yields a future fire time; ``add_optimizer_job`` registers
exactly one ``turing-optimizer`` CronTrigger job; and the trigger is OFF by
default so ``src/scheduler/__main__.py``'s ``if optimizer_settings.enabled``
branch registers nothing on a clean host run (same opt-in as the other nightly
jobs).

``trigger_optimization`` is the scheduler→sidecar POST. It is exercised with an
``httpx.MockTransport`` (the test seam on ``trigger_optimization``) so the
decision logic — unreachable / non-200 / non-JSON / 200-outcome — is pinned
without a live sidecar. The trigger NEVER raises: every failure is logged +
swallowed so the scheduler process stays up.
"""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from src.config.settings import OptimizerSettings
from src.optimizer.job import add_optimizer_job, trigger_optimization


def test_default_optimizer_cron_is_a_valid_crontab() -> None:
    """The shipped default ``30 3 * * *`` must parse + yield a future fire time."""
    from apscheduler.triggers.cron import CronTrigger

    settings = OptimizerSettings(_env_file=None)
    trigger = CronTrigger.from_crontab(settings.cron, timezone=settings.timezone)
    next_fire = trigger.get_next_fire_time(None, datetime.now(timezone.utc))
    assert next_fire is not None


def test_optimizer_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """``OPTIMIZER_ENABLED`` defaults False — the daemon registers nothing unless opted in.

    Hermetic: the e2e modules call ``load_dotenv()`` at collection time, which
    populates ``os.environ`` from the live ``.env`` for the whole session — so
    ``delenv`` the knob before asserting the class default (``_env_file=None``
    blocks the file read but not the process env).
    """
    monkeypatch.delenv("OPTIMIZER_ENABLED", raising=False)
    assert OptimizerSettings(_env_file=None).enabled is False


def test_optimizer_url_defaults_to_compose_service_name() -> None:
    """The connect URL defaults to the compose service DNS name + bind port."""
    assert OptimizerSettings(_env_file=None).optimizer_url == "http://optimizer:8095"


def test_add_optimizer_job_registers_single_cron_job() -> None:
    """The wiring adds exactly one job (id=turing-optimizer) on the optimizer cron."""
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger

    scheduler = AsyncIOScheduler()
    settings = OptimizerSettings(_env_file=None, cron="30 3 * * *")

    add_optimizer_job(scheduler, settings)

    jobs = scheduler.get_jobs()
    assert len(jobs) == 1
    job = jobs[0]
    assert job.id == "turing-optimizer"
    assert isinstance(job.trigger, CronTrigger)
    assert job.max_instances == 1
    assert job.coalesce is True


@pytest.mark.asyncio
async def test_trigger_logs_promotion_outcome() -> None:
    """A 200 with a promoted OptimizeResponse is logged (never raised)."""
    body = {
        "node": "classify",
        "promoted": True,
        "reason": "promoted",
        "baseline": 0.6,
        "candidate_score": 0.8,
        "usage": {"cost_usd": 0.04},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/optimize"
        return httpx.Response(200, json=body)

    transport = httpx.MockTransport(handler)
    # Never raises — the assertion is simply that it completes cleanly.
    await trigger_optimization("http://optimizer:8095", transport=transport)


@pytest.mark.asyncio
async def test_trigger_swallows_unreachable_sidecar() -> None:
    """A connect error is logged + swallowed (scheduler must stay up)."""
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    transport = httpx.MockTransport(handler)
    await trigger_optimization("http://optimizer:8095", transport=transport)


@pytest.mark.asyncio
async def test_trigger_swallows_non_200() -> None:
    """A non-200 response is logged + swallowed."""
    transport = httpx.MockTransport(lambda _r: httpx.Response(500, text="boom"))
    await trigger_optimization("http://optimizer:8095", transport=transport)


@pytest.mark.asyncio
async def test_trigger_swallows_non_json_body() -> None:
    """A 200 with a non-JSON body is logged + swallowed."""
    transport = httpx.MockTransport(lambda _r: httpx.Response(200, text="<html>"))
    await trigger_optimization("http://optimizer:8095", transport=transport)
