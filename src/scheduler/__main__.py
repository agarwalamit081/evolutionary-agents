"""Scheduler entrypoint — the nightly capability-curve battery feeder (#197).

Runnable as ``python -m src.scheduler``. Builds the Redis-backed queue +
``BatteryEnqueuer`` and an APScheduler that, on ``SCHEDULER_CRON``, enqueues
every ``BATTERY04_GOALS`` spec as a ``RunJob`` into ``turing:runs`` (the same
seam the API uses). The worker then runs them through the real deployed stack
and the eval layer populates ``eval_results`` — the autonomous capability curve,
instead of only when a human runs ``--eval``.

Opt-in: ``SCHEDULER_ENABLED`` defaults False; a host run with no env is a clean
no-op (exit 0). The compose ``scheduler`` service forces it true AND is
profile-gated, so bringing the profile up is the explicit opt-in.

Graceful SIGTERM/SIGINT → ``scheduler.shutdown(wait=False)``: an in-flight
enqueue is abandoned (it is per-run safe — a half-enqueued battery simply
enqueues fewer specs that night; nothing is corrupted), then the process exits.
``restart: unless-stopped`` (compose) retries the process after a Redis outage.
"""

from __future__ import annotations

import asyncio
import signal
import sys

from loguru import logger

from src.config import get_settings
from src.eval.golden import BATTERY04_GOALS
from src.observability.logging import setup_logging
from src.scheduler.battery import BatteryEnqueuer, make_battery_scheduler
from src.worker.queue import RunsQueue


async def _run() -> int:
    """Build the enqueuer + scheduler and run until stopped. Returns exit code."""
    settings = get_settings()
    setup_logging(settings.logging)
    sched_settings = settings.scheduler

    if not sched_settings.enabled:
        logger.info(
            "Battery scheduler disabled (SCHEDULER_ENABLED=false); exiting cleanly. "
            "The compose `scheduler` service forces this on; a host run sets it in env."
        )
        return 0

    import redis.asyncio as aioredis

    redis_client = aioredis.from_url(settings.redis.redis_url)
    # Fail fast + exit 1 if Redis is down: a scheduler that cannot enqueue is
    # useless. compose `depends_on: redis: service_healthy` covers cold start; a
    # later outage → exit 1 → `restart: unless-stopped` retries until Redis recovers.
    try:
        await redis_client.ping()  # type: ignore[union-attr]  # redis.asyncio stub returns sync bool
    except Exception as exc:  # noqa: BLE001 — best-effort close on the error path
        logger.error(f"Redis unreachable at {settings.redis.redis_url}: {exc}")
        try:
            await redis_client.aclose()
        except Exception:  # noqa: BLE001
            pass
        return 1

    # Reuse the worker's stream/group names so enqueued jobs land in the EXACT
    # stream the worker drains — no separate producer path to drift.
    queue = RunsQueue(redis_client, settings.worker)
    enqueuer = BatteryEnqueuer(queue, sched_settings)
    scheduler = make_battery_scheduler(enqueuer, sched_settings)

    # Phase 2 C1f: the nightly curve regression→rollback gate. Registers on the
    # SAME scheduler as the battery job, only when CAPABILITY_CURVE_GATE_ENABLED.
    # Fires at capability_curve.curve_cron (default 05:00 UTC — after the 02:00
    # battery so it reads the just-written night). Auto-rollback is a separate,
    # independent opt-in (CAPABILITY_CURVE_AUTO_ROLLBACK).
    curve_settings = settings.capability_curve
    if curve_settings.gate_enabled:
        from src.eval.curve import CapabilityCurve  # noqa: PLC0415
        from src.eval.store import EvalStore  # noqa: PLC0415
        from src.evolution.curve_gate import (  # noqa: PLC0415
            CurveRegressionGate,
            add_curve_gate_job,
        )
        from src.evolution.promote import PromotionGate  # noqa: PLC0415

        curve = CapabilityCurve(EvalStore())
        gate = CurveRegressionGate(curve, PromotionGate(), settings=curve_settings)
        add_curve_gate_job(scheduler, gate, curve_settings)
        logger.info(
            f"Capability-curve gate registered — cron={curve_settings.curve_cron!r} "
            f"tz={curve_settings.timezone} auto_rollback={curve_settings.auto_rollback}"
        )

    stop_event = asyncio.Event()

    def _request_stop(*_: object) -> None:
        logger.info("Scheduler stop signal received; shutting down")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except (NotImplementedError, RuntimeError):
            # add_signal_handler is unavailable on Windows / non-main threads;
            # fall through — CancelledError (asyncio.run teardown) still stops us.
            pass

    scheduler.start()
    logger.info(
        f"Battery scheduler running — cron={sched_settings.cron!r} "
        f"tz={sched_settings.timezone} specs={len(BATTERY04_GOALS)} → "
        f"stream={settings.worker.runs_stream}"
    )
    try:
        await stop_event.wait()
    except asyncio.CancelledError:
        logger.info("Scheduler cancelled")
        raise
    finally:
        scheduler.shutdown(wait=False)
        await redis_client.aclose()
    return 0


def main() -> None:
    """Module entrypoint: ``python -m src.scheduler``."""
    sys.exit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
