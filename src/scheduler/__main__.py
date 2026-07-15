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
from src.observability import init_process_observability
from src.observability.logging import setup_logging
from src.scheduler.battery import BatteryEnqueuer, make_battery_scheduler
from src.worker.queue import RunsQueue


async def _run() -> int:
    """Build the enqueuer + scheduler and run until stopped. Returns exit code."""
    settings = get_settings()
    setup_logging(settings.logging)
    # Observability (OTel tracing + Prometheus scrape server); opt-in, idempotent.
    init_process_observability(settings.observability, component="scheduler")
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
    # RunStatusStore lets the enqueuer poll each goal's terminal status for the
    # cross-query DAG-release (#575): a dependent is held until its upstream is
    # done. Constructed from the SAME redis + worker settings the queue uses.
    from src.worker.status import RunStatusStore  # noqa: PLC0415

    status_store = RunStatusStore(redis_client, settings.worker)
    enqueuer = BatteryEnqueuer(queue, sched_settings, status_store=status_store)
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

    # battery-04 q09 fix C: periodic capability-governance prune. Re-runs
    # consolidate.py retire/redundancy + the tool cumulative-cap enforce on
    # GOVERNANCE_PRUNE_CRON so a long-lived worker frees cap headroom BETWEEN
    # restarts (q09 saturated 25/25 tools mid-life and could never create more).
    # Opt-in (GOVERNANCE_PRUNE_ENABLED, default off). Default 04:00 UTC — a daily
    # debloat, well clear of the 05:00 curve-gate and the 02:00 battery.
    prune_settings = settings.governance_prune
    if prune_settings.enabled:
        from src.governance.prune import (  # noqa: PLC0415
            GovernancePruner,
            add_governance_prune_job,
        )

        add_governance_prune_job(scheduler, GovernancePruner(), prune_settings)
        logger.info(
            f"Governance-prune job registered — cron={prune_settings.cron!r} "
            f"tz={prune_settings.timezone}"
        )

    # Phase 6 q101: periodic checkpoint garbage-collection. langgraph's
    # AsyncPostgresSaver never expires checkpoint rows, so a long-lived worker
    # accumulates unbounded state (~14k ckpts / ~110k writes across ~240
    # threads in weeks). This drops WHOLE threads whose newest checkpoint is
    # older than the TTL (runs finish in <1 day). Opt-in (CHECKPOINT_GC_ENABLED,
    # default off) + dry-run default (CHECKPOINT_GC_DRY_RUN=true): logs
    # candidates + counts, deletes nothing until dry_run is flipped false.
    # Default 06:00 UTC — clear of the 02:00/03:30/04:00/05:00 jobs.
    checkpoint_gc_settings = settings.checkpoint_gc
    if checkpoint_gc_settings.enabled:
        from src.scheduler.checkpoint_gc import (  # noqa: PLC0415
            CheckpointGc,
            add_checkpoint_gc_job,
        )

        add_checkpoint_gc_job(
            scheduler, CheckpointGc(checkpoint_gc_settings), checkpoint_gc_settings
        )
        logger.info(
            f"Checkpoint-GC job registered — cron={checkpoint_gc_settings.cron!r} "
            f"tz={checkpoint_gc_settings.timezone} "
            f"ttl_days={checkpoint_gc_settings.ttl_days} "
            f"dry_run={checkpoint_gc_settings.dry_run}"
        )

    # Phase 2 C2: the nightly metric-driven prompt-optimizer TRIGGER. The
    # optimizer (DSPy + GEPA) runs in a separate sidecar container; the scheduler
    # just POSTs /optimize (empty body — the sidecar owns the node/backend/eval
    # knobs) on OPTIMIZER_CRON and logs the outcome. Opt-in (OPTIMIZER_ENABLED,
    # default off). Default 03:30 UTC — a fresh night after the 02:00 battery,
    # before the 04:00 prune / 05:00 curve-gate.
    optimizer_settings = settings.optimizer
    if optimizer_settings.enabled:
        from src.optimizer.job import add_optimizer_job  # noqa: PLC0415

        add_optimizer_job(scheduler, optimizer_settings)
        logger.info(
            f"Optimizer trigger registered — cron={optimizer_settings.cron!r} "
            f"tz={optimizer_settings.timezone} url={optimizer_settings.optimizer_url}"
        )

    # Phase 5 I1: agent-settable durable cron. The agent authors future runs into
    # ``scheduled_tasks`` via the ``create_scheduled_task`` builtin; this consumer
    # reconciles those rows onto this scheduler (per-task CronTrigger jobs) and
    # enqueues a RunJob through the SAME RunsQueue on each tick. Opt-in
    # (AGENT_CRON_ENABLED, default off). The periodic sync job + an immediate
    # reconcile arm pre-existing + mid-flight tasks without a daemon restart.
    agent_cron_settings = settings.agent_cron
    if agent_cron_settings.enabled:
        from src.scheduler.cron_consumer import (  # noqa: PLC0415
            AgentCronEnqueuer,
            make_agent_cron_sync_job,
        )

        cron_enqueuer = AgentCronEnqueuer(queue, agent_cron_settings)
        make_agent_cron_sync_job(scheduler, cron_enqueuer, agent_cron_settings)
        # Reconcile once immediately so existing tasks fire without waiting for
        # the first sync-interval tick (added before start() — APScheduler
        # schedules them when .start() runs).
        registered = await cron_enqueuer.reconcile(scheduler)
        logger.info(
            f"Agent-cron consumer registered — sync_interval_s="
            f"{agent_cron_settings.sync_interval_s} tz={agent_cron_settings.timezone} "
            f"tasks_armed={registered} → stream={settings.worker.runs_stream}"
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
