"""Scheduler job that triggers one nightly prompt-optimization run (Phase 2 C2).

The metric-driven optimizer (DSPy + GEPA) runs in a SEPARATE container
(``Dockerfile.optimizer``). The scheduler — which owns the nightly cadence
(battery 02:00 → optimizer 03:30 → governance-prune 04:00 → curve-gate 05:00) —
fires one POST to the sidecar's ``POST /optimize`` on ``OPTIMIZER_CRON`` and logs
the structured outcome.

The scheduler is a PURE TRIGGER. It POSTs an EMPTY body: the sidecar resolves
node / backend / eval knobs from its OWN ``OptimizerSettings`` (the single source
of truth for WHAT to optimize lives in the sidecar; the scheduler knows only WHEN
(``cron``), WHERE (``optimizer_url``), and WHETHER (``enabled``)).

The job NEVER raises: a nightly trigger must not crash the scheduler process. The
sidecar being down, returning an error, or timing out is logged + swallowed
(observability-only, mirroring :class:`CurveRegressionGate` /
:class:`GovernancePruner`). The sidecar logs its OWN detailed outcome regardless
(see :mod:`src.optimizer.engine`), so a client-side read timeout is NON-CRITICAL
— it only forgoes the scheduler's one-line summary, never the optimization itself
(the aiohttp handler keeps running in the sidecar after the client disconnects).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from src.config.settings import OptimizerSettings

if TYPE_CHECKING:
    import httpx as _httpx

# A cost-bounded ($0.50) optimization run is long: DSPy/GEPA search against the
# cheap proxy metric, plus baseline + candidate GoldenCanary validation (full
# agent runs over the ``eval_spec_limit`` golden specs). 30 min comfortably
# covers a nightly run on cheap tiers; the sidecar completes + logs regardless,
# so a timeout here is non-critical (it forgoes only the scheduler's summary).
_READ_TIMEOUT_S = 1800.0
# Fail fast if the sidecar is down so the scheduler isn't blocked on the connect.
_CONNECT_TIMEOUT_S = 5.0


async def trigger_optimization(
    url: str, *, transport: _httpx.AsyncBaseTransport | None = None
) -> None:
    """POST one optimization attempt to the sidecar at ``url``; log the outcome.

    Never raises — every failure path (unreachable, non-200, non-JSON) is logged
    and swallowed so the scheduler process stays up. ``transport`` is the test
    seam (mirrors :class:`RunnerClient`): prod leaves it ``None``; tests inject
    an ``httpx.MockTransport`` to drive the response without a live server.
    """
    import httpx  # noqa: PLC0415 — lazy, mirroring the other scheduler jobs

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=_CONNECT_TIMEOUT_S,
                read=_READ_TIMEOUT_S,
                write=_CONNECT_TIMEOUT_S,
                pool=_CONNECT_TIMEOUT_S,
            ),
            transport=transport,
        ) as client:
            resp = await client.post(f"{url.rstrip('/')}/optimize", json={})
    except httpx.HTTPError as exc:
        logger.warning("Optimizer job: sidecar unreachable at {} ({})", url, exc)
        return

    if resp.status_code != 200:
        logger.warning(
            "Optimizer job: sidecar returned HTTP {}: {}",
            resp.status_code,
            resp.text[:200],
        )
        return

    try:
        data = resp.json()
    except ValueError:
        logger.warning("Optimizer job: sidecar returned a non-JSON body")
        return

    _log_outcome(data)


def _log_outcome(data: dict[str, Any]) -> None:
    """Log the structured ``OptimizeResponse`` (promoted OR skipped + reason)."""
    node = str(data.get("node") or "?")
    promoted = bool(data.get("promoted", False))
    reason = data.get("reason") or "unknown"
    baseline = data.get("baseline")
    score = data.get("candidate_score")
    cost = float((data.get("usage") or {}).get("cost_usd") or 0.0)
    verb = "PROMOTED" if promoted else "skipped"
    logger.info(
        "Optimizer job {} '{}': {} (baseline={}, candidate={}, ${:.4f})",
        verb,
        node,
        reason,
        baseline,
        score,
        cost,
    )


def add_optimizer_job(scheduler: Any, settings_s: OptimizerSettings) -> None:
    """Register the nightly ``turing-optimizer`` job on ``scheduler``.

    apscheduler is imported LAZILY (inside this function) so importing this
    module never requires the dep — mirroring :func:`add_curve_gate_job` /
    :func:`add_governance_prune_job`. The job fires on ``settings_s.cron``
    (default 03:30 UTC, between the 02:00 battery and the 04:00 prune / 05:00
    curve-gate so the optimizer runs on a fresh night). Same discipline as the
    other nightly jobs: ``max_instances=1, coalesce=True, misfire_grace_time=3600``
    — two optimization runs never overlap.
    """
    from apscheduler.triggers.cron import CronTrigger  # noqa: PLC0415

    url = settings_s.optimizer_url

    async def _fire() -> None:
        await trigger_optimization(url)

    scheduler.add_job(
        _fire,
        CronTrigger.from_crontab(settings_s.cron, timezone=settings_s.timezone),
        id="turing-optimizer",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3600,
    )
