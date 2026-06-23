"""Curve-level regression→rollback gate (Phase 2 C1c).

Temporal regression gate that watches the nightly battery trend across nights
and reverts a PROMPT promotion whose benefit did not hold — the missing safety
loop the single-goal ``GoldenCanary`` (gates at promotion time only) does not
provide. Consumes ``CapabilityCurve.detect_regression`` and orchestrates rollback
via the EXISTING ``PromotionGate.rollback`` (no new rollback code).

Auto-rollback is OPT-IN (``CAPABILITY_CURVE_AUTO_ROLLBACK``, default off): the
default behavior is detect + set the ``capability_curve_score`` gauge + increment
``capability_curve_regressions_total`` + record telemetry + log a WARNING — the
safety evidence without the risk. ``run`` never raises (rollback failures are
logged and swallowed; observability-only, like CostTracker/EvalStore resilience).

Suspect selection is a heuristic, NOT causal proof: the active PROMPT promotions
whose ``promoted_at`` falls within ``lookback_days`` are the prime suspects (a
CODE/TOOL mutation is shadow/DB-governed and not rollback-eligible here). A
regression with NO active promotion within the window is model/provider drift,
not a mutation — correctly handled as alert-only (nothing promotion-side to
revert). Idempotent: once rolled back the pointer's active entry is gone, so the
next run finds nothing to roll back.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

from loguru import logger

from src.config.settings import CapabilityCurveSettings, get_settings
from src.eval.curve import CapabilityCurve
from src.evolution.promote import PromotionGate
from src.observability.metrics import CAPABILITY_CURVE_REGRESSIONS, CAPABILITY_CURVE_SCORE

# Telemetry sink: async (event_type, event_data) -> None. Injectable for tests.
TelemetryFn = Callable[[str, dict[str, Any]], Awaitable[None]]


class CurveRegressionGate:
    """Apply the curve regression verdict and (opt-in) revert suspect promotions."""

    def __init__(
        self,
        curve: CapabilityCurve,
        gate: PromotionGate,
        *,
        telemetry: TelemetryFn | None = None,
        settings: CapabilityCurveSettings | None = None,
    ) -> None:
        self._curve = curve
        self._gate = gate
        self._telemetry = telemetry
        s = settings or get_settings().capability_curve
        self._auto_rollback = s.auto_rollback
        self._lookback_days = s.lookback_days

    async def run(self, *, now: datetime | None = None) -> dict[str, Any]:
        """Detect a regression; set metrics; optionally revert suspect promotions.

        Never raises — rollback failures are logged and swallowed (observability-only).
        Returns the curve verdict augmented with ``rollbacks`` (list of per-node
        rollback results; empty unless auto_rollback reverts something).
        """
        verdict = await self._curve.detect_regression()
        current = verdict.get("current")
        if CAPABILITY_CURVE_SCORE is not None and isinstance(current, (int, float)):
            CAPABILITY_CURVE_SCORE.set(float(current))

        verdict["rollbacks"] = []
        if not verdict.get("regressed"):
            return verdict

        if CAPABILITY_CURVE_REGRESSIONS is not None:
            CAPABILITY_CURVE_REGRESSIONS.inc()
        await self._emit("curve_regression", {k: v for k, v in verdict.items() if k != "rollbacks"})

        if not self._auto_rollback:
            logger.warning(
                "Battery curve regression detected (auto_rollback OFF — alert only): "
                "current={} best_prior={} delta={} n_points={}",
                verdict.get("current"),
                verdict.get("best_prior"),
                verdict.get("delta"),
                verdict.get("n_points"),
            )
            return verdict

        suspects = self._suspect_nodes(now or _utcnow())
        if not suspects:
            logger.warning(
                "Battery curve regression detected but NO active promotion within "
                "lookback={}d — model/provider drift, not a mutation (alert only)",
                self._lookback_days,
            )
            return verdict

        verdict["rollbacks"] = await self._rollback_suspects(suspects)
        logger.warning(
            "Battery curve regression detected; rolled back {} suspect promotion(s): {}",
            len(verdict["rollbacks"]),
            verdict["rollbacks"],
        )
        return verdict

    def _suspect_nodes(self, now: datetime) -> list[dict[str, Any]]:
        """Active promotions whose ``promoted_at`` is within ``lookback_days`` of ``now``."""
        cutoff = now - timedelta(days=self._lookback_days)
        out: list[dict[str, Any]] = []
        for entry in self._gate.active_promotions():
            promoted = _parse_iso(entry.get("promoted_at"))
            if promoted is not None and promoted >= cutoff:
                out.append(entry)
        return out

    async def _rollback_suspects(self, suspects: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for entry in suspects:
            node = entry.get("node")
            if not node:
                continue
            try:
                res = self._gate.rollback(node)
                results.append({"node": node, "result": res})
                await self._emit("curve_regression_rollback", {"node": node, "result": res})
            except Exception as exc:  # noqa: BLE001 — never abort the gate on a rollback failure
                logger.warning("Curve-gate rollback failed for node={}: {}", node, exc)
                results.append({"node": node, "error": str(exc)})
        return results

    async def _emit(self, event_type: str, event_data: dict[str, Any]) -> None:
        """Record a telemetry event via the injected sink, else the EvolutionPersister default."""
        if self._telemetry is not None:
            try:
                await self._telemetry(event_type, event_data)
            except Exception as exc:  # noqa: BLE001 — telemetry is strictly non-critical
                logger.debug("Curve-gate telemetry failed ({}): {}", event_type, exc)
            return
        await _default_telemetry(event_type, event_data)


async def _default_telemetry(event_type: str, event_data: dict[str, Any]) -> None:
    """Record an evolution telemetry event with no chain (nullable chain_id)."""
    try:
        from src.evolution.persister import EvolutionPersister  # noqa: PLC0415

        await EvolutionPersister().record_event(None, event_type, event_data)
    except Exception as exc:  # noqa: BLE001 — telemetry is strictly non-critical
        logger.debug("Curve-gate default telemetry failed ({}): {}", event_type, exc)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(value: Any) -> datetime | None:
    """Parse an ISO ``promoted_at`` string to an aware datetime (assume UTC when naive)."""
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def add_curve_gate_job(
    scheduler: Any,
    gate: CurveRegressionGate,
    settings_s: CapabilityCurveSettings,
) -> None:
    """Register the nightly ``turing-curve-gate`` job on ``scheduler``.

    apscheduler is imported LAZILY (inside this function) so importing this module
    never requires the dep — mirroring ``make_battery_scheduler``. The job fires the
    gate on ``settings_s.curve_cron`` (default 05:00 UTC, after the 02:00 battery so
    it reads the just-written night). Same discipline as the battery job:
    ``max_instances=1, coalesce=True, misfire_grace_time=3600``.
    """
    from apscheduler.triggers.cron import CronTrigger  # noqa: PLC0415

    async def _fire() -> None:
        await gate.run()

    scheduler.add_job(
        _fire,
        CronTrigger.from_crontab(settings_s.curve_cron, timezone=settings_s.timezone),
        id="turing-curve-gate",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3600,
    )
