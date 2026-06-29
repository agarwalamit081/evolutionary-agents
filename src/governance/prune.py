"""Periodic capability-governance prune runner (battery-04 q09 fix C).

Governance already runs at worker-boot load time — ``ToolPersister.load_active_tools``
and ``SubAgentPersister.load_active_agents`` run ``retire_redundant`` → cumulative-cap
enforce → ``retire_underperforming`` when ``settings`` is threaded. But a long-lived
worker accumulates the active population ACROSS runs and never prunes between
restarts. q09 saturated its caps (25/25 tools, 60/60 sub-agents) mid-life and could
then never create a needed capability, looping spawn↔create with no progress until a
manual ``docker restart``.

This runner is the missing periodic prune: it re-runs the existing retire/redundancy
passes (``consolidate_all``, ``dry_run=False``) AND the tool cumulative-cap enforcement
(``_retire_excess_tools``) on a schedule — lowering active counts to free cap headroom
WITHOUT raising the caps themselves. ``run`` never raises (observability-only, like
``CostTracker`` / ``CurveRegressionGate``) so a DB hiccup can never abort the scheduler.

Why cap-enforce is needed at all: ``consolidate_all`` retires only REDUNDANT duplicates
(cosine >= threshold) plus chronic underperformers — but q09's 25 tools were DISTINCT
and mostly succeeding, so consolidation alone retired nothing; only the cumulative-cap
enforce (retire oldest over the cap) freed headroom.

Phase-4 dead-weight pass: ``consolidate_all`` + cap-enforce STILL miss one population —
0-call tools that are neither redundant, nor over-cap, nor have enough calls to
underperform (q07: 8 of 25 slots were never-invoked dead weight).
``retire_unused_days`` retires those (calls == 0, aged past the gate) so the prune is
no longer a no-op on accumulated cruft. Default 30 (mirrors ``retire_recency_days``);
``<= 0`` disables the pass.

Sub-agent cumulative-cap enforcement (``SubAgentRegistry.enforce_caps``) scores the
in-memory registry, so it is inherently load-time; this periodic prune handles the
tool cap directly + redundancy/performance for both tools and sub-agents, and a worker
restart reapplies the full sub-agent registry scoring. Reuses ``AgentSettings`` retire
knobs — no new retirement thresholds here.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from loguru import logger

from src.config.settings import AgentSettings, get_settings
from src.governance.consolidate import consolidate_all

# The cap-enforce method on ToolPersister (retire oldest active tools over the cap).
# ``Callable[[int], Awaitable[int]]``: takes max_active, returns the count retired.
ToolCapEnforcer = Callable[[int], Awaitable[int]]

# The unused-tool retire method on ToolPersister (retire 0-call tools aged past
# the gate). Same shape as ToolCapEnforcer: takes min_age_days, returns count.
UnusedToolEnforcer = Callable[[int], Awaitable[int]]


class GovernancePruner:
    """Re-run retire/redundancy + unused dead-weight + the tool cap-enforce.

    Mirrors the load-time governance sequence but standalone (no registry load), so
    a long-lived worker can debloat between restarts. Reuses ``AgentSettings`` retire
    knobs (``capability_redundancy_threshold`` / ``retire_min_runs`` /
    ``retire_success_floor`` / ``retire_empty_output_floor`` / ``max_active_tools`` /
    ``retire_unused_days``). The unused pass (``retire_unused_days``) closes the
    Phase-4 gap where 0-call dead weight survived every other retire path.
    """

    def __init__(
        self,
        settings: AgentSettings | None = None,
        *,
        consolidate: Callable[..., Awaitable[Any]] | None = None,
        tool_cap_enforcer: ToolCapEnforcer | None = None,
        unused_enforcer: UnusedToolEnforcer | None = None,
    ) -> None:
        s = settings or get_settings().agent
        self._threshold = s.capability_redundancy_threshold
        self._min_runs = s.retire_min_runs
        self._success_floor = s.retire_success_floor
        self._empty_output_floor = s.retire_empty_output_floor
        self._max_active_tools = s.max_active_tools
        self._retire_unused_days = s.retire_unused_days
        # Injectable for tests (mirrors CurveRegressionGate.telemetry). Defaults
        # resolve the real deps lazily inside run() so importing this module needs
        # no DB — and so a unit test never opens a session.
        self._consolidate = consolidate
        self._tool_cap_enforcer = tool_cap_enforcer
        self._unused_enforcer = unused_enforcer

    async def run(self) -> dict[str, Any]:
        """Prune redundant + underperforming + unused capabilities, enforce the cap.

        Never raises — a DB error is logged at WARNING and reported, not propagated,
        so the scheduler survives a governance hiccup (observability-only).
        """
        try:
            report = await self._consolidate_or_default(
                threshold=self._threshold,
                dry_run=False,
                min_runs=self._min_runs,
                success_floor=self._success_floor,
                empty_output_floor=self._empty_output_floor,
            )
            unused = await self._retire_unused()
            tool_excess = await self._enforce_tool_cap()

            redundant_tools = len(report.tools)
            redundant_agents = len(report.agents)
            underperformers = len(report.performance_retired)
            total = (
                redundant_tools + redundant_agents + underperformers + unused + tool_excess
            )
            logger.info(
                "Governance prune complete: freed {} capability slot(s) "
                "(redundant tools={} agents={} underperformers={} unused={} "
                "tool-cap-excess={})",
                total,
                redundant_tools,
                redundant_agents,
                underperformers,
                unused,
                tool_excess,
            )
            return {
                "pruned": True,
                "total_freed": total,
                "redundant_tools": redundant_tools,
                "redundant_agents": redundant_agents,
                "underperformers": underperformers,
                "unused": unused,
                "tool_cap_excess": tool_excess,
            }
        except Exception as exc:  # noqa: BLE001 — never abort the scheduler
            logger.warning("Governance prune failed (observability-only): {}", exc)
            return {"pruned": False, "error": str(exc)}

    async def _consolidate_or_default(self, **kwargs: Any) -> Any:
        """Run the consolidate pass — injected fn (tests) or the real consolidate_all."""
        fn = self._consolidate or consolidate_all
        return await fn(**kwargs)

    async def _enforce_tool_cap(self) -> int:
        """Retire oldest active tools over ``max_active_tools``; returns count freed.

        ``ToolPersister._retire_excess_tools`` is the same persisted cap-enforce the
        load path uses (``load_active_tools``); accessed with the established
        ``# noqa: SLF001`` convention consolidate.py already uses for persister
        privates. Injectable for tests so the unit never opens a session.
        """
        enforcer = self._tool_cap_enforcer
        if enforcer is None:
            from src.tools.dynamic.persister import ToolPersister  # noqa: PLC0415

            p = ToolPersister()
            enforcer = p._retire_excess_tools  # noqa: SLF001 — mirrors consolidate.py
        return int(await enforcer(self._max_active_tools) or 0)

    async def _retire_unused(self) -> int:
        """Retire never-invoked generated tools aged past ``retire_unused_days``.

        The Phase-4 complement to the underperformer pass:
        ``retire_underperforming`` spares untried tools (a fair chance before a
        performance verdict), so a 0-call tool that is neither redundant, nor
        over-cap, nor has enough calls to underperform would survive forever.
        This retires that objective dead weight (calls == 0, older than the age
        gate). Injectable for tests so the unit never opens a session.
        ``retire_unused_days <= 0`` disables the pass (returns 0, touches nothing).
        """
        if self._retire_unused_days <= 0:
            return 0
        enforcer = self._unused_enforcer
        if enforcer is None:
            from src.tools.dynamic.persister import ToolPersister  # noqa: PLC0415

            enforcer = ToolPersister().retire_unused
        return int(await enforcer(self._retire_unused_days) or 0)


async def enforce_caps_now(settings: AgentSettings | None = None) -> dict[str, Any]:
    """Run the same nightly governance prune immediately, mid-run (#4).

    Thin async wrapper over ``GovernancePruner(settings).run()`` — 100% reuse of
    the nightly retire/redundancy/unused/cap-enforce pipeline (no duplicated
    logic). Called by the ``tool_create`` / ``agent_spawn`` nodes after a creation
    round, gated by ``should_enforce_caps_now`` + ``MID_RUN_CAP_ENFORCE_ENABLED``,
    so a long-lived worker that accumulates the active population across runs can
    free DB-side headroom mid-run instead of saturating and looping until a
    restart. Never raises (``GovernancePruner.run`` is observability-only), so a
    DB hiccup can never break the creation round — the cap gate +
    ``consecutive_cap_blocks`` loop-break still bound the worst case.

    Returns ``GovernancePruner.run``'s report (``total_freed`` etc.).
    """
    return await GovernancePruner(settings).run()


def should_enforce_caps_now(
    current_iter: int,
    last_enforced_iter: int,
    settings: AgentSettings | None = None,
) -> bool:
    """Cadence gate for mid-run cap enforcement (#4).

    True iff ``MID_RUN_CAP_ENFORCE_ENABLED`` is on AND at least
    ``mid_run_cap_enforce_interval`` iterations have elapsed since the last
    mid-run enforce (``current_iter - last_enforced >= interval``). ``0`` for
    ``last_enforced_iter`` means "never enforced" → the first eligible creation
    round always fires (0 + interval >= interval). The interval is clamped to a
    minimum of 1 so a misconfigured ``<= 0`` can never divide-by-zero or fire on
    every single round. Pure (no I/O) so it is pinned without a DB/gateway.
    """
    s = settings or get_settings().agent
    if not s.mid_run_cap_enforce_enabled:
        return False
    interval = max(1, int(s.mid_run_cap_enforce_interval or 1))
    return (int(current_iter) - int(last_enforced_iter)) >= interval


async def maybe_enforce_caps_mid_run(
    *,
    current_iter: int,
    last_enforced_iter: int,
    fire: bool,
    settings: AgentSettings | None = None,
) -> int | None:
    """Cadence-gated mid-run cap enforcement for the creation nodes (#4).

    The single entry point ``tool_create`` / ``agent_spawn`` call after a
    creation round. ``fire`` is the per-node trigger — the node wants to enforce
    only after a meaningful round (a capability was created, growing the
    population, OR the cap was hit, signaling saturation). The cadence gate
    (``should_enforce_caps_now``) is the cross-round bound so a churny creation
    loop can't fire the prune every round.

    Returns ``current_iter`` (the new ``mid_run_cap_last_enforced_iter``) when the
    prune ran, so the caller stamps it into state; ``None`` when the trigger was
    off or the cadence gate did not allow it (no state write). Never raises —
    ``enforce_caps_now`` is observability-only.
    """
    if not fire or not should_enforce_caps_now(current_iter, last_enforced_iter, settings):
        return None
    report = await enforce_caps_now(settings)
    if report.get("pruned"):
        logger.info(
            "Mid-run capability-cap enforcement at iter {} freed {} slot(s)",
            current_iter,
            report.get("total_freed", 0),
        )
    else:
        logger.warning(
            "Mid-run capability-cap enforcement at iter {} reported "
            "non-success (observability-only): {}",
            current_iter,
            report.get("error"),
        )
    return current_iter


def add_governance_prune_job(
    scheduler: Any,
    pruner: GovernancePruner,
    settings_s: Any,
) -> None:
    """Register the periodic ``turing-governance-prune`` job on ``scheduler``.

    apscheduler is imported LAZILY (inside this function) so importing this module
    never requires the dep — mirroring ``add_curve_gate_job`` / ``make_battery_scheduler``.
    The job fires the pruner on ``settings_s.cron`` (default 04:00 UTC). Same
    discipline as the battery + curve-gate jobs: ``max_instances=1, coalesce=True,
    misfire_grace_time=3600`` so a missed fire (e.g. a brief scheduler outage) is
    coalesced into one prune, never piled up.
    """
    from apscheduler.triggers.cron import CronTrigger  # noqa: PLC0415

    async def _fire() -> None:
        await pruner.run()

    scheduler.add_job(
        _fire,
        CronTrigger.from_crontab(settings_s.cron, timezone=settings_s.timezone),
        id="turing-governance-prune",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3600,
    )
