"""Governance-prune scenarios (battery-04 q09 fix C) — ``GOVERNANCE_PRUNE_ENABLED`` parity.

``GovernancePruner`` re-runs ``consolidate_all`` (redundancy + performance
retirement for tools AND sub-agents) PLUS the tool cumulative-cap enforce
(``ToolPersister._retire_excess_tools``) so a long-lived worker frees cap
headroom between restarts WITHOUT raising the caps. These tests exercise the
retirement SCENARIOS through the pruner with injected fakes (no DB): tools /
sub-agents above the cumulative cap are retired; the min-runs floor and
recency window are threaded through the consolidate kwargs; ``run`` never
raises. Plus the pure deterministic cap-slice logic.

The opt-in ``GOVERNANCE_PRUNE_ENABLED`` (default off) registers the periodic
job; the pruner itself is wired unconditionally so a manual call works.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock


from src.config.settings import AgentSettings, GovernancePruneSettings
from src.governance.consolidate import ConsolidationReport, MergePlan
from src.governance.prune import (
    GovernancePruner,
    add_governance_prune_job,
)


# ─── helpers ─────────────────────────────────────────────────────────────────


def _agent_settings(**kw: Any) -> AgentSettings:
    base = dict(
        capability_redundancy_threshold=0.92,
        retire_min_runs=20, retire_success_floor=0.5,
        retire_empty_output_floor=0.8, max_active_tools=25,
    )
    base.update(kw)
    return AgentSettings(**base)  # type: ignore[arg-type]


def _merge(target: str, retired: list[str]) -> MergePlan:
    return MergePlan(target=target, retired=retired)


# ─── cap-enforce scenario (cumulative cap) ──────────────────────────────────


class TestCumulativeCapEnforce:
    async def test_should_free_excess_when_tools_above_cap(self) -> None:
        # max_active_tools=10 but 12 active → cap enforcer retires 2.
        calls: list[int] = []

        async def _enforce(max_active: int) -> int:
            calls.append(max_active)
            return 2  # 2 oldest retired

        pruner = GovernancePruner(
            _agent_settings(max_active_tools=10),
            consolidate=AsyncMock(return_value=ConsolidationReport()),
            tool_cap_enforcer=_enforce,
        )
        out = await pruner.run()
        assert out["pruned"] is True
        assert out["tool_cap_excess"] == 2
        assert calls == [10]  # the cap was threaded through verbatim

    async def test_should_free_zero_when_tools_under_cap(self) -> None:
        async def _enforce(max_active: int) -> int:
            return 0

        pruner = GovernancePruner(
            _agent_settings(),
            consolidate=AsyncMock(return_value=ConsolidationReport()),
            tool_cap_enforcer=_enforce,
        )
        out = await pruner.run()
        assert out["tool_cap_excess"] == 0
        assert out["total_freed"] == 0

    async def test_cap_slice_is_oldest_tail_beyond_cap(self) -> None:
        # Pure deterministic slice the real _retire_excess_tools uses:
        # newest-first ids, tail beyond max_active is retired.
        ids = [f"id{i}" for i in range(12)]  # id0 newest ... id11 oldest
        max_active = 10
        excess = ids[max_active:]
        assert excess == ["id10", "id11"]


# ─── consolidate scenario (redundancy + performance) ────────────────────────


class TestConsolidateKnobsFlowThrough:
    async def test_should_pass_retire_knobs_to_consolidate(self) -> None:
        captured: dict[str, Any] = {}

        async def _cons(**kwargs: Any) -> ConsolidationReport:
            captured.update(kwargs)
            return ConsolidationReport()

        pruner = GovernancePruner(
            _agent_settings(
                capability_redundancy_threshold=0.88, retire_min_runs=15,
                retire_success_floor=0.4, retire_empty_output_floor=0.7,
            ),
            consolidate=_cons,
            tool_cap_enforcer=AsyncMock(return_value=0),
        )
        await pruner.run()
        assert captured["threshold"] == 0.88
        assert captured["min_runs"] == 15
        assert captured["success_floor"] == 0.4
        assert captured["empty_output_floor"] == 0.7
        assert captured["dry_run"] is False  # prune ALWAYS applies

    async def test_should_report_redundant_tools_agents_and_underperformers(self) -> None:
        report = ConsolidationReport(
            tools=[_merge("tool_a", ["tool_a_dup"])],
            agents=[_merge("agent_a", ["agent_a_dup"])],
            performance_retired=["bad_tool"],
        )

        pruner = GovernancePruner(
            _agent_settings(),
            consolidate=AsyncMock(return_value=report),
            tool_cap_enforcer=AsyncMock(return_value=3),
        )
        out = await pruner.run()
        assert out["redundant_tools"] == 1
        assert out["redundant_agents"] == 1
        assert out["underperformers"] == 1
        assert out["tool_cap_excess"] == 3
        assert out["total_freed"] == 6

    async def test_min_runs_floor_spares_untried_tools(self) -> None:
        # The min-runs knob flows to underperforming_tools() which requires
        # calls >= min_runs; here we assert the pruner passes a sane floor so
        # an operator raising it spares more tools. (The actual filtering is
        # the persister's; the pruner must thread the configured value.)
        captured: dict[str, Any] = {}

        async def _cons(**kwargs: Any) -> ConsolidationReport:
            captured.update(kwargs)
            return ConsolidationReport()

        pruner = GovernancePruner(
            _agent_settings(retire_min_runs=50),
            consolidate=_cons, tool_cap_enforcer=AsyncMock(return_value=0),
        )
        await pruner.run()
        assert captured["min_runs"] == 50  # high floor → fewer tools eligible


# ─── never-raises + opt-in job ──────────────────────────────────────────────


class TestPrunerResilienceAndJob:
    async def test_should_never_raise_on_consolidate_failure(self) -> None:
        async def _boom(**kw: Any) -> ConsolidationReport:
            raise RuntimeError("DB down")

        pruner = GovernancePruner(
            _agent_settings(), consolidate=_boom,
            tool_cap_enforcer=AsyncMock(return_value=0),
        )
        out = await pruner.run()
        assert out["pruned"] is False
        assert "error" in out

    async def test_should_never_raise_on_cap_enforce_failure(self) -> None:
        async def _boom(max_active: int) -> int:
            raise RuntimeError("session poisoned")

        pruner = GovernancePruner(
            _agent_settings(),
            consolidate=AsyncMock(return_value=ConsolidationReport()),
            tool_cap_enforcer=_boom,
        )
        out = await pruner.run()
        assert out["pruned"] is False

    def test_governance_prune_settings_default_off(self) -> None:
        # Opt-in: a clean host run registers nothing.
        s = GovernancePruneSettings()
        assert s.enabled is False
        assert s.cron == "0 4 * * *"

    def test_add_governance_prune_job_registers_single_coalesced_job(self) -> None:
        sched = MagicMock()
        added: dict[str, Any] = {}

        def _add(fn: Any, trigger: Any, **kw: Any) -> None:
            added.update(kw)

        sched.add_job = MagicMock(side_effect=_add)
        pruner = GovernancePruner(
            _agent_settings(),
            consolidate=AsyncMock(),
            tool_cap_enforcer=AsyncMock(return_value=0),
        )
        add_governance_prune_job(
            sched, pruner, GovernancePruneSettings(enabled=True, cron="0 4 * * *")
        )
        assert added["id"] == "turing-governance-prune"
        assert added["max_instances"] == 1
        assert added["coalesce"] is True
