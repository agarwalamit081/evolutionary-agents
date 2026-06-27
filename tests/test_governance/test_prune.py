"""Unit tests for the periodic capability-governance prune (battery-04 q09 fix C).

The pruner re-runs the existing retire/redundancy pass (consolidate_all, dry_run=False)
+ the tool cumulative-cap enforce so a long-lived worker frees cap headroom between
restarts (q09 saturated 25/25 tools mid-life and could never create more). ``run`` is
never-raise (a DB error is reported, not propagated) — matching the CostTracker /
CurveRegressionGate observability-only contract. The consolidate fn and the tool
cap-enforcer are injected so the unit never opens a DB session.
"""

from __future__ import annotations

from typing import Any

from src.config.settings import AgentSettings
from src.governance.consolidate import ConsolidationReport, MergePlan
from src.governance.prune import GovernancePruner


def _settings() -> AgentSettings:
    """Build an AgentSettings isolated from .env (no .env reads)."""
    return AgentSettings(_env_file=None)


async def test_run_applies_real_knobs_and_enforces_tool_cap() -> None:
    """run() calls consolidate_all(dry_run=False) + the tool cap-enforcer, threading the
    AgentSettings retire knobs verbatim (asserted against the settings object, not literals,
    so a default change surfaces here rather than desyncing silently)."""
    settings = _settings()
    captured: dict[str, Any] = {}

    async def fake_consolidate(**kwargs: Any) -> ConsolidationReport:
        captured["consolidate_kwargs"] = kwargs
        return ConsolidationReport(
            tools=[MergePlan(target="keep_a", retired=["dup_a"])],
            agents=[MergePlan(target="keep_b", retired=["dup_b"])],
            performance_retired=["bad_tool"],
            dry_run=False,
        )

    async def fake_enforcer(max_active: int) -> int:
        captured["enforce_max_active"] = max_active
        return 3

    async def fake_unused_enforcer(min_age_days: int) -> int:
        captured["unused_min_age_days"] = min_age_days
        return 2  # 2 never-invoked dead-weight tools retired

    pruner = GovernancePruner(
        settings,
        consolidate=fake_consolidate,
        tool_cap_enforcer=fake_enforcer,
        unused_enforcer=fake_unused_enforcer,
    )

    result = await pruner.run()

    kwargs = captured["consolidate_kwargs"]
    assert kwargs["dry_run"] is False, "prune must APPLY, not dry-run"
    assert kwargs["threshold"] == settings.capability_redundancy_threshold
    assert kwargs["min_runs"] == settings.retire_min_runs
    assert kwargs["success_floor"] == settings.retire_success_floor
    assert kwargs["empty_output_floor"] == settings.retire_empty_output_floor
    assert captured["enforce_max_active"] == settings.max_active_tools
    # The unused dead-weight age gate threads through verbatim (default 30).
    assert captured["unused_min_age_days"] == settings.retire_unused_days
    # 1 redundant-tool plan + 1 redundant-agent plan + 1 underperformer + 2 unused + 3 cap-excess = 8.
    # (redundant_* count MergePlan objects, not retired items — see ConsolidationReport.)
    assert result == {
        "pruned": True,
        "total_freed": 8,
        "redundant_tools": 1,
        "redundant_agents": 1,
        "underperformers": 1,
        "unused": 2,
        "tool_cap_excess": 3,
    }


async def test_run_never_raises_on_consolidate_error() -> None:
    """A DB hiccup in the consolidate pass is reported, not propagated."""
    async def exploding_consolidate(**_kwargs: Any) -> ConsolidationReport:
        raise RuntimeError("DB down")

    async def fake_enforcer(_max_active: int) -> int:
        return 0

    pruner = GovernancePruner(
        _settings(),
        consolidate=exploding_consolidate,
        tool_cap_enforcer=fake_enforcer,
    )

    result = await pruner.run()  # must not raise

    assert result["pruned"] is False
    assert "DB down" in result["error"]


async def test_run_never_raises_on_cap_enforce_error() -> None:
    """A DB hiccup in the cap-enforce pass is reported, not propagated."""
    async def fake_consolidate(**_kwargs: Any) -> ConsolidationReport:
        return ConsolidationReport(dry_run=False)

    async def exploding_enforcer(_max_active: int) -> int:
        raise RuntimeError("cap-enforce DB down")

    # Inject the unused enforcer so this resilience test never opens a real DB
    # session via _retire_unused (runs before the exploding cap-enforce).
    async def fake_unused_enforcer(_min_age_days: int) -> int:
        return 0

    pruner = GovernancePruner(
        _settings(),
        consolidate=fake_consolidate,
        tool_cap_enforcer=exploding_enforcer,
        unused_enforcer=fake_unused_enforcer,
    )

    result = await pruner.run()  # must not raise

    assert result["pruned"] is False
    assert "cap-enforce DB down" in result["error"]
