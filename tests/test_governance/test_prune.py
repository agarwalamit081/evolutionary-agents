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
from unittest.mock import AsyncMock

import pytest

from src.config.settings import AgentSettings
from src.governance.consolidate import ConsolidationReport, MergePlan
from src.governance.prune import (
    GovernancePruner,
    enforce_caps_now,
    maybe_enforce_caps_mid_run,
    should_enforce_caps_now,
)


def _settings() -> AgentSettings:
    """Build an AgentSettings isolated from .env (no .env reads)."""
    return AgentSettings(_env_file=None)


def _cap_settings(*, enabled: bool, interval: int = 10) -> AgentSettings:
    """AgentSettings with the #4 mid-run-cap knobs set (still .env-isolated)."""
    return AgentSettings(
        _env_file=None,
        mid_run_cap_enforce_enabled=enabled,
        mid_run_cap_enforce_interval=interval,
    )


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


# ─── #4 mid-run capability-cap enforcement ───────────────────────────
# ``enforce_caps_now`` (thin delegator) + ``should_enforce_caps_now`` (pure
# cadence gate) + ``maybe_enforce_caps_mid_run`` (the single node entry point).
# The node wiring (tool_create / agent_spawn calling ``maybe_enforce_caps_mid_run``
# and stamping the return) is pinned in
# ``tests/test_graph/test_nodes/test_mid_run_cap_enforce.py``.


class TestShouldEnforceCapsNow:
    """Pure cadence gate — pinned without a DB/gateway."""

    def test_flag_off_never_fires(self) -> None:
        s = _cap_settings(enabled=False)
        assert should_enforce_caps_now(100, 0, settings=s) is False

    def test_not_enough_iters_elapsed(self) -> None:
        s = _cap_settings(enabled=True, interval=10)
        # last enforced at 0, now at 5 → 5 < 10
        assert should_enforce_caps_now(5, 0, settings=s) is False
        # last enforced at 5, now at 13 → 8 < 10
        assert should_enforce_caps_now(13, 5, settings=s) is False

    def test_fires_at_exactly_interval(self) -> None:
        s = _cap_settings(enabled=True, interval=10)
        # first eligible round (never enforced, now == interval)
        assert should_enforce_caps_now(10, 0, settings=s) is True
        # exactly one interval since the last
        assert should_enforce_caps_now(20, 10, settings=s) is True

    def test_fires_past_interval(self) -> None:
        s = _cap_settings(enabled=True, interval=3)
        assert should_enforce_caps_now(100, 0, settings=s) is True

    def test_interval_zero_or_negative_clamped_to_one(self) -> None:
        # A misconfigured <=0 interval must never divide-by-zero or fire every
        # round; clamped to 1 so delta >= 1 fires, delta 0 does not.
        s0 = _cap_settings(enabled=True, interval=0)
        assert should_enforce_caps_now(0, 0, settings=s0) is False
        assert should_enforce_caps_now(1, 0, settings=s0) is True
        s_neg = _cap_settings(enabled=True, interval=-5)
        assert should_enforce_caps_now(2, 0, settings=s_neg) is True


class TestEnforceCapsNow:
    """``enforce_caps_now`` is a thin delegator over GovernancePruner.run()."""

    async def test_delegates_to_pruner_and_returns_report(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        class FakePruner:
            def __init__(self, settings: Any) -> None:
                captured["settings"] = settings

            async def run(self) -> dict[str, Any]:
                return {"pruned": True, "total_freed": 7}

        monkeypatch.setattr("src.governance.prune.GovernancePruner", FakePruner)

        s = _settings()
        report = await enforce_caps_now(s)

        assert report == {"pruned": True, "total_freed": 7}
        assert captured["settings"] is s

    async def test_default_settings_passes_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # settings=None must delegate to GovernancePruner(None) — proving the
        # default path needs no caller-supplied settings (and the fake never
        # reads get_settings, so no .env read here).
        captured: dict[str, Any] = {}

        class FakePruner:
            def __init__(self, settings: Any) -> None:
                captured["settings"] = settings

            async def run(self) -> dict[str, Any]:
                return {"pruned": False, "error": "transient"}

        monkeypatch.setattr("src.governance.prune.GovernancePruner", FakePruner)

        report = await enforce_caps_now()

        assert report == {"pruned": False, "error": "transient"}
        assert captured["settings"] is None


class TestMaybeEnforceCapsMidRun:
    """The single node entry point: trigger × cadence → fire-or-skip."""

    async def test_trigger_off_returns_none_no_prune(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # fire=False short-circuits before the cadence gate — even with the flag
        # ON and the cadence satisfied, the prune never runs.
        run_mock = AsyncMock(return_value={"pruned": True, "total_freed": 5})
        monkeypatch.setattr("src.governance.prune.enforce_caps_now", run_mock)

        s = _cap_settings(enabled=True, interval=1)
        out = await maybe_enforce_caps_mid_run(
            current_iter=10, last_enforced_iter=0, fire=False, settings=s
        )

        assert out is None
        run_mock.assert_not_awaited()

    async def test_cadence_blocks_returns_none_no_prune(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_mock = AsyncMock(return_value={"pruned": True, "total_freed": 5})
        monkeypatch.setattr("src.governance.prune.enforce_caps_now", run_mock)

        s = _cap_settings(enabled=True, interval=10)  # delta 3 < 10
        out = await maybe_enforce_caps_mid_run(
            current_iter=3, last_enforced_iter=0, fire=True, settings=s
        )

        assert out is None
        run_mock.assert_not_awaited()

    async def test_trigger_and_cadence_fire_returns_current_iter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_mock = AsyncMock(return_value={"pruned": True, "total_freed": 4})
        monkeypatch.setattr("src.governance.prune.enforce_caps_now", run_mock)

        s = _cap_settings(enabled=True, interval=10)
        out = await maybe_enforce_caps_mid_run(
            current_iter=25, last_enforced_iter=10, fire=True, settings=s
        )

        assert out == 25  # the new mid_run_cap_last_enforced_iter
        run_mock.assert_awaited_once_with(s)

    async def test_flag_off_returns_none_even_when_fired(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Default-off behavior: the node may pass fire=True, but with the flag
        # off the cadence gate (should_enforce_caps_now) is False → no prune.
        run_mock = AsyncMock(return_value={"pruned": True, "total_freed": 9})
        monkeypatch.setattr("src.governance.prune.enforce_caps_now", run_mock)

        out = await maybe_enforce_caps_mid_run(
            current_iter=50, last_enforced_iter=0, fire=True, settings=_cap_settings(enabled=False)
        )

        assert out is None
        run_mock.assert_not_awaited()

