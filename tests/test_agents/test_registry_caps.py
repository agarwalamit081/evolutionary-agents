"""Tests for cumulative caps + retirement (B3) in SubAgentRegistry.

Covers the generalized :meth:`check_deprecation` retirement triggers (chronic
low performer, stale-by-recency, never-used protection) and the new
:meth:`enforce_caps` (apply deprecation, then retire lowest-scoring survivors
down to ``max_active``). The legacy single-arg behavior is exercised by the
existing tests/test_agents/test_registry.py::TestCheckDeprecation suite.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TypedDict

from src.agents.registry import SubAgentRegistry
from src.graph.models import SubAgentSpec

# Fixed "now" for deterministic recency math.
NOW = datetime(2026, 6, 16, 12, 0, tzinfo=timezone.utc)


class _RetireKwargs(TypedDict):
    """Keyword args for the stricter retire policy (vs legacy defaults)."""

    min_runs: int
    success_floor: float
    recency_days: int


_KW: _RetireKwargs = {"min_runs": 20, "success_floor": 0.25, "recency_days": 30}


def _spec(
    name: str,
    *,
    total_runs: int = 0,
    success_rate: float = 0.0,
    quality_score: float = 0.5,
    last_used_at: datetime | None = None,
    is_active: bool = True,
) -> SubAgentSpec:
    return SubAgentSpec(
        name=name,
        description="d",
        goal="g",
        parent_thread_id="t",
        total_runs=total_runs,
        success_rate=success_rate,
        quality_score=quality_score,
        last_used_at=last_used_at,
        is_active=is_active,
    )


class TestCheckDeprecationTriggers:
    def test_chronic_low_performer_retired(self) -> None:
        reg = SubAgentRegistry()
        reg.register(_spec("bad", total_runs=25, success_rate=0.1))
        assert reg.check_deprecation("bad", **_KW, now=NOW) is True
        assert reg.get("bad").is_active is False

    def test_low_runs_not_retired(self) -> None:
        """Bad success but not enough runs → keep (insufficient evidence)."""
        reg = SubAgentRegistry()
        reg.register(_spec("bad", total_runs=5, success_rate=0.1))
        assert reg.check_deprecation("bad", **_KW, now=NOW) is False
        assert reg.get("bad").is_active is True

    def test_stale_retired_even_if_good_performer(self) -> None:
        """A previously-good agent unused for the window is dead weight."""
        old = NOW - timedelta(days=45)
        reg = SubAgentRegistry()
        reg.register(
            _spec("old", total_runs=100, success_rate=0.9, last_used_at=old)
        )
        assert reg.check_deprecation("old", **_KW, now=NOW) is True
        assert reg.get("old").is_active is False

    def test_fresh_good_performer_kept(self) -> None:
        recent = NOW - timedelta(days=5)
        reg = SubAgentRegistry()
        reg.register(
            _spec("good", total_runs=100, success_rate=0.9, last_used_at=recent)
        )
        assert reg.check_deprecation("good", **_KW, now=NOW) is False

    def test_never_used_is_not_stale(self) -> None:
        """A brand-new capability (last_used_at None) gets a chance, not retired."""
        reg = SubAgentRegistry()
        reg.register(
            _spec("new", total_runs=0, success_rate=0.0, last_used_at=None)
        )
        assert reg.check_deprecation("new", **_KW, now=NOW) is False
        assert reg.get("new").is_active is True

    def test_unknown_returns_false(self) -> None:
        assert SubAgentRegistry().check_deprecation("nope", now=NOW) is False


class TestEnforceCaps:
    def test_under_cap_with_bad_performer_retires_only_bad(self) -> None:
        reg = SubAgentRegistry()
        reg.register(_spec("bad", total_runs=25, success_rate=0.1, last_used_at=NOW))
        reg.register(_spec("good", total_runs=25, success_rate=0.9, last_used_at=NOW))
        retired = reg.enforce_caps(max_active=15, **_KW, now=NOW)
        assert retired == ["bad"]
        assert reg.get("bad").is_active is False
        assert reg.get("good").is_active is True

    def test_overflow_retires_lowest_score(self) -> None:
        """All fresh good performers but over cap → lowest score retired."""
        reg = SubAgentRegistry()
        reg.register(_spec("low", total_runs=30, success_rate=0.6, last_used_at=NOW))
        reg.register(_spec("mid", total_runs=30, success_rate=0.7, last_used_at=NOW))
        reg.register(_spec("high", total_runs=30, success_rate=0.9, last_used_at=NOW))
        retired = reg.enforce_caps(max_active=2, **_KW, now=NOW)
        assert retired == ["low"]  # check_deprecation fires for none; overflow picks lowest
        assert reg.active_count == 2
        assert reg.get("low").is_active is False
        assert reg.get("high").is_active is True

    def test_overflow_respects_cap_exactly(self) -> None:
        reg = SubAgentRegistry()
        for i in range(2):
            reg.register(_spec(f"a{i}", total_runs=30, success_rate=0.9, last_used_at=NOW))
        assert reg.enforce_caps(max_active=2, **_KW, now=NOW) == []
        assert reg.active_count == 2

    def test_combined_bad_and_overflow(self) -> None:
        """A bad performer is retired first; remaining overflow trimmed."""
        reg = SubAgentRegistry()
        reg.register(_spec("bad", total_runs=25, success_rate=0.1, last_used_at=NOW))
        reg.register(_spec("a", total_runs=30, success_rate=0.8, last_used_at=NOW))
        reg.register(_spec("b", total_runs=30, success_rate=0.85, last_used_at=NOW))
        reg.register(_spec("c", total_runs=30, success_rate=0.9, last_used_at=NOW))
        retired = reg.enforce_caps(max_active=2, **_KW, now=NOW)
        # bad retired by deprecation; then 3 active over cap 2 → lowest (a) retired.
        assert "bad" in retired and "a" in retired
        assert reg.active_count == 2

    def test_tiebreak_on_total_runs(self) -> None:
        """Equal success_rate → fewer runs retired first."""
        reg = SubAgentRegistry()
        reg.register(_spec("few", total_runs=5, success_rate=0.8, last_used_at=NOW))
        reg.register(_spec("many", total_runs=50, success_rate=0.8, last_used_at=NOW))
        reg.register(_spec("top", total_runs=50, success_rate=0.9, last_used_at=NOW))
        retired = reg.enforce_caps(max_active=2, **_KW, now=NOW)
        assert retired == ["few"]  # lowest (success, runs, quality) tuple
        assert reg.get("many").is_active is True
