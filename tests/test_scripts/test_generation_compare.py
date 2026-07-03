"""Unit tests for scripts/generation_compare.py — the G0→G1→G2 delta table.

Hermetic tests of the PURE comparison core: summarize(), build_per_goal(),
build_comparison(), and the formatting helpers. Fake reports are built from the
run_metrics dataclasses (loaded transitively); no DB, no provider key.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Load the standalone script (lives in scripts/; it loads run_metrics itself).
_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "generation_compare.py"
_spec = importlib.util.spec_from_file_location("generation_compare", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
gc = importlib.util.module_from_spec(_spec)
sys.modules["generation_compare"] = gc
_spec.loader.exec_module(gc)

# run_metrics is loaded + bound inside generation_compare as gc.rm — reuse it to
# build fake reports from real dataclasses.
rm = gc.rm


def _goal(goal_id: str, score: float, n_checks: int = 1) -> rm.GoalScore:
    return rm.GoalScore(
        goal_id=goal_id,
        run_id=f"cli-{goal_id}",
        score=score,
        n_checks=n_checks,
        n_rows=n_checks,
        verify_passes_estimate=1,
        attempt_id="A1",
    )


def _report(
    *,
    battery_mean: float,
    goals: list[tuple[str, float]],
    cost_usd: float,
    input_tokens: int,
    cached_tokens: int,
    hit_rate: float,
    calls: int,
    span: float | None,
    subs_delegated: int | None,
    tools_created: int,
) -> rm.RunMetricsReport:
    score = rm.ScoreSummary(
        battery_mean=battery_mean,
        n_goals_ran=len(goals),
        per_goal=[_goal(g, s) for g, s in goals],
    )
    cost = rm.CostSummary(
        total_calls=calls,
        input_tokens=input_tokens,
        output_tokens=input_tokens // 10,
        cached_tokens=cached_tokens,
        total_tokens=input_tokens + input_tokens // 10,
        total_cost_usd=cost_usd,
        by_model=[],
        cache_hit_rate=hit_rate,
    )
    subs = (
        rm.SubagentSummary(
            delegated=subs_delegated,
            completed=subs_delegated,
            failed=0,
            timeout=0,
            total_tokens=0,
            total_cost_usd=0.0,
        )
        if subs_delegated is not None
        else None
    )
    return rm.RunMetricsReport(
        selector="x",
        matched_run_ids=["cli-x"],
        score=score,
        cost=cost,
        llm_span_seconds=span,
        subagents=subs,
        created=rm.CreatedSummary(tools_created, 0, None, None),
        global_tool_health=[],
    )


# ─── summarize ────────────────────────────────────────────────────────────────


def test_summarize_extracts_fields() -> None:
    rep = _report(
        battery_mean=0.9,
        goals=[("g1", 0.9)],
        cost_usd=1.23,
        input_tokens=1000,
        cached_tokens=200,
        hit_rate=0.2,
        calls=50,
        span=600.0,
        subs_delegated=3,
        tools_created=2,
    )
    s = gc.summarize("gen0", rep)
    assert s.suffix == "gen0"
    assert s.battery_mean == 0.9
    assert s.n_goals_ran == 1
    assert s.total_cost_usd == pytest.approx(1.23)
    assert s.input_tokens == 1000
    assert s.cached_tokens == 200
    assert s.cache_hit_rate == pytest.approx(0.2)
    assert s.llm_calls == 50
    assert s.llm_span_seconds == 600.0
    assert s.subagents_delegated == 3
    assert s.tools_created == 2


def test_summarize_handles_missing_subagents() -> None:
    rep = _report(
        battery_mean=0.5, goals=[("g1", 0.5)], cost_usd=0.1, input_tokens=10,
        cached_tokens=0, hit_rate=0.0, calls=1, span=None, subs_delegated=None,
        tools_created=0,
    )
    s = gc.summarize("gen1", rep)
    assert s.subagents_delegated is None
    assert s.llm_span_seconds is None


# ─── build_per_goal ───────────────────────────────────────────────────────────


def test_build_per_goal_union_and_missing_is_none() -> None:
    r0 = _report(battery_mean=0.5, goals=[("a", 0.5), ("b", 1.0)], cost_usd=0.1,
                 input_tokens=10, cached_tokens=0, hit_rate=0.0, calls=1, span=1.0,
                 subs_delegated=None, tools_created=0)
    r1 = _report(battery_mean=0.8, goals=[("b", 0.8), ("c", 1.0)], cost_usd=0.1,
                 input_tokens=10, cached_tokens=0, hit_rate=0.0, calls=1, span=1.0,
                 subs_delegated=None, tools_created=0)
    matrix, ran = gc.build_per_goal([r0, r1])
    assert set(matrix) == {"a", "b", "c"}
    assert matrix["a"] == [0.5, None]  # a only ran in gen0
    assert matrix["b"] == [1.0, 0.8]
    assert matrix["c"] == [None, 1.0]  # c only ran in gen1
    assert ran["a"] == [True, False]


# ─── build_comparison ────────────────────────────────────────────────────────


def test_build_comparison_end_to_end() -> None:
    r0 = _report(battery_mean=0.7, goals=[("g1", 0.7)], cost_usd=2.0,
                 input_tokens=1000, cached_tokens=0, hit_rate=0.0, calls=100,
                 span=500.0, subs_delegated=0, tools_created=0)
    r1 = _report(battery_mean=0.9, goals=[("g1", 0.9)], cost_usd=1.5,
                 input_tokens=800, cached_tokens=400, hit_rate=0.5, calls=80,
                 span=400.0, subs_delegated=1, tools_created=1)
    comp = gc.build_comparison(["gen0", "gen1"], [r0, r1])
    assert comp.suffixes == ["gen0", "gen1"]
    assert len(comp.summaries) == 2
    assert comp.summaries[1].battery_mean == 0.9
    assert comp.summaries[1].tools_created == 1
    assert comp.per_goal["g1"] == [0.7, 0.9]
    # JSON-serializable round-trip
    blob = comp.to_dict()
    assert blob["suffixes"] == ["gen0", "gen1"]
    assert len(blob["summaries"]) == 2


# ─── formatting helpers ──────────────────────────────────────────────────────


def test_fmt_and_delta() -> None:
    assert gc._fmt(None) == "—"
    assert gc._fmt(0.5, 4) == "0.5000"
    assert gc._fmt(42, 0) == "42"
    assert gc._delta(0.9, 0.7) == "+0.2000"
    assert gc._delta(None, 0.7) == ""
    assert gc._delta(5, 8) == "-3"


def test_render_table_smoke(capsys: pytest.CaptureFixture[str]) -> None:
    """Render must not raise and must print the headline + per-goal sections."""
    r0 = _report(battery_mean=0.7, goals=[("g1", 0.7)], cost_usd=2.0,
                 input_tokens=1000, cached_tokens=0, hit_rate=0.0, calls=100,
                 span=500.0, subs_delegated=0, tools_created=0)
    r1 = _report(battery_mean=0.9, goals=[("g1", 0.9)], cost_usd=1.5,
                 input_tokens=800, cached_tokens=400, hit_rate=0.5, calls=80,
                 span=400.0, subs_delegated=1, tools_created=1)
    comp = gc.build_comparison(["gen0", "gen1"], [r0, r1])
    gc.render_table(comp)
    out = capsys.readouterr().out
    assert "score" in out
    assert "PER-GOAL SCORE" in out
    assert "gen0" in out and "gen1" in out
