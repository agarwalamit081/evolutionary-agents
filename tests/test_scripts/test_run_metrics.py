"""Unit tests for scripts/run_metrics.py — the per-run/per-generation metrics instrument.

Fast, hermetic tests of the PURE aggregation core (dict → dataclass): cost/token
roll-up, terminal-state scoring (self-correction not penalized), verify-pass
estimate, subagent roll-up, global tool health, created-counts, and the LLM
span. No DB, no provider key — the thin fetch layer is exercised live by the
smoke run, not here.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

# Load the standalone script (lives in scripts/, not the src package). Register it
# in sys.modules BEFORE exec so Python 3.12 ``@dataclass(slots=True)`` can resolve
# ``cls.__module__`` (it walks sys.modules to build __slots__).
_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_metrics.py"
_spec = importlib.util.spec_from_file_location("run_metrics", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
rm = importlib.util.module_from_spec(_spec)
sys.modules["run_metrics"] = rm
_spec.loader.exec_module(rm)


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


# ─── aggregate_cost ───────────────────────────────────────────────────────────


def test_aggregate_cost_totals_and_cache_hit_rate() -> None:
    rows = [
        {"provider": "zai", "model": "glm-5.1", "input_tokens": 1000,
         "output_tokens": 100, "cached_tokens": 600, "total_tokens": 1100,
         "cost_usd": 0.01, "latency_ms": 500},
        {"provider": "zai", "model": "glm-5.1", "input_tokens": 2000,
         "output_tokens": 200, "cached_tokens": 0, "total_tokens": 2200,
         "cost_usd": 0.02, "latency_ms": 700},
        {"provider": "deepseek", "model": "deepseek-v4-flash", "input_tokens": 500,
         "output_tokens": 50, "cached_tokens": 0, "total_tokens": 550,
         "cost_usd": 0.005, "latency_ms": None},
    ]
    c = rm.aggregate_cost(rows)
    assert c.total_calls == 3
    assert c.input_tokens == 3500
    assert c.output_tokens == 350
    assert c.cached_tokens == 600
    assert c.total_cost_usd == pytest.approx(0.035)
    # cache_hit_rate = cached / input = 600/3500
    assert c.cache_hit_rate == pytest.approx(600 / 3500, rel=1e-3)
    assert len(c.by_model) == 2
    glm = next(m for m in c.by_model if m.model == "glm-5.1")
    assert glm.calls == 2
    assert glm.input_tokens == 3000
    assert glm.cost_usd == pytest.approx(0.03)
    assert glm.mean_latency_ms == 600.0  # mean(500,700)
    ds = next(m for m in c.by_model if m.model == "deepseek-v4-flash")
    assert ds.calls == 1
    assert ds.mean_latency_ms is None  # the only latency was None


def test_aggregate_cost_empty() -> None:
    c = rm.aggregate_cost([])
    assert c.total_calls == 0
    assert c.cache_hit_rate == 0.0
    assert c.by_model == []


# ─── score_goals (terminal-state, self-correction not penalized) ─────────────


def test_score_goals_self_correction_scores_terminal_pass() -> None:
    """A check that fails then passes (re-verify) scores its FINAL passing state."""
    rows = [
        {"goal_id": "g1", "run_id": "cli-g1", "attempt_id": "A1",
         "check_name": "c1", "score": 0.0, "created_at": "2026-07-01T00:00:00+00:00"},
        {"goal_id": "g1", "run_id": "cli-g1", "attempt_id": "A1",
         "check_name": "c1", "score": 1.0, "created_at": "2026-07-01T00:01:00+00:00"},
    ]
    s = rm.score_goals(rows)
    assert s.n_goals_ran == 1
    assert s.battery_mean == pytest.approx(1.0)
    assert s.per_goal[0].score == pytest.approx(1.0)  # terminal = the pass
    assert s.per_goal[0].n_checks == 1
    assert s.per_goal[0].n_rows == 2
    assert s.per_goal[0].verify_passes_estimate == 2  # 2 rows / 1 check


def test_score_goals_latest_attempt_wins() -> None:
    """Two attempts: the chronologically-latest attempt_id's rows are scored."""
    rows = [
        {"goal_id": "g1", "run_id": "cli-g1", "attempt_id": "2026-01-01T00:00:00",
         "check_name": "c1", "score": 0.0, "created_at": "2026-01-01T00:00:00+00:00"},
        {"goal_id": "g1", "run_id": "cli-g1", "attempt_id": "2026-01-02T00:00:00",
         "check_name": "c1", "score": 1.0, "created_at": "2026-01-02T00:00:00+00:00"},
    ]
    s = rm.score_goals(rows)
    assert s.battery_mean == pytest.approx(1.0)  # the newer attempt passed
    assert s.per_goal[0].attempt_id == "2026-01-02T00:00:00"


def test_score_goals_battery_mean_excludes_missing_goals() -> None:
    """A goal with no rows is excluded from the battery mean, not counted as 0."""
    rows = [
        {"goal_id": "gA", "run_id": "cli-a", "attempt_id": "A1",
         "check_name": "c1", "score": 1.0, "created_at": "2026-07-01T00:00:00+00:00"},
        {"goal_id": "gB", "run_id": "cli-b", "attempt_id": "A1",
         "check_name": "c1", "score": 0.5, "created_at": "2026-07-01T00:00:00+00:00"},
    ]
    s = rm.score_goals(rows)
    assert s.n_goals_ran == 2
    assert s.battery_mean == pytest.approx(0.75)


# ─── _spec_key_from_run_id (Track-1 adhoc-collapse fix) ──────────────────────


def test_spec_key_from_run_id_strips_prefix_gen_seed_date() -> None:
    """Every suffix convention reduces to the spec id (the goal-bucket key)."""
    assert rm._spec_key_from_run_id("api-battery04_q01-gen0-seed1-20260713") == "battery04_q01"
    assert rm._spec_key_from_run_id("cli-battery04_q02-gen2-seed3-20260713") == "battery04_q02"
    assert rm._spec_key_from_run_id("battery04_q01-gen0-20260712") == "battery04_q01"
    assert rm._spec_key_from_run_id("bench-battery04_q04-20260706") == "battery04_q04"
    # Non-battery run_ids keep a stable per-run bucket (no false merge).
    assert rm._spec_key_from_run_id("cli-g1") == "g1"
    assert rm._spec_key_from_run_id(None) == ""
    assert rm._spec_key_from_run_id("") == ""


def test_score_goals_buckets_adhoc_only_by_run_id_not_goal_id() -> None:
    """Regression (Track-1, 2026-07-13): adhoc-only rows must not collapse.

    Before the fix, ``score_goals`` grouped by the ``goal_id`` column. Adhoc-only
    rows all carry ``goal_id="adhoc-deliverables"`` regardless of which battery
    query produced them, so a full battery of adhoc-only runs collapsed into ONE
    bucket (``n_goals_ran=1``) and the per-goal matrix lost every query. The
    Track-1 G0 seed-1 runs hit exactly this — golden checks were silently skipped
    by the resolver bug, leaving only adhoc rows, so G0 "scored 1.0" on a single
    collapsed bucket. Bucketing by the spec id recovered from ``run_id`` restores
    the correct N-goal matrix whether or not golden fired.
    """
    rows = [
        {"goal_id": "adhoc-deliverables",
         "run_id": "api-battery04_q01-gen0-seed1-20260713", "attempt_id": "A1",
         "check_name": "c1", "score": 1.0, "created_at": "2026-07-13T00:00:00+00:00"},
        {"goal_id": "adhoc-deliverables",
         "run_id": "api-battery04_q02-gen0-seed1-20260713", "attempt_id": "A1",
         "check_name": "c1", "score": 0.5, "created_at": "2026-07-13T00:00:00+00:00"},
        {"goal_id": "adhoc-deliverables",
         "run_id": "api-battery04_q03-gen0-seed1-20260713", "attempt_id": "A1",
         "check_name": "c1", "score": 0.0, "created_at": "2026-07-13T00:00:00+00:00"},
    ]
    s = rm.score_goals(rows)
    assert s.n_goals_ran == 3  # was 1 before the fix — all collapsed into "adhoc"
    assert s.battery_mean == pytest.approx(0.5)  # mean(1.0, 0.5, 0.0)
    assert sorted(g.goal_id for g in s.per_goal) == [
        "battery04_q01",
        "battery04_q02",
        "battery04_q03",
    ]


# ─── verify-pass estimate ────────────────────────────────────────────────────


def test_estimate_verify_passes_ceil() -> None:
    # 3 distinct checks, 7 rows → ceil(7/3) = 3 passes
    rows = [{"check_name": c} for c in ("a", "b", "c", "a", "b", "c", "a")]
    assert rm._estimate_verify_passes(rows) == 3
    assert rm._estimate_verify_passes([]) == 0


# ─── aggregate_subagents ─────────────────────────────────────────────────────


def test_aggregate_subagents_status_counts() -> None:
    rows = [
        {"parent_thread_id": "cli-x", "status": "completed", "tokens_used": 100,
         "cost_usd": 0.1},
        {"parent_thread_id": "cli-x", "status": "failed", "tokens_used": 20,
         "cost_usd": 0.02},
        {"parent_thread_id": "cli-x", "status": "timeout", "tokens_used": 5,
         "cost_usd": 0.01},
    ]
    s = rm.aggregate_subagents(rows)
    assert s.delegated == 3
    assert s.completed == 1
    assert s.failed == 1
    assert s.timeout == 1
    assert s.total_tokens == 125
    assert s.total_cost_usd == pytest.approx(0.13)


# ─── aggregate_tool_health ───────────────────────────────────────────────────


def test_aggregate_tool_health_per_tool_rates() -> None:
    rows = [
        {"tool_name": "code_executor", "success": True, "empty_output": False,
         "latency_ms": 100},
        {"tool_name": "code_executor", "success": False, "empty_output": True,
         "latency_ms": 200},
        {"tool_name": "web_search", "success": True, "empty_output": False,
         "latency_ms": None},
    ]
    health = {r.tool_name: r for r in rm.aggregate_tool_health(rows)}
    assert health["code_executor"].calls == 2
    assert health["code_executor"].success_count == 1
    assert health["code_executor"].success_rate == pytest.approx(0.5)
    assert health["code_executor"].empty_output_count == 1
    assert health["code_executor"].mean_latency_ms == 150.0
    assert health["web_search"].mean_latency_ms is None
    assert health["web_search"].success_rate == 1.0


# ─── count_created ────────────────────────────────────────────────────────────


def test_count_created_filters_on_source_mutation() -> None:
    tool_rows = [
        {"source_mutation_id": "mut-1", "created_at": "2026-07-01T00:00:00+00:00"},
        {"source_mutation_id": None, "created_at": "2026-07-01T00:00:00+00:00"},  # builtin
    ]
    sub_rows = [
        {"source_mutation_id": "mut-2", "created_at": "2026-07-01T00:00:00+00:00"},
    ]
    c = rm.count_created(tool_rows, sub_rows)
    assert c.tools == 1  # only the generated tool
    assert c.subagents == 1
    assert c.window_start == "2026-07-01T00:00:00+00:00"


# ─── llm_span ─────────────────────────────────────────────────────────────────


def test_llm_span_seconds() -> None:
    rows = [
        {"_created_dt": _dt("2026-07-01T00:00:00+00:00")},
        {"_created_dt": _dt("2026-07-01T00:30:00+00:00")},
        {"_created_dt": _dt("2026-07-01T00:10:00+00:00")},
    ]
    assert rm.llm_span(rows) == 1800.0  # 30 min span


def test_llm_span_none_when_under_two_points() -> None:
    assert rm.llm_span([{"_created_dt": _dt("2026-07-01T00:00:00+00:00")}]) is None
    assert rm.llm_span([]) is None


# ─── aggregate_node_timing (Track-1 per-node attribution) ─────────────────────


def test_aggregate_node_timing_per_phase() -> None:
    rows = [
        {"phase": "execute", "duration_ms": 1000, "status": "completed"},
        {"phase": "execute", "duration_ms": 3000, "status": "completed"},
        {"phase": "verify", "duration_ms": 500, "status": "completed"},
        {"phase": "reflect", "duration_ms": 0, "status": "failed"},
    ]
    nt = rm.aggregate_node_timing(rows)
    assert nt is not None
    assert nt.total_ms == 4500
    by = {n.phase: n for n in nt.by_node}
    assert by["execute"].calls == 2
    assert by["execute"].total_ms == 4000
    assert by["execute"].mean_ms == 2000.0
    assert by["verify"].calls == 1
    assert by["reflect"].total_ms == 0


def test_aggregate_node_timing_empty_returns_none() -> None:
    assert rm.aggregate_node_timing([]) is None


# ─── selector validation ─────────────────────────────────────────────────────


def test_validate_selector_rejects_metacharacters() -> None:
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        rm._validate_selector("evil%_")
    # valid forms accepted
    assert rm._validate_selector("20260707") == "20260707"
    assert rm._validate_selector("q01-20260707") == "q01-20260707"


# ─── report_to_dict round-trip ───────────────────────────────────────────────


def test_report_to_dict_is_json_serializable() -> None:
    import json

    cost = rm.aggregate_cost(
        [{"provider": "zai", "model": "glm-5.1", "input_tokens": 10,
          "output_tokens": 1, "cached_tokens": 0, "total_tokens": 11,
          "cost_usd": 0.001, "latency_ms": None}]
    )
    score = rm.score_goals(
        [{"goal_id": "g1", "run_id": "cli-g1", "attempt_id": "A1",
          "check_name": "c1", "score": 1.0, "created_at": "2026-07-01T00:00:00+00:00"}]
    )
    report = rm.RunMetricsReport(
        selector="20260707",
        matched_run_ids=["cli-g1"],
        score=score,
        cost=cost,
        llm_span_seconds=12.0,
        subagents=None,
        created=rm.CreatedSummary(0, 0, None, None),
        global_tool_health=[],
        node_timing=None,
    )
    blob = json.dumps(rm.report_to_dict(report))
    parsed: Any = json.loads(blob)
    assert parsed["selector"] == "20260707"
    assert parsed["cost"]["total_cost_usd"] == pytest.approx(0.001)
    assert parsed["llm_span_seconds"] == 12.0
    assert parsed["node_timing"] is None  # no execution_steps rows for this run
    assert "attribution_gaps" not in parsed  # gaps dropped — all attribution direct
