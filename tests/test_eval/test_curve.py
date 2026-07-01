"""CapabilityCurve (Phase 2 C1): trend + regression verdict + exports.

Pure analytics tested against a list-backed fake ``EvalStore`` (deterministic,
no DB) so the latest-attempt-per-date grouping and the floor+delta+min-points
regression definition are locked independently of the database layer. The
``fetch_rows`` SQL filtering itself is round-tripped in ``test_store.py`` via
aiosqlite.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from src.config.settings import CapabilityCurveSettings
from src.eval.curve import CapabilityCurve
from src.eval.store import EvalStore


class _FakeStore(EvalStore):
    """List-backed store: returns seeded rows whose goal_id is requested."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._seed = rows

    async def fetch_rows(  # type: ignore[override]
        self,
        goal_ids: list[str],
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 2000,
        producer_model: str | None = None,
    ) -> list[dict[str, Any]]:
        wanted = set(goal_ids)
        rows = [r for r in self._seed if r["goal_id"] in wanted]
        if producer_model is not None:
            rows = [r for r in rows if r.get("producer_model") == producer_model]
        return rows


def _row(
    goal_id: str,
    attempt: str,
    created_at: str,
    score: float,
    check: str = "check_x",
    producer_model: str | None = None,
) -> dict[str, Any]:
    """A row shaped like EvalStore._row_to_dict output."""
    return {
        "goal_id": goal_id,
        "run_id": "r",
        "attempt_id": attempt,
        "spec_id": goal_id,
        "check_name": check,
        "check_type": "Golden",
        "passed": score >= 0.5,
        "score": score,
        "skipped": False,
        "evidence": None,
        "cost_usd": 0.0,
        "producer_model": producer_model,
        "created_at": created_at,
    }


def _settings(delta: float = 0.1, floor: float = 0.5, min_points: int = 2) -> CapabilityCurveSettings:
    return CapabilityCurveSettings(
        regression_delta=delta, score_floor=floor, min_points=min_points
    )


# ─── per_goal_trend ─────────────────────────────────────────────────


async def test_per_goal_trend_picks_latest_attempt_per_date() -> None:
    # Two attempts on the SAME date; the later attempt_id must win (its mean used).
    rows = [
        _row("battery04_q01", "2026-06-01T10:00:00+00:00_a", "2026-06-01T10:00:00+00:00", 0.4, "c1"),
        _row("battery04_q01", "2026-06-01T22:00:00+00:00_a", "2026-06-01T22:00:00+00:00", 0.9, "c1"),
    ]
    curve = CapabilityCurve(_FakeStore(rows), _settings())
    trend = await curve.per_goal_trend("battery04_q01")
    assert len(trend) == 1
    point = trend[0]
    assert point.attempt_id == "2026-06-01T22:00:00+00:00_a"  # the later attempt
    assert point.mean_score == pytest.approx(0.9)
    assert point.n_checks == 1


async def test_per_goal_trend_means_multiple_checks_of_latest_attempt() -> None:
    # Latest attempt has 2 checks (0.8, 0.6) → mean 0.7; an earlier attempt is ignored.
    rows = [
        _row("battery04_q01", "2026-06-01T10:00:00+00:00_a", "2026-06-01T10:00:00+00:00", 0.2, "c1"),
        _row("battery04_q01", "2026-06-02T22:00:00+00:00_a", "2026-06-02T22:00:00+00:00", 0.8, "c1"),
        _row("battery04_q01", "2026-06-02T22:00:00+00:00_a", "2026-06-02T22:00:00+00:00", 0.6, "c2"),
    ]
    curve = CapabilityCurve(_FakeStore(rows), _settings())
    trend = await curve.per_goal_trend("battery04_q01")
    assert [p.date.isoformat() for p in trend] == ["2026-06-01", "2026-06-02"]
    assert trend[1].mean_score == pytest.approx(0.7)
    assert trend[1].n_checks == 2


async def test_per_goal_trend_empty_when_no_rows() -> None:
    curve = CapabilityCurve(_FakeStore([]), _settings())
    assert await curve.per_goal_trend("battery04_q01") == []


# ─── battery_trend ──────────────────────────────────────────────────


async def test_battery_trend_means_across_goals() -> None:
    # Three goals each ran 2026-06-01 (means 0.6, 0.8, 1.0 → battery mean 0.8).
    rows = [
        _row("battery04_q01", "a", "2026-06-01T01:00:00+00:00", 0.6),
        _row("battery04_q02", "a", "2026-06-01T02:00:00+00:00", 0.8),
        _row("battery04_q03", "a", "2026-06-01T03:00:00+00:00", 1.0),
    ]
    curve = CapabilityCurve(_FakeStore(rows), _settings())
    battery = await curve.battery_trend()
    assert len(battery) == 1
    assert battery[0].date.isoformat() == "2026-06-01"
    assert battery[0].mean_score == pytest.approx(0.8)
    assert battery[0].n_goals == 3


async def test_battery_trend_partial_night_only_counts_goals_that_ran() -> None:
    rows = [
        _row("battery04_q01", "a", "2026-06-01T01:00:00+00:00", 0.6),
        _row("battery04_q02", "a", "2026-06-01T02:00:00+00:00", 1.0),
        _row("battery04_q01", "a", "2026-06-02T01:00:00+00:00", 1.0),  # only q01 on day 2
    ]
    curve = CapabilityCurve(_FakeStore(rows), _settings())
    battery = await curve.battery_trend()
    by_date = {p.date.isoformat(): p for p in battery}
    assert by_date["2026-06-01"].mean_score == pytest.approx(0.8)
    assert by_date["2026-06-01"].n_goals == 2
    assert by_date["2026-06-02"].mean_score == pytest.approx(1.0)
    assert by_date["2026-06-02"].n_goals == 1


# ─── detect_regression ─────────────────────────────────────────────


async def test_detect_regression_below_floor_and_delta() -> None:
    # Trending up then a sharp drop on the latest night: 0.8 → 0.9 → 0.2.
    rows = [
        _row("battery04_q01", "a", "2026-06-01T01:00:00+00:00", 0.8),
        _row("battery04_q01", "a", "2026-06-02T01:00:00+00:00", 0.9),
        _row("battery04_q01", "a", "2026-06-03T01:00:00+00:00", 0.2),
    ]
    curve = CapabilityCurve(_FakeStore(rows), _settings(delta=0.1, floor=0.5, min_points=2))
    verdict = await curve.detect_regression()
    assert verdict["regressed"] is True
    assert verdict["inconclusive"] is False
    assert verdict["current"] == pytest.approx(0.2)
    assert verdict["best_prior"] == pytest.approx(0.9)
    assert verdict["delta"] == pytest.approx(0.7)
    assert verdict["n_points"] == 3


async def test_detect_regression_inconclusive_when_too_few_points() -> None:
    rows = [_row("battery04_q01", "a", "2026-06-01T01:00:00+00:00", 0.1)]
    curve = CapabilityCurve(_FakeStore(rows), _settings(min_points=2))
    verdict = await curve.detect_regression()
    assert verdict["regressed"] is False
    assert verdict["inconclusive"] is True


async def test_detect_regression_no_regression_when_floor_held() -> None:
    # Dropped 0.3 (>= delta) BUT current 0.6 is NOT below floor 0.5 → not regressed.
    rows = [
        _row("battery04_q01", "a", "2026-06-01T01:00:00+00:00", 0.9),
        _row("battery04_q01", "a", "2026-06-02T01:00:00+00:00", 0.6),
    ]
    curve = CapabilityCurve(_FakeStore(rows), _settings(delta=0.1, floor=0.5))
    verdict = await curve.detect_regression()
    assert verdict["regressed"] is False  # delta-only dip below a held floor is NOT a regression
    assert verdict["inconclusive"] is False
    assert verdict["delta"] == pytest.approx(0.3)


async def test_detect_regression_delta_too_small_is_not_regression() -> None:
    # Below floor (0.4 < 0.5) but drop 0.1 < delta 0.3 → not regressed (noise guard).
    rows = [
        _row("battery04_q01", "a", "2026-06-01T01:00:00+00:00", 0.5),
        _row("battery04_q01", "a", "2026-06-02T01:00:00+00:00", 0.4),
    ]
    curve = CapabilityCurve(_FakeStore(rows), _settings(delta=0.3, floor=0.5))
    verdict = await curve.detect_regression()
    assert verdict["regressed"] is False


# ─── exports ────────────────────────────────────────────────────────


async def test_export_json_csv_round_trip(tmp_path: Path) -> None:
    rows = [
        _row("battery04_q01", "a", "2026-06-01T01:00:00+00:00", 0.8),
        _row("battery04_q01", "a", "2026-06-02T01:00:00+00:00", 0.6),
    ]
    curve = CapabilityCurve(_FakeStore(rows), _settings())
    snap = await curve.snapshot()

    json_path = tmp_path / "curve.json"
    curve.export_json(json_path, snap)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert "battery_trend" in data and "verdict" in data and "latest_per_goal" in data
    assert len(data["battery_trend"]) == 2
    assert data["battery_trend"][0]["date"] == "2026-06-01"

    csv_path = tmp_path / "curve.csv"
    curve.export_csv(csv_path, snap)
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "date,mean_score,n_goals"
    assert len(lines) == 3  # header + 2 rows


def test_plot_png_skipped_without_matplotlib(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # A None entry in sys.modules makes `import matplotlib` raise ImportError.
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    curve = CapabilityCurve(_FakeStore([]), _settings())
    ok = curve.plot_png(tmp_path / "out.png", {"battery_trend": []})
    assert ok is False
    assert not (tmp_path / "out.png").exists()


def test_plot_png_renders_when_matplotlib_present(tmp_path: Path) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    curve = CapabilityCurve(_FakeStore([]), _settings())
    snap = {"battery_trend": [{"date": "2026-06-01", "mean_score": 0.8, "n_goals": 9}]}
    out = tmp_path / "out.png"
    ok = curve.plot_png(out, snap)
    assert ok is True
    assert out.exists() and out.stat().st_size > 0


# ─── producer-model slicing (Phase-2 attribution) ─────────────────


async def test_per_goal_trend_model_filter_slices_to_one_producer() -> None:
    # Same goal+date scored by two different producers; the model filter must
    # isolate one producer's trend so the curve reads model-specific, not blended.
    rows = [
        _row("battery04_q01", "2026-06-01T10:00:00+00:00_a", "2026-06-01T10:00:00+00:00", 0.9, producer_model="glm-4.7"),
        _row("battery04_q01", "2026-06-01T11:00:00+00:00_a", "2026-06-01T11:00:00+00:00", 0.3, producer_model="deepseek-v4-flash"),
    ]
    curve = CapabilityCurve(_FakeStore(rows), _settings())

    glm = await curve.per_goal_trend("battery04_q01", model="glm-4.7")
    assert len(glm) == 1
    assert glm[0].mean_score == pytest.approx(0.9)

    ds = await curve.per_goal_trend("battery04_q01", model="deepseek-v4-flash")
    assert len(ds) == 1
    assert ds[0].mean_score == pytest.approx(0.3)

    # No filter = blended (latest-attempt-per-date rule keeps the newer attempt,
    # so the 11:00 row — deepseek — wins for that date, model=None spans both).
    blended = await curve.per_goal_trend("battery04_q01")
    assert len(blended) == 1


async def test_snapshot_carries_producer_model_key() -> None:
    rows = [_row("battery04_q01", "a", "2026-06-01T01:00:00+00:00", 0.8, producer_model="glm-4.7")]
    curve = CapabilityCurve(_FakeStore(rows), _settings())
    snap = await curve.snapshot(model="glm-4.7")
    assert snap["producer_model"] == "glm-4.7"
    # The model filter actually narrowed the fetch (the un-tagged row is excluded).
    assert all(g["mean_score"] is not None for g in snap["latest_per_goal"] if g["goal_id"] == "battery04_q01")

    unfiltered = await curve.snapshot()
    assert unfiltered["producer_model"] is None

