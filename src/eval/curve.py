"""Battery capability-curve analytics (Phase 2 C1).

Turns ``eval_results`` rows into a per-night battery trend + a regression
verdict — the missing evidence that the agent self-improves over time, plus
the *temporal* regression signal the promotion-gate canary (single-goal, at
promotion time) does not provide. This module is a **pure transform + verdict**:
it fetches via ``EvalStore.fetch_rows`` and writes exports, but performs NO
rollback (that is the gate's job in ``src/evolution/curve_gate.py``).

Regression definition (the conjunction prevents noise from ever firing):
``current < score_floor`` AND ``(best_prior - current) >= regression_delta``
AND ``n_points >= min_points``. ``current`` is the latest night's battery mean;
``best_prior`` is the max battery mean over nights strictly before the latest
(the high-water mark, excluding current so a fresh peak cannot self-trigger).
A delta-only dip that stays above ``score_floor`` is NOT a regression — that
is the noise guard. Too few nights is ``inconclusive`` (never a regression).

All defaults come from ``CapabilityCurveSettings`` (env ``CAPABILITY_CURVE_*``);
the constructor accepts an injectable ``EvalStore`` so the trend logic is
unit-tested deterministically against a list-backed fake store.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from src.config.settings import CapabilityCurveSettings, get_settings
from src.eval.store import EvalStore


@dataclass(frozen=True, slots=True)
class CurvePoint:
    """One goal's score for one date — the latest attempt's TERMINAL verify state.

    "Terminal state" = the most-recent row per check within the latest attempt
    (not the mean of every verify pass), so a self-correcting run scores its
    final outcome rather than its recovery journey.
    """

    date: date
    attempt_id: str | None
    mean_score: float
    n_checks: int


@dataclass(frozen=True, slots=True)
class BatteryPoint:
    """One night's battery mean — the mean of each goal's latest-attempt mean that night."""

    date: date
    mean_score: float
    n_goals: int


def _mean(values: list[float]) -> float:
    """Arithmetic mean (0.0 for an empty list — a night with no rows)."""
    return sum(values) / len(values) if values else 0.0


def _row_date(row: dict[str, Any]) -> date | None:
    """Parse a row's ``created_at`` ISO string to a date (None if absent/unparseable)."""
    raw = row.get("created_at")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw)).date()
    except ValueError:
        return None


def _row_created_key(row: dict[str, Any]) -> str:
    """Chronological sort key — the ``created_at`` ISO string (``''`` sorts oldest).

    ISO-8601 timestamps are lexicographically chronological (fixed-width,
    zero-padded), so ``max`` over this key selects the newest row without a
    datetime parse. Absent ``created_at`` sorts oldest so it never wins a tie
    against a real timestamp.
    """
    return str(row.get("created_at") or "")


def expected_check_names(spec_id: str | None) -> set[str]:
    """The declared ``CheckConfig.name`` set for a goal spec (empty if unregistered).

    A goal's verify battery is *complete* (the run converged on evaluation) iff
    the terminal attempt observed every check name the spec declares. Skipped
    checks are still written with their ``check_name`` (``EvalStore._store_check``
    persists every ``CheckResult``), so a full verify pass reaches coverage 1.0.
    Used by the convergence filter in ``scripts/run_metrics.py`` so a
    budget-exhausted run that never reached verify (q06 G2: no deliverable → no
    checks) is distinguished from a converged-but-low run (q06 G0: all checks
    ran, 3/9 passed). Returns ``set()`` for adhoc/custom goals with no spec so
    the caller can fall back to "unknown → do not filter".
    """
    if not spec_id:
        return set()
    from src.eval.golden import GOLDEN_SPECS  # noqa: PLC0415 — lazy; golden is a data module

    spec = GOLDEN_SPECS.get(spec_id)
    if spec is None:
        return set()
    return {c.name for c in spec.checks}


def coverage_ratio(observed: set[str], expected: set[str]) -> float:
    """Fraction of ``expected`` check names present in ``observed`` (1.0 = full).

    ``0.0`` when ``expected`` is empty is intentional — the ratio is undefined
    for a goal with no declared checks (adhoc), and callers treat the empty
    case as "unknown" (``expected_check_names`` returned ``set()``) rather than
    "fully covered", so they never silently pass an unmeasurable goal through a
    convergence gate. Pure; unit-tested directly.
    """
    if not expected:
        return 0.0
    return len(observed & expected) / len(expected)


class CapabilityCurve:
    """Battery correctness-over-time analytics + regression verdict."""

    def __init__(
        self,
        store: EvalStore | None = None,
        settings: CapabilityCurveSettings | None = None,
    ) -> None:
        self._store = store or EvalStore()
        s = settings or get_settings().capability_curve
        self._delta: float = s.regression_delta
        self._floor: float = s.score_floor
        self._min_points: int = s.min_points

    async def per_goal_trend(
        self,
        goal_id: str,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        model: str | None = None,
    ) -> list[CurvePoint]:
        """Latest-attempt-per-date mean score for one goal (ascending by date).

        Groups the goal's rows by ``created_at.date()``; within each date keeps
        only the rows of the chronologically-latest ``attempt_id`` (lexicographic
        max — attempt ids are timestamp-prefixed so this == newest) and means
        their ``score``. A re-run of the same date therefore replaces, not
        blends, the earlier attempt's score.

        ``model`` slices the trend to one producer model (``producer_model = m``)
        so the curve can show a per-model self-improvement trend rather than the
        blended system-wide one; ``None`` (default) spans all producers.
        """
        rows = await self._store.fetch_rows(
            [goal_id], since=since, until=until, producer_model=model
        )
        by_date: dict[date, list[dict[str, Any]]] = {}
        for row in rows:
            d = _row_date(row)
            if d is None:
                continue
            by_date.setdefault(d, []).append(row)

        points: list[CurvePoint] = []
        for d in sorted(by_date):
            day_rows = by_date[d]
            # The latest attempt that date = the max attempt_id present. Rows
            # arrive newest-first from fetch_rows, but group defensively by the
            # explicit attempt_id so the verdict is independent of row order.
            attempt = self._latest_attempt(day_rows)
            attempt_rows = [r for r in day_rows if r.get("attempt_id") == attempt] or day_rows
            # Score the attempt by its TERMINAL verify state, not the recovery
            # journey. ``attempt_id`` is run-scoped, so one attempt spans every
            # verify pass; averaging ALL passes would penalize the self-correction
            # the recomputation eval drives (a run that fails then fixes scores
            # <1.0 despite ending all-pass). Keep the most-recent row per check,
            # then mean across checks.
            latest = self._latest_per_check(attempt_rows)
            points.append(
                CurvePoint(
                    date=d,
                    attempt_id=attempt,
                    mean_score=_mean([float(r.get("score") or 0.0) for r in latest]),
                    n_checks=len(latest),
                )
            )
        return points

    @staticmethod
    def _latest_attempt(rows: list[dict[str, Any]]) -> str | None:
        """Max ``attempt_id`` among rows (chronologically latest); None when none tagged."""
        ids: list[str] = [str(r["attempt_id"]) for r in rows if r.get("attempt_id")]
        return max(ids) if ids else None

    @staticmethod
    def _latest_per_check(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Most-recent row per ``check_name`` — the attempt's terminal verify state.

        Within one ``attempt_id`` the verify node runs repeatedly; each pass
        writes a fresh row for every check. Averaging ALL passes penalizes
        self-correction (a run that fails then fixes scores <1.0 despite ending
        all-pass). Taking the latest row per ``check_name`` scores the run by its
        final state: a check that ultimately passes counts as passed; one the run
        abandoned still failing counts as failed. Selection is by ``created_at``
        (max wins), so input order is irrelevant. Returns one row per distinct
        check name.
        """
        best: dict[str, dict[str, Any]] = {}
        for r in rows:
            name = str(r.get("check_name") or "")
            prev = best.get(name)
            if prev is None or _row_created_key(r) > _row_created_key(prev):
                best[name] = r
        return list(best.values())

    async def battery_trend(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        model: str | None = None,
    ) -> list[BatteryPoint]:
        """Per-night battery mean across the 9 ``BATTERY04_GOALS`` (ascending by date).

        For each date, the battery mean is the mean of each goal's latest-attempt
        mean that ran that night (only goals that ran count; ``n_goals`` records
        how many). Unioning the per-goal trends means a partial night (fewer than
        9 goals) still yields a point rather than a gap.

        ``model`` restricts the trend to a single producer model's rows so the
        curve can be read as a model-specific self-improvement trend.
        """
        from src.eval.golden import BATTERY04_GOALS  # noqa: PLC0415 — lazy; golden is a data module

        per_goal: dict[date, list[float]] = {}
        for spec in BATTERY04_GOALS:
            for point in await self.per_goal_trend(
                spec.spec_id, since=since, until=until, model=model
            ):
                per_goal.setdefault(point.date, []).append(point.mean_score)

        return [
            BatteryPoint(date=d, mean_score=_mean(per_goal[d]), n_goals=len(per_goal[d]))
            for d in sorted(per_goal)
        ]

    @staticmethod
    def best_so_far(points: list[BatteryPoint]) -> BatteryPoint | None:
        """High-water mark EXCLUDING the latest date (None when <2 points)."""
        if len(points) < 2:
            return None
        prior = points[:-1]
        return max(prior, key=lambda p: p.mean_score)

    async def detect_regression(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Apply the grounded regression definition to the battery trend.

        Returns a verdict dict: ``regressed`` (bool), ``inconclusive`` (bool — too
        few nights), ``current`` / ``best_prior`` (floats or None), ``delta``
        (best_prior - current), ``n_points``, plus the configured ``floor`` /
        ``delta_floor`` (the regression_delta threshold) for observability.
        ``inconclusive`` and ``regressed`` are mutually exclusive.

        ``model`` scopes the verdict to one producer model's trend.
        """
        points = await self.battery_trend(since=since, until=until, model=model)
        n = len(points)
        if n < self._min_points or n < 2:
            return {
                "regressed": False,
                "inconclusive": True,
                "current": points[-1].mean_score if points else None,
                "best_prior": None,
                "delta": None,
                "n_points": n,
                "floor": self._floor,
                "delta_floor": self._delta,
            }

        current = points[-1].mean_score
        best_prior = self.best_so_far(points)
        prior_score = best_prior.mean_score if best_prior else current
        delta = round(prior_score - current, 4)
        regressed = (current < self._floor) and (delta >= self._delta)
        return {
            "regressed": regressed,
            "inconclusive": False,
            "current": round(current, 4),
            "best_prior": round(prior_score, 4),
            "delta": delta,
            "n_points": n,
            "floor": self._floor,
            "delta_floor": self._delta,
        }

    async def snapshot(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """One-fetch inspection bundle for the CLI: battery trend + per-goal latest + verdict.

        ``model`` scopes every sub-fetch to one producer model so the bundle is a
        single model's capability view (the CLI ``--model`` flag).
        """
        from src.eval.golden import BATTERY04_GOALS  # noqa: PLC0415

        verdict = await self.detect_regression(since=since, until=until, model=model)
        battery = await self.battery_trend(since=since, until=until, model=model)
        latest_per_goal: list[dict[str, Any]] = []
        for spec in BATTERY04_GOALS:
            trend = await self.per_goal_trend(
                spec.spec_id, since=since, until=until, model=model
            )
            last = trend[-1] if trend else None
            latest_per_goal.append(
                {
                    "goal_id": spec.spec_id,
                    "date": last.date.isoformat() if last else None,
                    "mean_score": round(last.mean_score, 4) if last else None,
                    "n_checks": last.n_checks if last else 0,
                }
            )
        return {
            "battery_trend": [_battery_to_json(p) for p in battery],
            "latest_per_goal": latest_per_goal,
            "verdict": verdict,
            "producer_model": model,
        }

    def export_json(self, path: str | Path, snapshot: dict[str, Any]) -> None:
        """Write a snapshot to ``path`` as JSON (date-aware)."""
        Path(path).write_text(
            json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Capability-curve JSON exported → {}", path)

    def export_csv(self, path: str | Path, snapshot: dict[str, Any]) -> None:
        """Write the battery trend rows to ``path`` as CSV (date, mean_score, n_goals)."""
        rows = snapshot.get("battery_trend") or []
        with Path(path).open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["date", "mean_score", "n_goals"])
            for row in rows:
                writer.writerow([row.get("date"), row.get("mean_score"), row.get("n_goals")])
        logger.info("Capability-curve CSV exported → {}", path)

    def plot_png(self, path: str | Path, snapshot: dict[str, Any]) -> bool:
        """Render the battery trend to a PNG. Returns False (skip+log) if matplotlib absent.

        matplotlib is import-guarded so it needs no allowlist entry; the
        JSON/CSV/table surfaces are the primary exports.
        """
        try:
            import matplotlib  # noqa: PLC0415 — optional dep, guarded
            matplotlib.use("Agg")  # headless backend (no display)
            import matplotlib.pyplot as plt  # noqa: PLC0415
        except ImportError:
            logger.info("matplotlib unavailable; skipping PNG export (use --export .json/.csv)")
            return False

        rows = snapshot.get("battery_trend") or []
        labels = [r.get("date", "") for r in rows]
        scores = [r.get("mean_score", 0.0) for r in rows]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(range(len(scores)), scores, marker="o", linewidth=2)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("battery mean correctness")
        model = snapshot.get("producer_model")
        title = (
            f"Capability curve — {model}" if model else "Capability curve — nightly battery mean"
        )
        ax.set_title(title)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(Path(path))
        plt.close(fig)
        logger.info("Capability-curve PNG exported → {}", path)
        return True


def _battery_to_json(point: BatteryPoint) -> dict[str, Any]:
    d = asdict(point)
    d["date"] = point.date.isoformat()
    return d
