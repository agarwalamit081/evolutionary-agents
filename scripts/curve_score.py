#!/usr/bin/env python3
"""Suffix-based capability-curve scorer (scoring-fix re-baseline + C1 A/B).

The nightly ``--capability-curve`` groups ``eval_results`` by ``created_at``
DATE, which a battery run spanning midnight splits across two dates. Curves are
identified by their run_id SUFFIX (``-YYYYMMDD``), so this scorer groups by
suffix instead — the discriminator the historical baselines actually used.

For each of the 9 ``BATTERY04_GOALS`` it takes the latest ``attempt_id`` among
all runs whose run_id ends in ``-<suffix>``, scores its **terminal verify
state** (latest row per check, via ``CapabilityCurve._latest_per_check``), and
means across checks. The battery mean is the mean of the per-goal means of the
goals that ran (a goal with no rows is excluded, not counted as 0 — mirroring
``CapabilityCurve.battery_trend``). This is the post-fix Method A:
self-correction is no longer penalized.

Read-only (no writes). Connects via the app's ``DATABASE_URL``.

Usage::

    python scripts/curve_score.py                       # every suffix in eval_results
    python scripts/curve_score.py 20260701              # one suffix
    python scripts/curve_score.py 20260701 20260705     # compare two (e.g. an A/B)
"""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path
from typing import Any

# Project root on sys.path so ``src.*`` imports resolve when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from src.db.models import EvalResult  # noqa: E402
from src.db.session import get_session  # noqa: E402
from src.eval.curve import CapabilityCurve  # noqa: E402
from src.eval.golden import BATTERY04_GOALS  # noqa: E402

_SUFFIX_RE = re.compile(r"-(\d{8})$")


async def _rows_for(suffix: str, goal_id: str) -> list[dict[str, Any]]:
    """All check rows for one goal whose run_id ends in ``-<suffix>`` (newest-first)."""
    like = f"%-{suffix}"
    async with get_session() as session:
        rows = (
            await session.execute(
                select(EvalResult)
                .where(EvalResult.goal_id == goal_id, EvalResult.run_id.like(like))
                .order_by(EvalResult.created_at.desc())
            )
        ).scalars().all()
        return [
            {
                "goal_id": r.goal_id,
                "run_id": r.run_id,
                "attempt_id": r.attempt_id,
                "check_name": r.check_name,
                "score": float(r.score),
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]


def _goal_score(rows: list[dict[str, Any]]) -> tuple[float, int, str | None]:
    """Latest attempt → terminal verify state (latest row per check) → mean.

    Mirrors ``CapabilityCurve.per_goal_trend`` exactly, minus the date grouping:
    the latest ``attempt_id`` wins, then ``_latest_per_check`` collapses the
    repeated verify passes to the terminal state per check. Returns
    ``(mean_score, n_checks, attempt_id)`` or ``(0.0, 0, None)`` when no rows.
    """
    if not rows:
        return 0.0, 0, None
    attempt = CapabilityCurve._latest_attempt(rows)
    attempt_rows = [r for r in rows if r.get("attempt_id") == attempt] or rows
    latest = CapabilityCurve._latest_per_check(attempt_rows)
    if not latest:
        return 0.0, 0, attempt
    mean = sum(float(r.get("score") or 0.0) for r in latest) / len(latest)
    return mean, len(latest), attempt


async def score_suffix(suffix: str) -> dict[str, Any]:
    """Score one curve (suffix): per-goal terminal-state means + battery mean."""
    detail: list[dict[str, Any]] = []
    ran_means: list[float] = []
    for spec in BATTERY04_GOALS:
        rows = await _rows_for(suffix, spec.spec_id)
        mean, n_checks, attempt = _goal_score(rows)
        if rows:
            ran_means.append(mean)
        detail.append(
            {
                "goal": spec.spec_id,
                "mean": round(mean, 4),
                "n_checks": n_checks,
                "ran": bool(rows),
                "attempt": attempt,
            }
        )
    battery = sum(ran_means) / len(ran_means) if ran_means else 0.0
    return {
        "suffix": suffix,
        "battery_mean": round(battery, 4),
        "n_goals_ran": len(ran_means),
        "detail": detail,
    }


async def _discover_suffixes() -> list[str]:
    """Distinct ``-YYYYMMDD`` suffixes present in eval_results (ascending)."""
    async with get_session() as session:
        rows = (
            await session.execute(select(EvalResult.run_id).where(EvalResult.run_id.is_not(None)))
        ).scalars().all()
    found: set[str] = set()
    for rid in rows:
        m = _SUFFIX_RE.search(str(rid))
        if m:
            found.add(m.group(1))
    return sorted(found)


def _print(result: dict[str, Any]) -> None:
    print(f"\n=== curve suffix {result['suffix']} ===")
    print(f"battery_mean = {result['battery_mean']:.4f}  (n_goals_ran={result['n_goals_ran']})")
    print(f"{'goal':<20} {'mean':>7} {'checks':>6}  ran  attempt")
    for d in result["detail"]:
        print(
            f"{d['goal']:<20} {d['mean']:>7.4f} {d['n_checks']:>6}  "
            f"{'yes' if d['ran'] else 'no':<3}  {d['attempt'] or '-'}"
        )


async def main(argv: list[str]) -> int:
    suffixes = argv if argv else await _discover_suffixes()
    if not suffixes:
        print("no battery suffixes found in eval_results", file=sys.stderr)
        return 1
    print(f"Scoring {len(suffixes)} curve suffix(es) under post-fix Method A "
          f"(terminal verify state per check): {', '.join(suffixes)}")
    summaries: list[tuple[str, float, int]] = []
    for suffix in suffixes:
        result = await score_suffix(suffix)
        _print(result)
        summaries.append((result["suffix"], result["battery_mean"], result["n_goals_ran"]))
    print("\n=== summary (battery mean per suffix) ===")
    for suffix, mean, n in summaries:
        print(f"  {suffix}: {mean:.4f}  (n_goals={n})")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
