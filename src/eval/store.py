"""Persistent eval-result store (Phase 3).

Writes one ``eval_results`` row per (run, goal, check) and queries them back by
goal or run for regression tracking and the Phase-8 evolution canary. Like the
cost ledger, this is **observability-only**: a DB hiccup here is logged at
WARNING and never re-raises — a poisoned write can never abort a run or a verify
cycle. Writes are gated behind ``EvalSettings.eval_store_enabled``.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import EvalResult

if TYPE_CHECKING:
    from src.eval.models import CheckResult, CorrectnessResult


async def _store_check(
    session: AsyncSession,
    *,
    goal_id: str,
    run_id: str,
    attempt_id: str | None,
    spec_id: str | None,
    check: CheckResult,
    cost_usd: float,
    producer_model: str | None = None,
) -> None:
    session.add(
        EvalResult(
            goal_id=goal_id,
            run_id=run_id,
            attempt_id=attempt_id,
            spec_id=spec_id,
            check_name=check.check_name,
            check_type=check.check_type,
            passed=check.passed,
            score=float(check.score),
            skipped=check.skipped,
            evidence=dict(check.evidence) or None,
            cost_usd=cost_usd,
            producer_model=producer_model,
        )
    )


class EvalStore:
    """Durable projection of correctness results into ``eval_results``.

    Every method is non-fatal: a DB failure is logged and swallowed so eval
    persistence can never break a run. Construct once per process; it opens a
    fresh session per write so a poisoned session recovers on the next call.
    """

    async def record_correctness(
        self,
        correctness: CorrectnessResult,
        *,
        goal_id: str,
        run_id: str,
        attempt_id: str | None = None,
        cost_usd: float = 0.0,
        producer_model: str | None = None,
    ) -> int:
        """Persist every check in ``correctness``; return rows written.

        ``attempt_id`` tags the rows with THIS invocation so a re-run of the
        same ``--run_id`` (which shares ``thread_id``/``run_id``) does not blend
        attempts; ``query_latest_attempt(run_id)`` returns only the newest one.
        ``producer_model`` tags the rows with the model id that produced the goal
        so the capability curve can be sliced per-model (``curve --model``);
        ``None`` leaves the column NULL (legacy/unattributed rows).

        Returns 0 (and logs) if the store is disabled or the write fails.
        """
        from src.config.settings import get_settings

        if not get_settings().eval.eval_store_enabled:
            return 0
        if not correctness.checks:
            return 0

        from src.db.session import get_session

        written = 0
        try:
            async with get_session() as session:
                for check in correctness.checks:
                    await _store_check(
                        session,
                        goal_id=goal_id,
                        run_id=run_id,
                        attempt_id=attempt_id,
                        spec_id=correctness.spec_id or None,
                        check=check,
                        cost_usd=cost_usd,
                        producer_model=producer_model,
                    )
                    written += 1
        except Exception as exc:  # noqa: BLE001 — observability-only, never re-raise
            logger.warning(
                "EvalStore write failed for run={} goal={} attempt={} "
                "({} checks): {}",
                run_id,
                goal_id,
                attempt_id,
                len(correctness.checks),
                exc,
            )
            return 0
        logger.debug(
            "EvalStore wrote {} check rows for run={} goal={} attempt={} model={}",
            written,
            run_id,
            goal_id,
            attempt_id,
            producer_model,
        )
        return written

    async def query_by_run(self, run_id: str) -> list[dict[str, Any]]:
        """Return all eval rows for a run as plain dicts (empty on failure)."""
        from src.db.session import get_session

        try:
            async with get_session() as session:
                rows = (
                    await session.execute(
                        select(EvalResult)
                        .where(EvalResult.run_id == run_id)
                        .order_by(EvalResult.created_at)
                    )
                ).scalars().all()
                return [_row_to_dict(r) for r in rows]
        except Exception as exc:  # noqa: BLE001
            logger.warning("EvalStore query_by_run failed for run={}: {}", run_id, exc)
            return []

    async def query_by_goal(self, goal_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """Return recent eval rows for a goal (empty on failure)."""
        from src.db.session import get_session

        try:
            async with get_session() as session:
                rows = (
                    await session.execute(
                        select(EvalResult)
                        .where(EvalResult.goal_id == goal_id)
                        .order_by(EvalResult.created_at.desc())
                        .limit(limit)
                    )
                ).scalars().all()
                return [_row_to_dict(r) for r in rows]
        except Exception as exc:  # noqa: BLE001
            logger.warning("EvalStore query_by_goal failed for goal={}: {}", goal_id, exc)
            return []

    async def query_latest_attempt(self, run_id: str) -> list[dict[str, Any]]:
        """Return the check rows of the MOST RECENT attempt of a run.

        A run's ``run_id`` (graph ``thread_id``) is stable across re-runs, so
        ``query_by_run`` blends every attempt. This resolves the newest
        ``attempt_id`` (the one whose rows have the max ``created_at``) and
        returns only its rows — so a score means ONE attempt, not a blend.
        Returns ``[]`` when no rows carry an ``attempt_id`` (e.g. only legacy
        rows exist) — callers may then fall back to ``query_by_run``.
        """
        from src.db.session import get_session

        try:
            async with get_session() as session:
                newest = (
                    await session.execute(
                        select(EvalResult.attempt_id)
                        .where(
                            EvalResult.run_id == run_id,
                            EvalResult.attempt_id.is_not(None),
                        )
                        .order_by(EvalResult.created_at.desc())
                        .limit(1)
                    )
                ).scalar_one_or_none()
                if newest is None:
                    return []
                rows = (
                    await session.execute(
                        select(EvalResult)
                        .where(
                            EvalResult.run_id == run_id,
                            EvalResult.attempt_id == newest,
                        )
                        .order_by(EvalResult.created_at, EvalResult.check_name)
                    )
                ).scalars().all()
                return [_row_to_dict(r) for r in rows]
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "EvalStore query_latest_attempt failed for run={}: {}", run_id, exc
            )
            return []

    async def fetch_rows(
        self,
        goal_ids: list[str],
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 2000,
        producer_model: str | None = None,
    ) -> list[dict[str, Any]]:
        """Goal-scoped rows in a ``created_at`` window, newest-first (empty on failure).

        A single ``goal_id IN (...)`` fetch for the capability-curve analytics
        layer (``src/eval/curve.py``): it pulls every check row for the given
        goals within an optional [since, until] window, ordered newest-first so
        the curve's "latest attempt per date" grouping is cheap. The curve does
        the aggregation (mean per date / per goal); this is the thin fetcher.

        Args:
            goal_ids: Goal ids to include (e.g. the 9 ``BATTERY04_GOALS`` spec ids).
            since: Optional inclusive lower bound on ``created_at``.
            until: Optional inclusive upper bound on ``created_at``.
            limit: Row cap (default 2000; the 9-spec battery writes ~tens/night).
            producer_model: Optional model-id filter (``WHERE producer_model = :m``)
                so ``curve --model`` slices a single model's trend instead of the
                blended system-wide one. ``None`` returns rows for all producers
                (including legacy NULL-attributed rows).

        Returns:
            List of ``_row_to_dict`` rows (``created_at`` as an ISO string), or
            ``[]`` if the query fails (observability-only, never raises).
        """
        from src.db.session import get_session

        if not goal_ids:
            return []
        try:
            async with get_session() as session:
                stmt = (
                    select(EvalResult)
                    .where(EvalResult.goal_id.in_(goal_ids))
                    .order_by(EvalResult.created_at.desc())
                    .limit(limit)
                )
                if since is not None:
                    stmt = stmt.where(EvalResult.created_at >= since)
                if until is not None:
                    stmt = stmt.where(EvalResult.created_at <= until)
                if producer_model is not None:
                    stmt = stmt.where(EvalResult.producer_model == producer_model)
                rows = (await session.execute(stmt)).scalars().all()
                return [_row_to_dict(r) for r in rows]
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "EvalStore fetch_rows failed for goals={}: {}", goal_ids, exc
            )
            return []


def _row_to_dict(row: EvalResult) -> dict[str, Any]:
    return {
        "goal_id": row.goal_id,
        "run_id": row.run_id,
        "attempt_id": row.attempt_id,
        "spec_id": row.spec_id,
        "check_name": row.check_name,
        "check_type": row.check_type,
        "passed": row.passed,
        "score": float(row.score),
        "skipped": row.skipped,
        "evidence": row.evidence,
        "cost_usd": float(row.cost_usd),
        "producer_model": row.producer_model,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }
