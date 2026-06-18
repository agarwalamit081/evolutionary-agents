"""Persistent eval-result store (Phase 3).

Writes one ``eval_results`` row per (run, goal, check) and queries them back by
goal or run for regression tracking and the Phase-8 evolution canary. Like the
cost ledger, this is **observability-only**: a DB hiccup here is logged at
WARNING and never re-raises — a poisoned write can never abort a run or a verify
cycle. Writes are gated behind ``EvalSettings.eval_store_enabled``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import EvalResult

if TYPE_CHECKING:
    from src.eval.models import BenchmarkResult, CheckResult, CorrectnessResult


async def _store_check(
    session: AsyncSession,
    *,
    goal_id: str,
    run_id: str,
    spec_id: str | None,
    check: CheckResult,
    cost_usd: float,
) -> None:
    session.add(
        EvalResult(
            goal_id=goal_id,
            run_id=run_id,
            spec_id=spec_id,
            check_name=check.check_name,
            check_type=check.check_type,
            passed=check.passed,
            score=float(check.score),
            skipped=check.skipped,
            evidence=dict(check.evidence) or None,
            cost_usd=cost_usd,
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
        cost_usd: float = 0.0,
    ) -> int:
        """Persist every check in ``correctness``; return rows written.

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
                        spec_id=correctness.spec_id or None,
                        check=check,
                        cost_usd=cost_usd,
                    )
                    written += 1
        except Exception as exc:  # noqa: BLE001 — observability-only, never re-raise
            logger.warning(
                "EvalStore write failed for run={} goal={} ({} checks): {}",
                run_id,
                goal_id,
                len(correctness.checks),
                exc,
            )
            return 0
        logger.debug(
            "EvalStore wrote {} check rows for run={} goal={}",
            written,
            run_id,
            goal_id,
        )
        return written

    async def record_benchmark(
        self,
        result: BenchmarkResult,
        *,
        run_id: str,
        cost_usd: float = 0.0,
    ) -> int:
        """Persist a BenchmarkResult's checks (one row per check)."""
        from src.eval.models import CorrectnessResult

        if not result.checks:
            return 0
        correctness = CorrectnessResult(
            spec_id="",
            overall_score=result.correctness_score or 0.0,
            passed=all(not c.skipped and c.passed for c in result.checks),
            checks=list(result.checks),
        )
        return await self.record_correctness(
            correctness, goal_id=result.goal_name, run_id=run_id, cost_usd=cost_usd
        )

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


def _row_to_dict(row: EvalResult) -> dict[str, Any]:
    return {
        "goal_id": row.goal_id,
        "run_id": row.run_id,
        "spec_id": row.spec_id,
        "check_name": row.check_name,
        "check_type": row.check_type,
        "passed": row.passed,
        "score": float(row.score),
        "skipped": row.skipped,
        "evidence": row.evidence,
        "cost_usd": float(row.cost_usd),
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }
