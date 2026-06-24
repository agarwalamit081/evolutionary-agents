"""PostgreSQL-backed cost tracking for LLM API usage."""

from __future__ import annotations

from datetime import datetime, timezone

import sqlalchemy as sa
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.model_registry import MODEL_REGISTRY
from src.config.settings import Settings
from src.db.models import CostLedger


class CostTracker:
    """Tracks LLM API costs in PostgreSQL and enforces budget limits."""

    def __init__(self, session: AsyncSession, settings: Settings) -> None:
        self._session = session
        self._budget = settings.budget

    async def record_usage(
        self,
        model: str,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        task_id: str | None = None,
        latency_ms: int | None = None,
        run_id: str | None = None,
    ) -> float:
        """Record an LLM API call and return the calculated cost.

        Args:
            model: The model identifier used.
            provider: The provider name.
            input_tokens: Number of input tokens consumed.
            output_tokens: Number of output tokens generated.
            task_id: Optional task execution ID for correlation.
            latency_ms: Optional request latency in milliseconds.
            run_id: Optional per-run correlation key — the graph ``thread_id`` of
                the run that issued the call. Enables per-run cost attribution
                (``get_run_spend`` / ``get_runs_summary``). Defaults to ``None``
                (unattributed) when the gateway has no run bound.

        Returns:
            The calculated cost in USD.
        """
        cost_usd = self.calculate_cost(model, input_tokens, output_tokens)

        entry = CostLedger(
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_usd=cost_usd,
            task_id=task_id,
            latency_ms=latency_ms,
            run_id=run_id,
        )
        # Cost tracking is observability-only — a persistence failure here
        # (e.g. a duplicate-key IntegrityError from interleaved commits on the
        # shared session, or a transient connection blip) must never crash the
        # run. Before this guard, a single failed INSERT poisoned the shared
        # session: every later LLM call's record_usage re-raised the same error
        # out of gateway.acompletion, cascading every LLM-integrated node
        # (plan/verify/memory-folding) onto its heuristic fallback. Roll back to
        # clear the failed pending row and recover the session, then return the
        # calculated cost so budgeting stays approximately correct.
        try:
            self._session.add(entry)
            await self._session.commit()
        except Exception as exc:
            logger.warning(
                f"Cost-ledger write failed for {model} (run continues): {exc}"
            )
            try:
                await self._session.rollback()
            except Exception as rollback_exc:  # never mask the original failure
                logger.debug(f"Cost-ledger rollback failed: {rollback_exc}")
            return cost_usd

        logger.debug(
            f"Cost recorded: {model} | "
            f"{input_tokens}+{output_tokens} tokens | "
            f"${cost_usd:.6f}"
        )
        return cost_usd

    async def check_budget(self, run_id: str | None = None) -> tuple[bool, str]:
        """Check if the budget allows another LLM call.

        Enforces two independent caps (findings.md Fact 1 / A2·B2):

        1. **Daily cost** cap (``max_cost_usd``) — the historical pool, checked
           against ``get_daily_spend()``. At the critical/warn thresholds the
           call is allowed and the gateway downgrades the model tier.
        2. **Per-run token** cap (``per_task_token_limit``) — bounds a single
           runaway run independent of the daily pool, so one hard query cannot
           consume the whole day. Only enforced when a ``run_id`` is bound
           (the gateway binds it from the graph ``thread_id``).

        Both caps reuse the gateway's existing not-within-budget path: it
        downgrades to a cheaper model, and only hard-``raise``s when no cheaper
        fallback remains — so a capped run is pushed onto the cheapest tier and
        then stops, rather than aborting abruptly.

        Args:
            run_id: Optional per-run correlation key (the graph ``thread_id``).
                When provided, the per-run token cap is also enforced.

        Returns:
            Tuple of (is_within_budget, message).
        """
        daily_total = await self.get_daily_spend()
        daily_limit = self._budget.max_cost_usd

        if daily_total >= daily_limit:
            return False, f"Daily budget exhausted: ${daily_total:.2f} / ${daily_limit:.2f}"

        per_run_limit = self._budget.per_task_token_limit
        if run_id is not None and per_run_limit > 0:
            run_tokens = await self.get_run_token_usage(run_id)
            if run_tokens >= per_run_limit:
                return (
                    False,
                    f"Per-run token cap reached: {run_tokens} / {per_run_limit} "
                    f"tokens (run {run_id})",
                )

        if daily_total >= daily_limit * self._budget.budget_critical_threshold:
            return True, f"WARNING: Daily budget at {self._budget.budget_critical_threshold:.0%}: ${daily_total:.2f} / ${daily_limit:.2f}"

        if daily_total >= daily_limit * self._budget.budget_warn_threshold:
            return True, f"Budget warning: ${daily_total:.2f} / ${daily_limit:.2f}"

        return True, f"Budget OK: ${daily_total:.2f} / ${daily_limit:.2f}"

    async def get_daily_spend(self) -> float:
        """Get total spend for the current UTC day."""
        today = datetime.now(timezone.utc).date()
        tomorrow = today + __import__("datetime").timedelta(days=1)

        result = await self._session.execute(
            sa.select(sa.func.coalesce(sa.func.sum(CostLedger.cost_usd), 0.0)).where(
                CostLedger.created_at >= datetime.combine(today, datetime.min.time()).replace(tzinfo=timezone.utc),
                CostLedger.created_at < datetime.combine(tomorrow, datetime.min.time()).replace(tzinfo=timezone.utc),
            )
        )
        total = float(result.scalar_one())
        return total

    async def get_daily_token_usage(self) -> dict[str, int]:
        """Get aggregate token usage for the current UTC day."""
        today = datetime.now(timezone.utc).date()
        tomorrow = today + __import__("datetime").timedelta(days=1)

        result = await self._session.execute(
            sa.select(
                sa.func.coalesce(sa.func.sum(CostLedger.input_tokens), 0),
                sa.func.coalesce(sa.func.sum(CostLedger.output_tokens), 0),
                sa.func.coalesce(sa.func.sum(CostLedger.total_tokens), 0),
            ).where(
                CostLedger.created_at >= datetime.combine(today, datetime.min.time()).replace(tzinfo=timezone.utc),
                CostLedger.created_at < datetime.combine(tomorrow, datetime.min.time()).replace(tzinfo=timezone.utc),
            )
        )
        row = result.one()
        return {"input_tokens": int(row[0]), "output_tokens": int(row[1]), "total_tokens": int(row[2])}

    async def get_run_spend(self, run_id: str) -> float:
        """Total USD spend attributed to a single run.

        Sums ``cost_usd`` over every cost_ledger row carrying ``run_id``. A run
        with no rows (or whose calls predate run-attribution) returns ``0.0``.

        Args:
            run_id: The graph ``thread_id`` of the run (e.g. ``cli-q05``).

        Returns:
            Total spend in USD for that run.
        """
        result = await self._session.execute(
            sa.select(sa.func.coalesce(sa.func.sum(CostLedger.cost_usd), 0.0)).where(
                CostLedger.run_id == run_id
            )
        )
        return float(result.scalar_one())

    async def get_run_token_usage(self, run_id: str) -> int:
        """Total tokens attributed to a single run.

        Sums ``total_tokens`` over every cost_ledger row carrying ``run_id``.
        Backs the per-run token cap in ``check_budget`` (findings.md A2/B2). A
        run with no rows (or whose calls predate run-attribution) returns ``0``.

        Args:
            run_id: The graph ``thread_id`` of the run (e.g. ``cli-q05``).

        Returns:
            Total tokens consumed by that run.
        """
        result = await self._session.execute(
            sa.select(sa.func.coalesce(sa.func.sum(CostLedger.total_tokens), 0)).where(
                CostLedger.run_id == run_id
            )
        )
        return int(result.scalar_one())

    async def get_runs_summary(self) -> list[dict[str, object]]:
        """Aggregate spend per run, most-expensive first.

        Groups cost_ledger by ``run_id`` (excluding NULL/unattributed rows) and
        returns one dict per run with its total cost, call count, token totals,
        and the timestamp of its most recent call. Used to answer "which run
        cost what" — the attribution that was impossible while run_id was always
        NULL.

        Returns:
            List of per-run summary dicts, sorted by cost descending.
        """
        result = await self._session.execute(
            sa.select(
                CostLedger.run_id,
                sa.func.coalesce(sa.func.sum(CostLedger.cost_usd), 0.0).label("cost_usd"),
                sa.func.count(CostLedger.id).label("calls"),
                sa.func.coalesce(sa.func.sum(CostLedger.input_tokens), 0).label("input_tokens"),
                sa.func.coalesce(sa.func.sum(CostLedger.output_tokens), 0).label("output_tokens"),
                sa.func.coalesce(sa.func.sum(CostLedger.total_tokens), 0).label("total_tokens"),
                sa.func.max(CostLedger.created_at).label("last_call"),
            )
            .where(CostLedger.run_id.is_not(None))
            .group_by(CostLedger.run_id)
            .order_by(sa.func.sum(CostLedger.cost_usd).desc())
        )
        return [
            {
                "run_id": row.run_id,
                "cost_usd": float(row.cost_usd),
                "calls": int(row.calls),
                "input_tokens": int(row.input_tokens),
                "output_tokens": int(row.output_tokens),
                "total_tokens": int(row.total_tokens),
                "last_call": row.last_call,
            }
            for row in result.all()
        ]

    @staticmethod
    def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for a model call based on per-token pricing.

        A model that is in MODEL_REGISTRY is priced from its explicit
        input_cost_per_1k / output_cost_per_1k fields — including free-tier
        models (NVIDIA API, Ollama local, OpenRouter :free), which carry 0.0 by
        design and therefore cost $0.0. Only a model absent from the registry
        entirely falls back to the conservative generic rate.

        Why this matters: the previous ``> 0`` guard treated every free-tier
        model as "lacking cost data" and billed it at the $0.005/$0.015 fallback
        rate, which inflated daily spend with phantom free-tier charges and
        falsely tripped the budget gate into a degradation cascade.
        """
        spec = MODEL_REGISTRY.get(model)
        if spec is not None:
            return (input_tokens * spec.input_cost_per_1k / 1000) + (
                output_tokens * spec.output_cost_per_1k / 1000
            )

        # Fallback pricing ONLY for models absent from the registry. This
        # over-estimates (≈ sonnet rate), which is the safe direction for an
        # unknown paid model — the budget gate trips early rather than late.
        logger.warning(f"Using fallback pricing for model '{model}' (not in registry)")
        return (input_tokens * 0.005 / 1000) + (output_tokens * 0.015 / 1000)
