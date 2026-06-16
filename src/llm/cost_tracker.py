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
    ) -> float:
        """Record an LLM API call and return the calculated cost.

        Args:
            model: The model identifier used.
            provider: The provider name.
            input_tokens: Number of input tokens consumed.
            output_tokens: Number of output tokens generated.
            task_id: Optional task execution ID for correlation.
            latency_ms: Optional request latency in milliseconds.

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

    async def check_budget(self) -> tuple[bool, str]:
        """Check if the budget allows another LLM call.

        Returns:
            Tuple of (is_within_budget, message).
        """
        daily_total = await self.get_daily_spend()
        daily_limit = self._budget.max_cost_usd

        if daily_total >= daily_limit:
            return False, f"Daily budget exhausted: ${daily_total:.2f} / ${daily_limit:.2f}"

        if daily_total >= daily_limit * 0.9:
            return True, f"WARNING: Daily budget at 90%: ${daily_total:.2f} / ${daily_limit:.2f}"

        if daily_total >= daily_limit * 0.7:
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

    @staticmethod
    def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for a model call based on per-token pricing.

        Falls back to $0.005/1K input + $0.015/1K output if model not in
        registry or registry entry lacks cost fields.
        """
        spec = MODEL_REGISTRY.get(model)
        if spec and spec.input_cost_per_1k > 0 and spec.output_cost_per_1k > 0:
            cost = (input_tokens * spec.input_cost_per_1k / 1000) + (
                output_tokens * spec.output_cost_per_1k / 1000
            )
            return cost

        # Fallback pricing for unknown models or models without cost data
        logger.warning(f"Using fallback pricing for model '{model}'")
        return (input_tokens * 0.005 / 1000) + (output_tokens * 0.015 / 1000)
