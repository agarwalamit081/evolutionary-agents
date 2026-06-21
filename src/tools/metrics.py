"""Per-tool success-metrics recording (M4).

``ToolMetricsRecorder.record`` appends one ``tool_call_metrics`` detail row per
tool invocation and updates the running aggregates on ``tool_registrations``
(``calls``/``success_rate``/``empty_output_rate``/``last_run_at``) **atomically in
a single UPDATE** — the incremental mean is computed server-side from the old
column values so concurrent tool calls (the execute node runs them under a
semaphore) can't lose updates. The recorder is observability-only: a DB hiccup
is logged at DEBUG and never re-raises (the CostTracker-resilience pattern), so
a poisoned write can never break an agent run.

Gated behind ``AgentSettings.tool_metrics_enabled`` (env ``TOOL_METRICS_ENABLED``).
"""

from __future__ import annotations

from typing import Any

from loguru import logger


def roll_rate(prev_rate: float, prev_count: int, sample: float) -> float:
    """Incremental mean: fold one new ``sample`` into a running rate.

    Equivalent to recomputing the mean over ``prev_count + 1`` samples. The SQL
    UPDATE in :meth:`ToolMetricsRecorder.record` mirrors this server-side for
    atomicity; this pure form is unit-tested directly. ``prev_count`` may be 0
    (first sample) — the denominator is then 1, never zero.
    """
    next_count = prev_count + 1
    return (prev_rate * prev_count + sample) / next_count


class ToolMetricsRecorder:
    """Record per-invocation tool metrics; maintain running aggregates.

    Stateless — safe to instantiate per call site or hold as a singleton.
    """

    async def record(
        self,
        tool_name: str,
        *,
        success: bool,
        empty_output: bool,
        run_id: str | None = None,
        latency_ms: int | None = None,
    ) -> None:
        """Record one tool invocation's outcome.

        Args:
            tool_name: Executed tool (built-in or generated).
            success: Whether the handler returned without raising.
            empty_output: Whether a *successful* result was blank/whitespace —
                a chronic empty-output tool is a poor performer even when it
                doesn't error.
            run_id: Optional run/thread id for the detail-row audit trail.
            latency_ms: Optional wall-clock latency in milliseconds.
        """
        # Lazy import so tests can patch src.config.settings.get_settings.
        from src.config.settings import get_settings

        if not get_settings().agent.tool_metrics_enabled:
            return

        sample_ok = 1.0 if success else 0.0
        sample_empty = 1.0 if empty_output else 0.0
        try:
            from sqlalchemy import func, insert, update

            from src.db.models import ToolCallMetric, ToolRegistration
            from src.db.session import get_session

            async with get_session() as session:
                # Atomic incremental mean. Every column reference on the RHS
                # resolves to the OLD row value within a single UPDATE, so this
                # is race-free under concurrent tool calls. Denominator is
                # calls+1 (>= 1) — no division by zero.
                await session.execute(
                    update(ToolRegistration)
                    .where(ToolRegistration.tool_name == tool_name)
                    .values(
                        calls=ToolRegistration.calls + 1,
                        success_rate=(
                            ToolRegistration.success_rate * ToolRegistration.calls
                            + sample_ok
                        )
                        / (ToolRegistration.calls + 1),
                        empty_output_rate=(
                            ToolRegistration.empty_output_rate * ToolRegistration.calls
                            + sample_empty
                        )
                        / (ToolRegistration.calls + 1),
                        last_run_at=func.now(),
                    )
                )
                await session.execute(
                    insert(ToolCallMetric).values(
                        tool_name=tool_name,
                        run_id=run_id,
                        success=success,
                        empty_output=empty_output,
                        latency_ms=latency_ms,
                    )
                )
        except Exception as exc:  # noqa: BLE001 — metrics are observability-only
            logger.debug("Tool metric record skipped for '{}': {}", tool_name, exc)

    async def record_result(self, tool_name: str, result: Any, *, run_id: str | None = None) -> None:
        """Convenience: record from a ``ToolResult``-like object.

        Reads ``success``/``output``/``error`` off ``result`` (the execute node's
        ``ToolResult``). A failed result (``success`` False) is recorded as
        non-empty (the failure signal is success_rate, not empty_output).
        """
        success = bool(getattr(result, "success", False))
        # Empty-output is only meaningful for a successful call: a blank success
        # is a poor deliverable; an error always carries an error string.
        out = getattr(result, "output", "") or ""
        empty = success and not str(out).strip()
        await self.record(tool_name, success=success, empty_output=empty, run_id=run_id)
