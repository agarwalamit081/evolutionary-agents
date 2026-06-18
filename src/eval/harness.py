"""Benchmark harness for running agent performance evaluations.

Orchestrates benchmark runs against the agent graph, collects per-node
metrics, and generates markdown/JSON reports.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.eval.models import BenchmarkGoal, BenchmarkResult, CheckResult, GoalSpec

if TYPE_CHECKING:
    from src.agents.registry import SubAgentRegistry
    from src.llm.gateway import LLMGateway
    from src.tools.registry import ToolRegistry


class BenchmarkHarness:
    """Orchestrates benchmark runs and collects metrics.

    Usage:
        harness = BenchmarkHarness(gateway, tools, registry)
        results = await harness.run_suite(goals)
        report = harness.generate_report(results)
    """

    def __init__(
        self,
        gateway: LLMGateway,
        tools: ToolRegistry,
        sub_agent_registry: SubAgentRegistry,
    ) -> None:
        self._gateway = gateway
        self._tools = tools
        self._registry = sub_agent_registry

    async def run_benchmark(
        self, goal: BenchmarkGoal, spec: GoalSpec | None = None
    ) -> BenchmarkResult:
        """Run a single benchmark goal and collect metrics.

        Args:
            goal: Benchmark scenario to execute.
            spec: Optional GoalSpec whose correctness checks the verify node
                runs (when ``EVAL_ENABLED``); its ``spec_id`` is threaded into
                state so verify can look it up.

        Returns:
            BenchmarkResult with latency, tokens, cost, quality, and
            (when a spec scored) correctness metrics.
        """
        from src.config import get_settings
        from src.graph.factory import initial_state
        from src.graph.task_graph import compile_task_graph

        get_settings()  # Ensure settings loaded
        start_time = time.monotonic()

        try:
            thread_id = f"bench-{goal.name}-{int(start_time)}"
            state = initial_state(
                goal_text=goal.goal_text,
                thread_id=thread_id,
                max_iterations=goal.max_iterations,
            )
            if spec is not None:
                # Thread the spec id so verify runs its correctness checks when
                # EVAL_ENABLED. A plain key set on the TypedDict (total=False).
                state["eval_goal_spec_id"] = spec.spec_id

            compiled = compile_task_graph(
                gateway=self._gateway,
                memory=None,
                tools=self._tools,
                checkpointer=None,
                sub_agent_registry=self._registry,
            )

            result_state = await compiled.ainvoke(dict(state))
            result_state = dict(result_state) if not isinstance(result_state, dict) else result_state
            latency_ms = int((time.monotonic() - start_time) * 1000)

            return self._extract_result(goal, result_state, latency_ms)

        except Exception as exc:
            latency_ms = int((time.monotonic() - start_time) * 1000)
            logger.error(f"Benchmark '{goal.name}' failed: {exc}")
            return BenchmarkResult(
                goal_name=goal.name,
                category=goal.category,
                success=False,
                total_latency_ms=latency_ms,
                total_tokens=0,
                total_cost_usd=0.0,
                iterations=0,
                errors=[str(exc)],
            )

    async def run_suite(self, goals: list[BenchmarkGoal]) -> list[BenchmarkResult]:
        """Run all benchmark goals sequentially.

        Args:
            goals: List of benchmark scenarios.

        Returns:
            List of BenchmarkResult in the same order.
        """
        results: list[BenchmarkResult] = []
        for i, goal in enumerate(goals, 1):
            logger.info(f"Running benchmark {i}/{len(goals)}: {goal.name}")
            result = await self.run_benchmark(goal)
            results.append(result)
            logger.info(
                f"Benchmark '{goal.name}': "
                f"{'PASS' if result.success else 'FAIL'} "
                f"({result.total_latency_ms}ms, {result.iterations} iterations)"
            )
        return results

    def generate_report(self, results: list[BenchmarkResult]) -> str:
        """Generate a markdown report from benchmark results.

        Args:
            results: List of benchmark results.

        Returns:
            Markdown report string.
        """
        lines: list[str] = []
        lines.append("# Agent Benchmark Report")
        lines.append("")

        # Summary
        total = len(results)
        passed = sum(1 for r in results if r.success)
        total_latency = sum(r.total_latency_ms for r in results)
        total_tokens = sum(r.total_tokens for r in results)
        total_cost = sum(r.total_cost_usd for r in results)

        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Pass rate**: {passed}/{total} ({100 * passed / max(total, 1):.0f}%)")
        lines.append(f"- **Total latency**: {total_latency:,}ms")
        lines.append(f"- **Total tokens**: {total_tokens:,}")
        lines.append(f"- **Total cost**: ${total_cost:.4f}")
        lines.append("")

        # Results table
        lines.append("## Results")
        lines.append("")
        lines.append("| Goal | Category | Status | Latency | Tokens | Cost | Iterations |")
        lines.append("|------|----------|--------|---------|--------|------|------------|")
        for r in results:
            status = "✅" if r.success else "❌"
            lines.append(
                f"| {r.goal_name} | {r.category} | {status} | "
                f"{r.total_latency_ms:,}ms | {r.total_tokens:,} | "
                f"${r.total_cost_usd:.4f} | {r.iterations} |"
            )
        lines.append("")

        # Per-result details
        for r in results:
            lines.append(f"### {r.goal_name}")
            lines.append("")
            lines.append(f"- **Goal**: {r.final_output[:200] if r.final_output else 'N/A'}")
            lines.append(f"- **Tools used**: {r.tool_results_count}")
            lines.append(f"- **Sub-agents**: {r.sub_agents_spawned}")
            lines.append(f"- **Tools created**: {r.tools_created}")
            lines.append(f"- **Quality**: {r.quality_score:.2f}")
            if r.errors:
                lines.append(f"- **Errors**: {len(r.errors)}")
                for err in r.errors[:5]:
                    lines.append(f"  - {err[:200]}")
            lines.append("")

        return "\n".join(lines)

    def export_json(self, results: list[BenchmarkResult]) -> str:
        """Export benchmark results as JSON.

        Args:
            results: List of benchmark results.

        Returns:
            JSON string.
        """
        data = [r.model_dump(mode="json") for r in results]
        return json.dumps(data, indent=2, default=str)

    def _extract_result(
        self,
        goal: BenchmarkGoal,
        state: dict[str, Any],
        latency_ms: int,
    ) -> BenchmarkResult:
        """Extract structured BenchmarkResult from final agent state."""
        errors = state.get("errors", [])
        final_output = state.get("final_output", "")
        is_complete = state.get("is_complete", False)

        # Token and cost extraction
        total_tokens = state.get("total_tokens_used", 0)
        cost_records = state.get("cost_records", [])
        total_cost = sum(
            getattr(cr, "cost_usd", 0) if hasattr(cr, "cost_usd") else cr.get("cost_usd", 0)
            for cr in cost_records
        )

        # Tool and sub-agent counts
        tool_results = state.get("tool_results", [])
        tools_created = state.get("tools_created", [])
        sub_agents_spawned = state.get("sub_agents_spawned", [])

        # Quality score: based on completion and output presence
        quality_score = 0.0
        if is_complete and final_output:
            quality_score = 1.0
        elif final_output:
            quality_score = 0.5

        # Correctness layer (Phase 3): the verify node writes the aggregate
        # score + per-check breakdown to state when a GoalSpec ran. None when no
        # spec was registered or the goal didn't reach a completion verify.
        correctness_score = state.get("eval_correctness_score")
        checks_raw = state.get("eval_checks") or []
        checks: list[CheckResult] = []
        for raw in checks_raw:
            if isinstance(raw, dict):
                try:
                    checks.append(CheckResult(**raw))
                except Exception:  # noqa: BLE001 — never let one bad row drop the result
                    logger.debug("Skipping malformed eval check row: {}", raw)

        return BenchmarkResult(
            goal_name=goal.name,
            category=goal.category,
            success=is_complete,
            total_latency_ms=latency_ms,
            total_tokens=total_tokens,
            total_cost_usd=total_cost,
            iterations=state.get("iteration_count", 0),
            tool_results_count=len(tool_results),
            sub_agents_spawned=len(sub_agents_spawned),
            tools_created=len(tools_created),
            quality_score=quality_score,
            errors=[str(e)[:500] for e in errors] if errors else [],
            final_output=str(final_output)[:1000],
            correctness_score=correctness_score,
            checks=checks,
        )
