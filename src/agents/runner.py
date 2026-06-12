"""Sub-agent runner — executes a subgraph and collects results.

SubAgentRunner orchestrates a single sub-agent execution: builds the
subgraph from a SubAgentSpec, initializes isolated state, compiles,
invokes, and returns structured results.

run_parallel() executes multiple sub-agents concurrently via asyncio.gather.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.agents.state import initial_sub_agent_state
from src.agents.subgraph import build_subgraph
from src.graph.models import SubAgentSpec

if TYPE_CHECKING:
    from src.llm.gateway import LLMGateway
    from src.memory.manager import MemoryManager
    from src.tools.registry import ToolRegistry


class SubAgentRunner:
    """Runs a single sub-agent execution.

    Built from a SubAgentSpec by SubAgentRegistry.spawn().
    Executes the subgraph in isolation and returns results to the parent.
    """

    def __init__(
        self,
        definition: SubAgentSpec,
        gateway: LLMGateway,
        tools: ToolRegistry,
        memory: MemoryManager | None = None,
    ) -> None:
        self._definition = definition
        self._gateway = gateway
        self._tools = tools
        self._memory = memory
        self._model_affinity: str = ""  # Set by delegate_node for diverse routing

    @property
    def definition(self) -> SubAgentSpec:
        """The sub-agent specification."""
        return self._definition

    async def run(
        self,
        goal: str,
        parent_thread_id: str,
        budget_remaining: float | None = None,
        depth: int = 0,
    ) -> dict[str, Any]:
        """Execute the sub-agent and return results.

        Args:
            goal: The subtask goal for this invocation.
            parent_thread_id: Parent's thread ID for tracking.
            budget_remaining: Remaining budget (for shared mode).
            depth: Current nesting depth.

        Returns:
            Dict with 'success', 'result', 'tokens_used', 'cost_usd',
            'latency_ms', 'iterations', 'errors', 'goal'.
        """
        spec = self._definition

        # ── Depth limit enforcement ────────────────────────────────────
        if spec.depth_limit > 0 and depth >= spec.depth_limit:
            logger.warning(
                f"Sub-agent '{spec.name}': depth limit ({spec.depth_limit}) "
                f"reached at depth {depth}"
            )
            return {
                "success": False,
                "result": "",
                "tokens_used": 0,
                "cost_usd": 0.0,
                "latency_ms": 0,
                "iterations": 0,
                "errors": [f"Depth limit ({spec.depth_limit}) reached"],
                "goal": goal,
                "sub_agent_name": spec.name,
                "sub_agent_id": spec.id,
            }

        # ── Budget setup ──────────────────────────────────────────────
        budget = (
            budget_remaining
            if spec.budget_mode == "shared"
            else spec.budget_limit
        )

        start_time = time.monotonic()

        try:
            # Build and compile subgraph
            graph = build_subgraph(
                spec, self._gateway, self._tools, self._memory, budget,
                preferred_model=self._model_affinity,
            )
            compiled = graph.compile()

            # Initialize isolated state
            state = initial_sub_agent_state(
                goal_text=goal,
                parent_thread_id=parent_thread_id,
                max_iterations=spec.max_iterations,
                depth=depth,
            )
            if budget is not None:
                state["budget_remaining"] = budget

            # Execute subgraph
            result_state = await compiled.ainvoke(dict(state))

            latency_ms = int((time.monotonic() - start_time) * 1000)

            # Extract results
            return _extract_results(result_state, latency_ms, goal, spec)

        except Exception as e:
            latency_ms = int((time.monotonic() - start_time) * 1000)
            logger.error(f"Sub-agent '{spec.name}' execution failed: {e}")
            return {
                "success": False,
                "result": "",
                "tokens_used": 0,
                "cost_usd": 0.0,
                "latency_ms": latency_ms,
                "iterations": 0,
                "errors": [f"Sub-agent execution error: {e!s}"],
                "goal": goal,
                "sub_agent_name": spec.name,
                "sub_agent_id": spec.id,
            }


def _extract_results(
    result_state: dict[str, Any],
    latency_ms: int,
    goal: str,
    spec: SubAgentSpec,
) -> dict[str, Any]:
    """Extract structured results from the final sub-agent state."""
    from src.graph.enums import Confidence

    errors = result_state.get("errors", [])
    final_output = result_state.get("final_output", "")
    is_complete = result_state.get("is_complete", False)

    # Calculate cost from records
    cost_records = result_state.get("cost_records", [])
    total_cost = sum(r.cost_usd for r in cost_records) if cost_records else 0.0

    # Calculate tokens
    total_tokens = result_state.get("total_tokens_used", 0)

    # Determine success based on completion and meaningful output.
    # Transient errors may accumulate via the Annotated reducer but should not
    # override a completed sub-agent with useful output. Errors are still
    # reported in the result for observability.
    #
    # The main graph's verify node sets is_complete/final_output, but the
    # sub-agent subgraph only has classify→plan→execute→reflect→END.
    # When the subgraph reaches END via reflect with high/medium confidence
    # and completed steps, treat it as success even without explicit
    # is_complete/final_output.
    if not is_complete:
        completed_steps = result_state.get("completed_steps", [])
        confidence = result_state.get("confidence", Confidence.LOW)
        if isinstance(confidence, str):
            confidence = Confidence(confidence)

        if completed_steps and confidence in {
            Confidence.HIGH, Confidence.VERY_HIGH, Confidence.MEDIUM,
        }:
            is_complete = True
            if not final_output:
                reflection = result_state.get("reflection")
                if reflection and hasattr(reflection, "summary"):
                    final_output = reflection.summary
                else:
                    final_output = (
                        f"Completed {len(completed_steps)} steps for: "
                        f"{goal[:100]}"
                    )

    success = is_complete and bool(final_output)

    # Quality: average of cost efficiency from reflection
    reflection = result_state.get("reflection")
    quality_rating = None
    if reflection and hasattr(reflection, "cost_efficiency"):
        quality_rating = min(1.0, max(0.0, reflection.cost_efficiency))

    return {
        "success": success,
        "result": final_output,
        "tokens_used": total_tokens,
        "cost_usd": total_cost,
        "latency_ms": latency_ms,
        "iterations": result_state.get("iteration_count", 0),
        "errors": errors,
        "goal": goal,
        "sub_agent_name": spec.name,
        "sub_agent_id": spec.id,
        "quality_rating": quality_rating,
    }


async def run_parallel(
    runners_with_params: list[tuple[SubAgentRunner, str, str, float | None, int]],
) -> list[dict[str, Any]]:
    """Run multiple sub-agents concurrently via asyncio.gather.

    Args:
        runners_with_params: List of tuples (runner, goal, parent_thread_id,
            budget_remaining, depth).

    Returns:
        List of result dicts in the same order as input.
    """
    if not runners_with_params:
        return []

    tasks = [
        runner.run(goal, thread_id, budget, depth)
        for runner, goal, thread_id, budget, depth in runners_with_params
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Convert exceptions to error results
    processed: list[dict[str, Any]] = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            runner = runners_with_params[i][0]
            processed.append({
                "success": False,
                "result": "",
                "tokens_used": 0,
                "cost_usd": 0.0,
                "latency_ms": 0,
                "iterations": 0,
                "errors": [f"Parallel execution error: {result!s}"],
                "goal": runners_with_params[i][1],
                "sub_agent_name": runner.definition.name,
                "sub_agent_id": runner.definition.id,
            })
        else:
            processed.append(result)  # type: ignore[arg-type]

    return processed
