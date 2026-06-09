"""delegate node — delegates subtasks to existing sub-agents.

Selects the best sub-agent from the registry for the current subtask,
executes it via SubAgentRunner, collects results, and records metrics.

Supports parallel delegation for multiple independent subtasks.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.graph.enums import Phase
from src.graph.models import ToolResult

if TYPE_CHECKING:
    from src.agents.registry import SubAgentRegistry
    from src.llm.gateway import LLMGateway
    from src.memory.manager import MemoryManager
    from src.tools.registry import ToolRegistry


async def delegate_node(
    state: dict[str, Any],
    *,
    gateway: LLMGateway | None = None,
    tools: ToolRegistry | None = None,
    sub_agent_registry: SubAgentRegistry | None = None,
    memory: MemoryManager | None = None,
) -> dict[str, Any]:
    """Delegate subtask to existing sub-agents and collect results.

    Flow:
        1. Select sub-agent (LLM selection or heuristic from spawned agents)
        2. Create SubAgentRunner from definition
        3. Execute subgraph with isolated state
        4. Collect results into parent state
        5. Record run metrics via SubAgentPersister
        6. Auto-deprecation check

    Returns:
        Partial state update with delegation_results, tool_results,
        and phase set to VERIFY (success) or EXECUTE (failure).
    """
    spawned = state.get("sub_agents_spawned", [])

    if not spawned:
        logger.debug("No sub-agents to delegate to, routing to execute")
        return {
            "phase": Phase.EXECUTE,
            "delegation_results": [],
        }

    if gateway is None or sub_agent_registry is None:
        logger.warning(
            "delegate requires gateway and sub_agent_registry, "
            "falling back to execute"
        )
        return {
            "phase": Phase.EXECUTE,
            "delegation_results": [],
        }

    # Collect results from each spawned agent
    delegation_results: list[dict[str, Any]] = []
    new_tool_results: list[ToolResult] = []
    all_success = True

    for agent_info in spawned:
        agent_name = agent_info.get("name", "")
        spec = sub_agent_registry.get(agent_name)

        if spec is None or not spec.is_active:
            logger.warning(f"Sub-agent '{agent_name}' not found or inactive, skipping")
            delegation_results.append({
                "sub_agent_name": agent_name,
                "success": False,
                "result": "",
                "errors": ["Sub-agent not found or inactive"],
            })
            all_success = False
            continue

        result = await _delegate_single(
            spec=spec,
            state=state,
            gateway=gateway,
            tools=tools,
            registry=sub_agent_registry,
            memory=memory,
        )

        delegation_results.append(result)

        if result.get("success"):
            # Add result as a tool result for the parent agent to use
            new_tool_results.append(ToolResult(
                tool_name=f"sub_agent:{agent_name}",
                success=True,
                output=result.get("result", ""),
                tokens_used=result.get("tokens_used", 0),
                duration_ms=result.get("latency_ms", 0),
            ))
        else:
            all_success = False
            new_tool_results.append(ToolResult(
                tool_name=f"sub_agent:{agent_name}",
                success=False,
                output=result.get("result", ""),
                error="; ".join(result.get("errors", [])),
                tokens_used=result.get("tokens_used", 0),
                duration_ms=result.get("latency_ms", 0),
            ))

        # Record metrics (best-effort)
        await _record_metrics(spec, result, state)

        # Auto-deprecation check
        sub_agent_registry.check_deprecation(agent_name)

    return {
        "phase": Phase.VERIFY if all_success else Phase.EXECUTE,
        "delegation_results": delegation_results,
        "tool_results": new_tool_results,
    }


async def _delegate_single(
    spec: Any,
    state: dict[str, Any],
    gateway: LLMGateway,
    tools: ToolRegistry | None,
    registry: SubAgentRegistry,
    memory: MemoryManager | None,
) -> dict[str, Any]:
    """Delegate a single subtask to a sub-agent.

    Args:
        spec: SubAgentSpec for the target sub-agent.
        state: Current parent agent state.
        gateway: LLMGateway instance.
        tools: Parent's ToolRegistry.
        registry: SubAgentRegistry for spawning.
        memory: Optional MemoryManager.

    Returns:
        Result dict from the sub-agent execution.
    """
    # Determine the goal for this sub-agent
    goal_text = _build_delegation_goal(spec, state)
    parent_thread_id = state.get("thread_id", "unknown")
    budget_remaining = state.get("budget_remaining")

    # Spawn the runner
    if tools is None:
        return {
            "sub_agent_name": spec.name,
            "sub_agent_id": spec.id,
            "success": False,
            "result": "",
            "errors": ["No tool registry available for delegation"],
        }

    runner = registry.spawn(
        name=spec.name,
        goal=goal_text,
        parent_thread_id=parent_thread_id,
        gateway=gateway,
        tools=tools,
        memory=memory,
    )

    if runner is None:
        return {
            "sub_agent_name": spec.name,
            "sub_agent_id": spec.id,
            "success": False,
            "result": "",
            "errors": ["Failed to spawn sub-agent runner"],
        }

    # Execute
    result = await runner.run(
        goal=goal_text,
        parent_thread_id=parent_thread_id,
        budget_remaining=budget_remaining,
        depth=0,
    )

    return result


def _build_delegation_goal(spec: Any, state: dict[str, Any]) -> str:
    """Build a goal string for the sub-agent delegation."""
    main_goal = ""
    goal = state.get("current_goal")
    if goal and hasattr(goal, "text"):
        main_goal = goal.text

    # Use the agent's description + main goal context
    return (
        f"Main goal context: {main_goal[:300]}\n\n"
        f"Your specialization: {spec.description}\n\n"
        f"Complete the subtask assigned to you."
    )


async def _record_metrics(
    spec: Any,
    result: dict[str, Any],
    state: dict[str, Any],
) -> None:
    """Record sub-agent execution metrics (best-effort, non-fatal)."""
    try:
        # Try to get the DB model ID
        agent_id_str = spec.id if hasattr(spec, "id") else None
        if not agent_id_str:
            return

        # Parse UUID (might be hex short ID, skip if not parseable)
        try:
            agent_uuid = uuid.UUID(agent_id_str)
        except ValueError:
            logger.debug(
                f"Sub-agent '{spec.name}' has non-UUID id '{agent_id_str}', "
                f"skipping metric recording"
            )
            return

        from src.agents.persister import SubAgentPersister

        persister = SubAgentPersister()
        await persister.record_run_and_update_metrics(
            sub_agent_id=agent_uuid,
            run_result={
                **result,
                "goal": result.get("goal", spec.description),
            },
            parent_thread_id=state.get("thread_id", ""),
        )

    except Exception as e:
        logger.debug(f"Metric recording skipped for '{spec.name}': {e}")
