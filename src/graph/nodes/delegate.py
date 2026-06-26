"""delegate node — delegates subtasks to existing sub-agents.

Selects the best sub-agent from the registry for the current subtask,
executes them via SubAgentRunner with parallel delegation, collects
results, and records metrics.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, cast

from langchain_core.messages import HumanMessage
from loguru import logger

from src.agents.selection import select_subagents_for_subtask
from src.config import get_settings
from src.graph.enums import Phase
from src.graph.models import ToolResult
from src.graph.state import AgentState, objective_goal_text

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

    # F1 — semantic sub-agent selection (default-off). Prune the spawned fan-out
    # to the top-k most-relevant specialists before spawn/execute so an
    # over-spawned round doesn't fan out to irrelevant agents. ``agent_spawn``
    # already decided membership; this only prunes. Fail-safe ⇒ full spawned set
    # on any error (selection can never drop a needed agent). No-op until
    # ``AGENT_SELECTION_ENABLED`` and the spawned set exceeds the cap.
    spawned = await select_subagents_for_subtask(
        spawned, objective_goal_text(cast(AgentState, state)), get_settings()
    )

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
    # Sub-agent tool activity, propagated up so the parent's reducer-backed
    # tool lists (and the e2e report) reflect delegated work, not just the
    # sub_agent:* wrapper results.
    delegated_tool_results: list[ToolResult] = []
    delegated_tools_created: list[dict[str, Any]] = []
    delegated_tools_called: list[dict[str, Any]] = []
    all_success = True

    # Phase 1: Validate all agents and spawn runners
    runners_with_params: list[tuple[Any, str, str, float | None, int]] = []
    runner_agent_names: list[str] = []
    runner_specs: list[Any] = []

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

        goal_text = _build_delegation_goal(spec, state)
        parent_thread_id = state.get("thread_id", "unknown")
        budget_remaining = state.get("budget_remaining")

        if tools is None:
            delegation_results.append({
                "sub_agent_name": agent_name,
                "success": False,
                "result": "",
                "errors": ["No tool registry available for delegation"],
            })
            all_success = False
            continue

        runner = sub_agent_registry.spawn(
            name=agent_name,
            goal=goal_text,
            parent_thread_id=parent_thread_id,
            gateway=gateway,
            tools=tools,
            memory=memory,
        )

        if runner is None:
            delegation_results.append({
                "sub_agent_name": agent_name,
                "success": False,
                "result": "",
                "errors": ["Failed to spawn sub-agent runner"],
            })
            all_success = False
            continue

        runners_with_params.append((runner, goal_text, parent_thread_id, budget_remaining, 0))
        runner_agent_names.append(agent_name)
        runner_specs.append(spec)

    # Phase 2: Execute all runners in parallel
    if runners_with_params:
        # Route each sub-agent to its declared ``spec.model_tier`` (Phase 4 F).
        # Previously every sibling was forced to a SIMPLE-tier diverse set, so a
        # CRITICAL sub-agent ran on a CHEAP model — its declared tier was
        # ignored. Now each runner is pinned to the model its own tier resolves
        # to, and siblings that SHARE a tier still spread across providers via
        # route_diverse (load spread / rate-limit avoidance). Fail-safe: any
        # error or non-list result leaves affinity at its default, so the
        # subgraph routes each call dynamically instead — delegation never
        # aborts. runner.py / _ModelOverrideProxy are unchanged; they honor the
        # now-tier-correct affinity.
        _assign_tier_models(runners_with_params, runner_specs, gateway)

        from src.agents.runner import run_parallel

        parallel_results = await run_parallel(runners_with_params)

        # Phase 3: Post-process results
        for i, result in enumerate(parallel_results):
            agent_name = runner_agent_names[i]
            spec = runner_specs[i]

            delegation_results.append(result)

            # Surface the sub-agent's own tool activity into the parent state.
            delegated_tool_results.extend(result.get("tool_results") or [])
            delegated_tools_created.extend(result.get("tools_created") or [])
            delegated_tools_called.extend(result.get("tools_called") or [])

            if result.get("success"):
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

    # Append a concise delegation summary to the conversation thread so the
    # main agent (now a stateful ReAct loop) sees sub-agent outcomes in its own
    # context when it resumes execution. Minimum-sufficient: name + status +
    # short output excerpt per agent.
    summary_lines = [f"[Delegation complete — {len(new_tool_results)} sub-agent(s)]"]
    for tr in new_tool_results:
        status = "success" if tr.success else "failed"
        excerpt = (tr.output or tr.error or "")[:200]
        summary_lines.append(f"- {tr.tool_name}: {status} — {excerpt}")
    delegation_summary = HumanMessage(content="\n".join(summary_lines))

    return {
        "phase": Phase.VERIFY if all_success else Phase.EXECUTE,
        "delegation_results": delegation_results,
        # Merge the sub_agent:* wrapper results with the delegated sub-agent
        # tool activity so both the verify node and the e2e report see the
        # real tool work performed during delegation.
        "tool_results": new_tool_results + delegated_tool_results,
        "tools_created": delegated_tools_created,
        "tools_called": delegated_tools_called,
        "messages": [delegation_summary],
    }


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


def _assign_tier_models(
    runners: list[tuple[Any, str, str, float | None, int]],
    specs: list[Any],
    gateway: Any,
) -> None:
    """Pin each sub-agent runner to a model for its declared tier (Phase 4 F).

    Runners are grouped by ``spec.model_tier``; within a tier ``route_diverse``
    returns distinct providers (cross-provider load spread / rate-limit
    avoidance), so two CRITICAL siblings don't both hammer one provider. A lone
    sub-agent in a tier is still pinned to that tier's model — a CRITICAL
    sub-agent no longer silently runs on a CHEAP model.

    Best-effort + fail-safe: a missing router, an exception, or a non-list
    result leaves ``runner._model_affinity`` at its default (""), so the
    subgraph routes each call dynamically. Routing never aborts delegation.
    """
    router = getattr(gateway, "_model_router", None)
    if router is None:
        logger.debug("Sub-agent tier routing skipped: no model router on gateway")
        return

    from src.graph.enums import TaskComplexity

    # Group runner indices by declared tier so route_diverse spreads siblings
    # WITHIN a tier across providers, while distinct tiers each resolve to
    # their own tier-appropriate model.
    tier_groups: dict[TaskComplexity, list[int]] = {}
    for idx, spec in enumerate(specs):
        tier = getattr(spec, "model_tier", None) or TaskComplexity.SIMPLE
        tier_groups.setdefault(tier, []).append(idx)

    for tier, indices in tier_groups.items():
        try:
            models = router.route_diverse(n=len(indices), complexity=tier)
        except Exception as e:  # routing must never abort delegation
            logger.debug(f"Sub-agent route_diverse failed for tier {tier}: {e}")
            continue
        if not isinstance(models, list) or not models:
            logger.debug(
                f"Sub-agent route_diverse returned no models for tier {tier}"
            )
            continue
        for slot, idx in enumerate(indices):
            runner = runners[idx][0]
            runner._model_affinity = models[slot % len(models)]

    assigned: list[str] = []
    for entry in runners:
        value = getattr(entry[0], "_model_affinity", "")
        assigned.append(value if isinstance(value, str) else "<unset>")
    logger.debug(
        f"Assigned tier-routed models to {len(runners)} sub-agent(s): {assigned}"
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
