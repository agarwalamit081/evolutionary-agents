"""agent_spawn node — creates and registers new sub-agents.

Called when the reflect node detects a need for a specialized sub-agent
(pending_agent_gaps). Uses LLM to generate a SubAgentProposal, validates
it, persists to DB, and registers in SubAgentRegistry.

Follows the exact pattern of tool_create_node (src/graph/nodes/tool_create.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from src.agents.registry import MAX_SUB_AGENTS_PER_RUN
from src.graph.enums import Phase, TaskComplexity
from src.graph.models import SubAgentSpec

if TYPE_CHECKING:
    from src.agents.registry import SubAgentRegistry
    from src.graph.schemas import SubAgentProposal
    from src.llm.gateway import LLMGateway
    from src.tools.registry import ToolRegistry


async def agent_spawn_node(
    state: dict[str, Any],
    *,
    gateway: LLMGateway | None = None,
    tools: ToolRegistry | None = None,
    sub_agent_registry: SubAgentRegistry | None = None,
) -> dict[str, Any]:
    """Create and register new sub-agents for identified capability gaps.

    Flow:
        1. Read pending_agent_gaps from state
        2. For each gap, call LLM to generate SubAgentProposal
        3. Validate proposal (name uniqueness, tool availability)
        4. Create SubAgentSpec → persist via SubAgentPersister → register
        5. Return updated state with sub_agents_spawned

    Returns:
        Partial state update with sub_agents_spawned, pending_agent_gaps
        (cleared), and phase set to DELEGATE (if agents created) or EXECUTE.
    """
    pending_gaps: list[str] = state.get("pending_agent_gaps", [])

    if not pending_gaps:
        logger.debug("No agent gaps to spawn, routing to execute")
        return {
            "phase": Phase.EXECUTE,
            "pending_agent_gaps": [],
            "sub_agents_spawned": [],
        }

    if gateway is None or sub_agent_registry is None:
        logger.warning(
            "agent_spawn requires gateway and sub_agent_registry, "
            "falling back to execute"
        )
        return {
            "phase": Phase.EXECUTE,
            "pending_agent_gaps": [],
            "sub_agents_spawned": [],
        }

    spawned: list[dict[str, Any]] = []
    converted_tool_gaps: list[str] = []
    created_count = len(state.get("sub_agents_spawned", []))

    for gap_description in pending_gaps:
        # Rate limit check
        if created_count >= MAX_SUB_AGENTS_PER_RUN:
            logger.warning(
                f"Max sub-agents per run ({MAX_SUB_AGENTS_PER_RUN}) reached, "
                f"converting remaining agent gaps to tool gaps"
            )
            # Convert remaining unhandled agent gaps into tool creation opportunities
            remaining_idx = pending_gaps.index(gap_description)
            converted_tool_gaps = [
                f"tool to handle subtask: {g}"
                for g in pending_gaps[remaining_idx:]
            ]
            break

        spawn_result = await _spawn_single_agent(
            gap_description=gap_description,
            gateway=gateway,
            tools=tools,
            registry=sub_agent_registry,
            state=state,
        )

        if spawn_result is not None:
            spawned.append(spawn_result)
            created_count += 1
            logger.info(f"Spawned sub-agent '{spawn_result['name']}'")
        else:
            # Failed spawn — convert to tool gap as fallback
            converted_tool_gaps.append(f"tool to handle subtask: {gap_description}")

    result: dict[str, Any] = {
        "phase": Phase.DELEGATE if spawned else Phase.EXECUTE,
        "pending_agent_gaps": [],
        "sub_agents_spawned": spawned,
    }

    # If remaining gaps were converted to tool gaps, include them
    if converted_tool_gaps:
        result["pending_tool_gaps"] = converted_tool_gaps

    return result


async def _spawn_single_agent(
    gap_description: str,
    gateway: LLMGateway,
    tools: ToolRegistry | None,
    registry: SubAgentRegistry,
    state: dict[str, Any],
) -> dict[str, Any] | None:
    """Generate, validate, persist, and register a single sub-agent.

    Returns:
        Dict with spawn details on success, None on failure.
    """
    from src.graph.prompts import AGENT_SPAWN_SYSTEM, AGENT_SPAWN_USER
    from src.graph.schemas import SubAgentProposal

    goal_text = ""
    goal = state.get("current_goal")
    if goal and hasattr(goal, "text"):
        goal_text = goal.text

    # Build context for LLM
    available_tools = tools.list_names() if tools else []
    existing_agents = registry.list_names()
    strategy = state.get("strategy", "react")

    user_prompt = AGENT_SPAWN_USER.format(
        gap_description=gap_description,
        goal_text=goal_text[:500],
        available_tools=", ".join(available_tools),
        existing_agents=", ".join(existing_agents) if existing_agents else "none",
        strategy=strategy,
    )

    try:
        # Generate proposal via LLM
        from src.llm.structured_output import StructuredOutputManager

        extractor = StructuredOutputManager()
        response = await gateway.acompletion(
            messages=[
                {"role": "system", "content": AGENT_SPAWN_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
        )

        if not response.content:
            logger.warning(f"LLM returned empty response for agent spawn: {gap_description}")
            return None

        proposal = await extractor.extract(response.content, SubAgentProposal)

        if proposal is None:
            logger.warning(f"Failed to parse SubAgentProposal for: {gap_description}")
            return None

    except Exception as e:
        logger.warning(f"LLM call failed for agent spawn: {e}")
        return None

    # Validate proposal
    validation_errors = _validate_proposal(proposal, registry, tools)
    if validation_errors:
        logger.warning(
            f"Invalid sub-agent proposal '{proposal.name}': "
            f"{'; '.join(validation_errors)}"
        )
        return None

    # Create SubAgentSpec from proposal
    spec = SubAgentSpec(
        name=proposal.name,
        description=proposal.description,
        goal=proposal.goal_description,
        template_type=proposal.template_type,
        tool_scope=proposal.tool_scope,
        tool_subset=proposal.tool_subset,
        model_tier=_parse_model_tier(proposal.model_tier),
        parent_thread_id=state.get("thread_id", ""),
        max_iterations=10,
        depth_limit=0,
    )

    # Persist to DB (best-effort, non-blocking)
    await _persist_agent(spec)

    # Register in memory
    registry.register(spec)

    return {
        "name": spec.name,
        "description": spec.description,
        "template_type": spec.template_type,
        "tool_scope": spec.tool_scope,
        "id": spec.id,
    }


def _validate_proposal(
    proposal: SubAgentProposal,
    registry: SubAgentRegistry,
    tools: ToolRegistry | None,
) -> list[str]:
    """Validate a sub-agent proposal.

    Returns:
        List of validation error strings. Empty means valid.
    """
    errors: list[str] = []

    # Name uniqueness
    if registry.has(proposal.name):
        errors.append(f"Sub-agent '{proposal.name}' already exists")

    # Name format (snake_case)
    if not proposal.name.replace("_", "").isalnum():
        errors.append(f"Invalid name '{proposal.name}' (must be snake_case)")

    # Tool availability
    if proposal.tool_scope == "inherit_subset" and tools:
        for tool_name in proposal.tool_subset:
            if not tools.has(tool_name):
                errors.append(f"Requested tool '{tool_name}' not found in registry")

    # Template type
    if proposal.template_type not in ("fixed", "custom"):
        errors.append(f"Invalid template_type: {proposal.template_type}")

    # Tool scope
    if proposal.tool_scope not in ("inherit_all", "inherit_subset", "self_create"):
        errors.append(f"Invalid tool_scope: {proposal.tool_scope}")

    return errors


def _parse_model_tier(tier_str: str) -> TaskComplexity:
    """Parse model tier string to TaskComplexity enum."""
    tier_map = {t.value: t for t in TaskComplexity}
    return tier_map.get(tier_str, TaskComplexity.SIMPLE)


async def _persist_agent(spec: SubAgentSpec) -> None:
    """Persist a sub-agent definition to DB (best-effort, non-fatal)."""
    try:
        from src.agents.persister import SubAgentPersister

        persister = SubAgentPersister()
        agent_id = await persister.persist(spec)
        if agent_id:
            logger.debug(f"Persisted sub-agent '{spec.name}' to DB: {agent_id}")
    except Exception as e:
        logger.debug(f"Sub-agent persistence skipped: {e}")
