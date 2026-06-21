"""agent_spawn node — creates and registers new sub-agents.

Called when the reflect node detects a need for a specialized sub-agent
(pending_agent_gaps). Uses LLM to generate a SubAgentProposal, validates
it, persists to DB, and registers in SubAgentRegistry.

Follows the exact pattern of tool_create_node (src/graph/nodes/tool_create.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from src.config import get_settings
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
            "attempted_agent_gaps": [],
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
            "attempted_agent_gaps": [],
            "sub_agents_spawned": [],
        }

    spawned: list[dict[str, Any]] = []
    converted_tool_gaps: list[str] = []
    attempted: list[str] = []
    created_count = len(state.get("sub_agents_spawned", []))

    max_sub_agents = get_settings().agent.max_sub_agents_per_run
    for gap_description in pending_gaps:
        # Rate limit check
        if created_count >= max_sub_agents:
            logger.warning(
                f"Max sub-agents per run ({max_sub_agents}) reached, "
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

        attempted.append(gap_description)

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
        "attempted_agent_gaps": attempted,
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
        # Generate proposal via LLM — include JSON schema in system prompt
        # so the LLM knows the expected output format
        from src.llm.structured_output import StructuredOutputManager

        extractor = StructuredOutputManager()
        system_content = StructuredOutputManager.build_structured_prompt(
            str(AGENT_SPAWN_SYSTEM), SubAgentProposal,
        )
        response = await gateway.acompletion(
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_prompt},
            ],
        )

        if not response.content:
            logger.warning(f"LLM returned empty response for agent spawn: {gap_description}")
            return None

        proposal = await extractor.extract(response.content, SubAgentProposal)

        if proposal is None:
            # Retry with feedback — send the malformed output back to the LLM
            logger.debug("First parse failed for SubAgentProposal, retrying with feedback")
            error_msg = (
                "Your previous response could not be parsed as valid JSON matching "
                "the SubAgentProposal schema. Please respond with ONLY a valid JSON "
                "object matching the schema provided in the system prompt."
            )
            retry_response = await gateway.acompletion(
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": response.content[:2000]},
                    {"role": "user", "content": error_msg},
                ],
            )
            proposal = await extractor.extract(retry_response.content, SubAgentProposal)

        if proposal is None:
            logger.warning(f"Failed to parse SubAgentProposal for: {gap_description}")
            return None

    except Exception as e:
        logger.warning(f"LLM call failed for agent spawn: {e}")
        return None

    # ── Semantic dedup / recall (B3) — MUST run before name-uniqueness ────
    # Embed the capability (gap + proposal) and reuse an existing ACTIVE agent
    # whose capability is semantically identical (cosine >=
    # capability_dedup_threshold) instead of spawning a duplicate. Only real
    # ("api") embeddings participate; the reused agent must already be in the
    # in-memory registry so delegate can spawn it this run. Best-effort: any
    # failure degrades to validate-and-spawn, never blocks the run.
    #
    # ORDERING IS LOAD-BEARING (battery-04 q2 F-c): reuse MUST precede
    # _validate_proposal's name-uniqueness check. A sub-agent that persisted in
    # a prior run is ACTIVE → loaded into the registry at startup →
    # registry.has(name) is True. When that ran first it appended "already
    # exists" and returned None BEFORE this block, so the recall+reuse path was
    # dead code and re-running such a goal rejected the spawn and rerouted to
    # tool_create (which failed). Dedup-first means a semantically-identical
    # active agent is reused; name-uniqueness then only blocks a GENUINE exact-
    # name collision with an agent that was NOT reused.
    from src.memory.embeddings import embed_capability

    dedup_text = (
        f"{gap_description} | {proposal.description} | {proposal.goal_description}"
    )
    cap_embedding, emb_source = await embed_capability(dedup_text)
    if cap_embedding is not None and emb_source == "api":
        try:
            from src.agents.persister import SubAgentPersister

            threshold = get_settings().agent.capability_dedup_threshold
            similar = await SubAgentPersister().find_similar(
                cap_embedding, threshold=threshold
            )
            for cand in similar:
                existing_spec = registry.get(cand["name"])
                if existing_spec is not None:
                    logger.info(
                        f"Reusing existing sub-agent '{cand['name']}' for "
                        f"gap '{gap_description[:60]}' "
                        f"(similarity={cand['similarity']:.3f}) — skipping spawn"
                    )
                    return {
                        "name": existing_spec.name,
                        "description": existing_spec.description,
                        "template_type": existing_spec.template_type,
                        "tool_scope": existing_spec.tool_scope,
                        "id": existing_spec.id,
                        "reused": True,
                    }
        except Exception as e:
            logger.debug(f"Sub-agent capability dedup skipped: {e}")

    # Validate proposal (name-uniqueness now only blocks a genuine exact-name
    # collision with an agent that dedup did NOT reuse).
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

    # Persist to DB (best-effort, non-blocking). Store the capability embedding
    # (only when a real "api" vector was produced) so future semantically-
    # identical gaps reuse this agent instead of spawning a duplicate (B3).
    await _persist_agent(
        spec,
        capability_embedding=cap_embedding if emb_source == "api" else None,
        capability_text=dedup_text if emb_source == "api" else None,
    )

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


async def _persist_agent(
    spec: SubAgentSpec,
    capability_embedding: list[float] | None = None,
    capability_text: str | None = None,
) -> None:
    """Persist a sub-agent definition to DB (best-effort, non-fatal).

    Args:
        spec: SubAgentSpec to persist.
        capability_embedding: Optional capability vector to store (B3 dedup).
        capability_text: The text the embedding was derived from.
    """
    try:
        from src.agents.persister import SubAgentPersister

        persister = SubAgentPersister()
        agent_id = await persister.persist(
            spec,
            capability_embedding=capability_embedding,
            capability_text=capability_text,
        )
        if agent_id:
            logger.debug(f"Persisted sub-agent '{spec.name}' to DB: {agent_id}")
    except Exception as e:
        logger.debug(f"Sub-agent persistence skipped: {e}")
