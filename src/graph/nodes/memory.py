"""Memory node — retrieve and store memories across 3 tiers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from src.graph.enums import Phase
from src.graph.state import AgentState

if TYPE_CHECKING:
    from src.llm.gateway import LLMGateway
    from src.memory.manager import MemoryManager


async def retrieve_memory_node(
    state: AgentState,
    *,
    memory: MemoryManager | None = None,
) -> dict[str, Any]:
    """Retrieve relevant memories for the current goal.

    When a MemoryManager is provided, queries all 3 tiers (Redis hot,
    PostgreSQL warm, pgvector cold) for context relevant to the task.

    Args:
        state: Current agent state.
        memory: Optional MemoryManager for 3-tier memory queries.

    Returns:
        Partial state update with retrieved memories.
    """
    goal = state.get("current_goal")
    goal_text = goal.text if goal else ""

    logger.info(f"Retrieving memories for: {goal_text[:60]}...")

    retrieved: list[dict[str, Any]] = []

    if memory is not None:
        try:
            results = await memory.retrieve_context(query=goal_text, limit=5)
            if results:
                retrieved = [
                    {"content": r.get("content", ""), "tier": r.get("tier", ""), "score": r.get("score", 0.0)}
                    for r in results
                    if isinstance(r, dict) and "content" in r
                ]
                # Fallback: results might be plain objects
                if not retrieved:
                    retrieved = [
                        {"content": str(r)}
                        for r in results[:5]
                    ]
                logger.info(f"Retrieved {len(retrieved)} memories from 3-tier system")
        except Exception as e:
            logger.warning(f"Memory retrieval failed: {e}")

        # Load evolved prompts from warm memory (crystallized by evolution)
        try:
            evolved = await memory.warm.retrieve(
                memory_type="evolved_prompt",
                min_fitness=0.5,
                limit=3,
            )
            for entry in evolved:
                retrieved.append({
                    "content": entry.get("content", ""),
                    "tier": "evolved",
                    "score": entry.get("fitness_score", 0.6),
                })
            if evolved:
                logger.info(f"Loaded {len(evolved)} evolved prompt(s) from warm memory")
        except Exception as e:
            logger.debug(f"Evolved prompt loading skipped: {e}")

        # Recall folded-memory summaries persisted by earlier runs. Each fold
        # stores compact episode/working/tool JSON as warm memory so later
        # runs can reuse compressed context instead of re-deriving it.
        try:
            folded = await memory.warm.retrieve(
                memory_type="folded_memory",
                min_fitness=0.5,
                limit=3,
            )
            for entry in folded:
                retrieved.append({
                    "content": entry.get("content", ""),
                    "tier": "folded",
                    "score": entry.get("fitness_score", 0.5),
                })
            if folded:
                logger.info(
                    f"Loaded {len(folded)} folded memory summary/summaries from warm memory"
                )
        except Exception as e:
            logger.debug(f"Folded memory loading skipped: {e}")

        # Recall durable facts (Phase 5). Facts are entity-ish knowledge mined
        # from prior folds/runs — distinct from skills and from episodic cold
        # memory — and ranked semantically against the goal when possible.
        try:
            facts = await memory.retrieve_facts(query=goal_text, limit=3)
            for entry in facts:
                value = entry.get("value", "")
                key = entry.get("key", "")
                content = f"{key}: {value}" if key else value
                retrieved.append({
                    "content": content,
                    "tier": "fact",
                    "score": entry.get("confidence", 0.5),
                })
            if facts:
                logger.info(f"Loaded {len(facts)} fact(s) from the semantic tier")
        except Exception as e:
            logger.debug(f"Fact recall skipped: {e}")
    else:
        logger.debug("No MemoryManager available, returning empty memories")

    return {
        "phase": Phase.EXECUTE,
        "retrieved_memories": retrieved,
    }


async def store_memory_node(
    state: AgentState,
    *,
    memory: MemoryManager | None = None,
    gateway: LLMGateway | None = None,
) -> dict[str, Any]:
    """Store execution learnings and observations to memory.

    When a MemoryManager is provided, persists observations as hot
    memories and lessons learned as warm memories.

    Args:
        state: Current agent state with reflection and observations.
        memory: Optional MemoryManager for 3-tier memory storage.

    Returns:
        Partial state update confirming storage.
    """
    reflection = state.get("reflection")
    memory_observations = state.get("memory_observations", [])
    is_complete = state.get("is_complete", False)

    observations_count = len(memory_observations)
    lessons_count = len(reflection.lessons_learned) if reflection else 0

    logger.info(
        f"Storing memory: {observations_count} observations, "
        f"{lessons_count} lessons learned"
    )

    if memory is not None:
        stored_count = 0

        # Store each observation as hot memory
        for obs in memory_observations:
            try:
                await memory.store_observation(
                    content=obs,
                    tags=["reflection", "complete" if is_complete else "incomplete"],
                )
                stored_count += 1
            except Exception as e:
                logger.warning(f"Failed to store observation: {e}")

        # Store lessons learned as warm memory skills
        if reflection and reflection.lessons_learned:
            try:
                await memory.store_skill(
                    name=f"lesson_{stored_count}",
                    content="; ".join(reflection.lessons_learned),
                    tags=["lesson", str(state.get("current_goal", ""))[:50]],
                )
            except Exception as e:
                logger.warning(f"Failed to store lessons: {e}")

        logger.info(f"Stored {stored_count}/{observations_count} observations to memory")
    else:
        logger.debug("No MemoryManager available, skipping memory storage")

    result: dict[str, Any] = {
        "phase": Phase.COMPLETE if is_complete else Phase.HITL_GATE,
    }

    # Flush accumulated LLM cost/token records into graph state. This node is
    # reached on every terminating path (complete / partial-accepted /
    # evolve→store), so it is the single sink that populates cost_records and
    # total_tokens_used for run-history, eval, and report consumers. No other
    # node writes these fields, so the operator.add reducer sees one append.
    if gateway is not None:
        cost_records = gateway.get_cost_records()
        if cost_records:
            result["cost_records"] = cost_records
            result["total_tokens_used"] = sum(
                r.input_tokens + r.output_tokens for r in cost_records
            )

    return result
