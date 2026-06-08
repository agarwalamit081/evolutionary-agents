"""Memory node — retrieve and store memories across 3 tiers."""

from __future__ import annotations

from typing import Any

from loguru import logger

from turing_agent.graph.enums import Phase
from turing_agent.graph.state import AgentState


async def retrieve_memory_node(state: AgentState) -> dict[str, Any]:
    """Retrieve relevant memories for the current goal.

    Queries the memory system for context relevant to the task.
    In production, this uses the MemoryManager with Redis + pgvector.

    Args:
        state: Current agent state.

    Returns:
        Partial state update with retrieved memories.
    """
    goal = state.get("current_goal")
    goal_text = goal.text if goal else ""

    logger.info(f"Retrieving memories for: {goal_text[:60]}...")

    # Placeholder: return empty memories until MemoryManager is integrated
    # In production, this would:
    # 1. Query Redis hot cache for recent context
    # 2. Query PostgreSQL warm memory for relevant skills/procedures
    # 3. Query pgvector cold memory for semantic similarity search
    retrieved: list[dict[str, Any]] = []

    return {
        "phase": Phase.EXECUTE,
        "retrieved_memories": retrieved,
    }


async def store_memory_node(state: AgentState) -> dict[str, Any]:
    """Store execution learnings and observations to memory.

    Persists lessons learned, skill observations, and execution metadata.
    In production, this uses the MemoryManager with consolidation.

    Args:
        state: Current agent state with reflection and observations.

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

    # Placeholder: memory storage happens here in production
    # 1. Store hot observations in Redis with TTL
    # 2. Consolidate important observations to PostgreSQL warm memory
    # 3. Generate embeddings and store in pgvector cold memory
    # 4. Update knowledge graph with new entities/relations

    return {
        "phase": Phase.COMPLETE if is_complete else Phase.HITL_GATE,
    }
