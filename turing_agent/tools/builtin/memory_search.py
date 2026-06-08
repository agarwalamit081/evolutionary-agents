"""Memory search tool — queries the 3-tier memory system."""

from __future__ import annotations


from loguru import logger


async def memory_search(query: str, top_k: int = 5) -> str:
    """Search the agent's memory for relevant information.

    Queries across all memory tiers: Redis hot cache, PostgreSQL warm
    memory (skills/procedures), and pgvector cold memory (episodic).

    Args:
        query: Natural language search query.
        top_k: Maximum number of results to return.

    Returns:
        Formatted memory search results.
    """
    logger.info(f"Memory search: {query[:60]}... (top_k={top_k})")

    # Placeholder: in production this would query MemoryManager
    # 1. Check Redis hot cache for recent context
    # 2. Search PostgreSQL warm memory for skills/procedures
    # 3. Run pgvector similarity search on cold memory
    # 4. Merge and rank results

    return (
        f"Memory search for '{query}' returned 0 results.\n"
        f"(Memory system not yet connected — requires MemoryManager integration)"
    )


TOOL_DEFINITION = {
    "name": "memory_search",
    "handler": memory_search,
    "description": (
        "Search the agent's memory system for relevant information. "
        "Queries across all tiers: hot cache (recent context), warm memory "
        "(skills and procedures), and cold memory (episodic knowledge)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language search query.",
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum number of results (default: 5).",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}
