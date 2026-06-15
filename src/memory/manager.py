"""Unified MemoryManager across all 3 tiers."""

from __future__ import annotations

from typing import Any

import redis.asyncio as aioredis
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.settings import Settings
from src.memory.cold import ColdMemory as ColdMemoryStore
from src.memory.embeddings import EmbeddingGenerator
from src.memory.hot import HotMemory as HotMemoryStore
from src.memory.warm import WarmMemoryStore


class MemoryManager:
    """Unified interface across all 3 memory tiers.

    Hot: Redis — ephemeral recent context (TTL-based)
    Warm: PostgreSQL — persistent skills and procedures
    Cold: pgvector — episodic knowledge with semantic search
    """

    def __init__(
        self,
        redis_client: aioredis.Redis,
        db_session: AsyncSession,
        settings: Settings,
    ) -> None:
        # Shared embedding generator (§10.2) — real vectors when a provider key
        # is configured, deterministic hash vectors otherwise.
        embedding_gen = EmbeddingGenerator(settings)
        self.hot = HotMemoryStore(
            redis_client=redis_client,
            ttl_seconds=settings.redis.cache_ttl_seconds,
        )
        self.warm = WarmMemoryStore(session=db_session, generator=embedding_gen)
        self.cold = ColdMemoryStore(
            session=db_session,
            embedding_dim=settings.llm.embedding_dim,
            generator=embedding_gen,
        )
        self._settings = settings
        logger.info("MemoryManager initialized with hot/warm/cold tiers")

    async def store_observation(
        self,
        content: str,
        importance: float = 0.5,
        tags: list[str] | None = None,
        episode_type: str = "execution",
    ) -> None:
        """Store an observation across memory tiers.

        Hot: immediate context
        Cold: long-term episodic memory (embedding TBD)

        Args:
            content: The observation content.
            importance: Importance score (0.0-1.0).
            tags: Context tags.
            episode_type: Episode type classification.
        """
        # Store in hot memory for immediate access
        await self.hot.set(
            key=f"obs:{episode_type}:{hash(content) % 10000}",
            value={"content": content, "tags": tags or []},
            ttl=3600,  # 1 hour
        )

        # Store in cold memory for long-term retrieval
        # Note: embedding generation happens in consolidation
        await self.cold.store(
            episode_type=episode_type,
            content=content,
            importance=importance,
            context_tags=tags,
        )

    async def store_skill(
        self,
        name: str,
        content: str,
        skill_type: str = "procedure",
        tags: list[str] | None = None,
    ) -> str:
        """Store a learned skill in warm memory.

        Args:
            name: Skill name.
            content: Skill content (code, prompt, etc.).
            skill_type: Type (skill, procedure, workflow).
            tags: Categorization tags.

        Returns:
            UUID of the stored skill.
        """
        return await self.warm.store(
            memory_type=skill_type,
            name=name,
            content=content,
            tags=tags,
            fitness_score=0.5,
        )

    async def retrieve_context(
        self,
        query: str,
        tags: list[str] | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant context across all tiers.

        Searches hot cache first, then warm skills, then cold episodic.

        Args:
            query: Natural language query.
            tags: Optional tag filters.
            limit: Maximum results.

        Returns:
            Combined results from all tiers.
        """
        results: list[dict[str, Any]] = []

        # Hot memory: recent observations
        hot_results = await self.hot.search("obs:*")
        for item in hot_results[:2]:
            results.append({"tier": "hot", **item})

        # Warm memory: skills and procedures
        warm_results = await self.warm.retrieve(
            tags=tags,
            min_fitness=0.3,
            limit=3,
        )
        for item in warm_results:
            results.append({"tier": "warm", **item})

        # Cold memory: semantic recall on the query (primary) plus a tag-based
        # filter when tags are given (supplementary). Dedup by id so a memory
        # matching both paths appears once. Semantic recall is best-effort —
        # cold.search_by_query returns [] when no generator is wired.
        seen_cold_ids: set[str] = set()
        if query:
            for item in await self.cold.search_by_query(query=query, limit=3):
                if item.get("id") not in seen_cold_ids:
                    seen_cold_ids.add(item["id"])
                    results.append({"tier": "cold", **item})
        if tags:
            for item in await self.cold.search_by_tags(tags=tags, limit=3):
                if item.get("id") not in seen_cold_ids:
                    seen_cold_ids.add(item["id"])
                    results.append({"tier": "cold", **item})

        return results[:limit]

    async def update_skill_fitness(self, skill_id: str, success: bool) -> None:
        """Update a skill's fitness score after usage.

        Args:
            skill_id: UUID of the skill.
            success: Whether the usage was successful.
        """
        await self.warm.update_fitness(skill_id, success)

    async def consolidate(self) -> dict[str, int]:
        """Run background consolidation across tiers.

        Decays old cold memories, archives stale hot entries.

        Returns:
            Dict with consolidation stats.
        """
        cold_deleted = await self.cold.consolidate(
            max_age_days=90,
            min_importance=0.1,
        )

        return {
            "cold_deleted": cold_deleted,
        }
