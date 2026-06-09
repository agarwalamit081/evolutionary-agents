"""Cold memory — pgvector-backed episodic knowledge with embeddings."""

from __future__ import annotations

from typing import Any

import sqlalchemy as sa
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import ColdMemory


class ColdMemory:
    """pgvector-backed cold memory for episodic knowledge.

    Stores long-term memories as vector embeddings, enabling
    semantic similarity search for context retrieval.
    """

    def __init__(self, session: AsyncSession, embedding_dim: int = 768) -> None:
        self._session = session
        self._embedding_dim = embedding_dim

    async def store(
        self,
        episode_type: str,
        content: str,
        importance: float = 0.5,
        context_tags: list[str] | None = None,
        embedding: list[float] | None = None,
    ) -> str:
        """Store an episodic memory with optional embedding.

        Args:
            episode_type: Type of episode (execution, reflection, learning, error).
            content: The memory content text.
            importance: Importance score (0.0-1.0).
            context_tags: Tags for filtering.
            embedding: Pre-computed embedding vector.

        Returns:
            UUID of the stored memory.
        """
        import uuid

        memory_id = str(uuid.uuid4())
        entry = ColdMemory(
            id=uuid.UUID(memory_id),
            episode_type=episode_type,
            content=content,
            context_tags=context_tags or [],
            importance=importance,
            embedding=embedding,
        )
        self._session.add(entry)
        await self._session.commit()

        logger.debug(f"Cold memory stored: {episode_type} ({content[:40]}...)")
        return memory_id

    async def search_by_embedding(
        self,
        query_embedding: list[float],
        limit: int = 5,
        min_importance: float = 0.0,
        episode_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search cold memory using vector similarity.

        Args:
            query_embedding: The query vector.
            limit: Maximum results to return.
            min_importance: Minimum importance threshold.
            episode_type: Optional type filter.

        Returns:
            List of similar memories with similarity scores.
        """
        # Use cosine distance via pgvector

        distance = ColdMemory.embedding.cosine_distance(query_embedding)
        query = (
            sa.select(
                ColdMemory,
                distance.label("distance"),
            )
            .where(
                ColdMemory.embedding.isnot(None),
                ColdMemory.importance >= min_importance,
            )
            .order_by(distance)
            .limit(limit)
        )

        if episode_type:
            query = query.where(ColdMemory.episode_type == episode_type)

        result = await self._session.execute(query)
        rows = result.all()

        return [
            {
                "id": str(row[0].id),
                "episode_type": row[0].episode_type,
                "content": row[0].content,
                "importance": row[0].importance,
                "context_tags": row[0].context_tags,
                "similarity": 1.0 - float(row[1]),  # Convert distance to similarity
            }
            for row in rows
        ]

    async def search_by_tags(
        self,
        tags: list[str],
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search cold memory by context tags.

        Args:
            tags: Tags to search for.
            limit: Maximum results.

        Returns:
            List of matching memories.
        """
        # JSONB contains any match
        query = (
            sa.select(ColdMemory)
            .where(ColdMemory.context_tags.bool_op("?|")(tags))
            .order_by(ColdMemory.importance.desc())
            .limit(limit)
        )

        result = await self._session.execute(query)
        entries = result.scalars().all()

        return [
            {
                "id": str(e.id),
                "episode_type": e.episode_type,
                "content": e.content,
                "importance": e.importance,
                "context_tags": e.context_tags,
            }
            for e in entries
        ]

    async def consolidate(
        self,
        max_age_days: int = 90,
        min_importance: float = 0.1,
    ) -> int:
        """Consolidate old memories: reduce importance of stale entries.

        Args:
            max_age_days: Age threshold for consolidation.
            min_importance: Memories below this after decay are deleted.

        Returns:
            Number of memories consolidated (deleted or decayed).
        """
        import datetime as dt

        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=max_age_days)

        # Decay importance of old memories
        result = await self._session.execute(
            sa.select(ColdMemory).where(
                ColdMemory.created_at < cutoff,
                ColdMemory.importance > min_importance,
            )
        )
        old_entries = result.scalars().all()

        deleted = 0
        for entry in old_entries:
            # Decay importance by 50%
            entry.importance *= 0.5
            if entry.importance < min_importance:
                await self._session.delete(entry)
                deleted += 1

        await self._session.commit()
        logger.info(f"Consolidated {len(old_entries)} memories, deleted {deleted}")
        return deleted
