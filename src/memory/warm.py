"""Warm memory — PostgreSQL-backed skills, procedures, and workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import sqlalchemy as sa
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import MemoryEmbedding, WarmMemory

if TYPE_CHECKING:
    from src.memory.embeddings import EmbeddingGenerator


class WarmMemoryStore:
    """PostgreSQL-backed warm memory for persistent skills and procedures.

    Stores skills, procedures, and workflows that have been validated
    and crystallized from execution patterns. Survives restarts.
    """

    def __init__(
        self,
        session: AsyncSession,
        generator: EmbeddingGenerator | None = None,
    ) -> None:
        self._session = session
        self._generator = generator

    async def store(
        self,
        memory_type: str,
        name: str,
        content: str,
        tags: list[str] | None = None,
        fitness_score: float = 0.5,
    ) -> str:
        """Store a skill or procedure in warm memory.

        When a generator is injected, a real embedding is also written to the
        ``memory_embeddings`` table (its FK-correct home —
        ``memory_embeddings.memory_id`` → ``warm_memories.id``), so the table
        the §10.2 review found empty is populated on store.

        Args:
            memory_type: Type of memory (skill, procedure, workflow).
            name: Unique name for the memory entry.
            content: The actual content (code, prompt, etc.).
            tags: Optional tags for categorization.
            fitness_score: Initial fitness score (0.0-1.0).

        Returns:
            The UUID of the created memory entry.
        """
        import uuid

        memory_id = str(uuid.uuid4())
        entry = WarmMemory(
            id=uuid.UUID(memory_id),
            memory_type=memory_type,
            title=name,
            content=content,
            tags=tags or [],
            fitness_score=fitness_score,
            access_count=0,
        )
        self._session.add(entry)

        # Persist an embedding row alongside the warm memory (§10.2). Both rows
        # are committed together; SQLAlchemy inserts the FK parent first.
        if self._generator is not None:
            try:
                embedding = await self._generator.generate(content)
                self._session.add(
                    MemoryEmbedding(
                        memory_id=entry.id,
                        embedding=embedding,
                        embedding_model=self._generator.model,
                    )
                )
            except Exception as e:  # embedding is non-critical; warm store must not fail
                logger.debug(f"Warm memory embedding skipped: {e}")

        await self._session.commit()

        logger.info(f"Warm memory stored: {memory_type}/{name} (id={memory_id[:8]})")
        return memory_id

    async def retrieve(
        self,
        memory_type: str | None = None,
        name: str | None = None,
        tags: list[str] | None = None,
        min_fitness: float = 0.0,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Retrieve skills/procedures from warm memory.

        Args:
            memory_type: Filter by type (skill, procedure, workflow).
            name: Filter by name (exact match).
            tags: Filter by tags (any match).
            min_fitness: Minimum fitness score threshold.
            limit: Maximum results to return.

        Returns:
            List of matching memory entries as dicts.
        """
        query = sa.select(WarmMemory).where(
            WarmMemory.fitness_score >= min_fitness,
            WarmMemory.expires_at.is_(None),
        ).order_by(WarmMemory.fitness_score.desc()).limit(limit)

        if memory_type:
            query = query.where(WarmMemory.memory_type == memory_type)
        if name:
            query = query.where(WarmMemory.title == name)
        if tags:
            # Match entries that contain ANY of the requested tags (JSONB overlap)
            query = query.where(WarmMemory.tags.bool_op("?|")(tags))

        result = await self._session.execute(query)
        entries = result.scalars().all()

        return [
            {
                "id": str(entry.id),
                "type": entry.memory_type,
                "name": entry.title,
                "content": entry.content,
                "tags": entry.tags,
                "fitness_score": entry.fitness_score,
                "access_count": entry.access_count,
            }
            for entry in entries
        ]

    async def update_fitness(self, memory_id: str, success: bool) -> None:
        """Update fitness score after a usage event.

        Args:
            memory_id: UUID of the memory entry.
            success: Whether the usage was successful.
        """
        import uuid

        result = await self._session.execute(
            sa.select(WarmMemory).where(WarmMemory.id == uuid.UUID(memory_id))
        )
        entry = result.scalar_one_or_none()
        if not entry:
            return

        entry.access_count += 1

        # Recalculate fitness using exponential moving average
        # Weight successful uses more heavily than total access count
        if entry.access_count > 0:
            adjustment = 0.1 if success else -0.05
            entry.fitness_score = max(0.0, min(1.0, entry.fitness_score + adjustment))

        await self._session.commit()
        logger.debug(f"Fitness updated for {memory_id[:8]}: {entry.fitness_score:.3f}")
