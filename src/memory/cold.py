"""Cold memory — pgvector-backed episodic knowledge with embeddings."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import ColdMemory as ColdMemoryModel

if TYPE_CHECKING:
    from src.memory.embeddings import EmbeddingGenerator


def _tags_any_predicate(tags: Sequence[str]) -> sa.ColumnElement[bool]:
    """Build a JSONB "any-of" predicate over ``ColdMemory.context_tags``.

    True when the episode's ``context_tags`` array contains any of ``tags``.
    Implemented with the JSONB existence operator ``?`` OR'd per tag rather
    than the ``?|`` (any-of-array) operator: SQLAlchemy binds a Python list as
    jsonb and Postgres only defines ``jsonb ?| text[]`` — there is no
    ``jsonb ?| jsonb`` operator, so ``context_tags.bool_op("?|")(list)`` raises
    ``operator does not exist: jsonb ?| jsonb`` at execution time (asyncpg).
    Each ``?`` binds a single text param, sidestepping the mismatch. Empty
    ``tags`` yields ``false`` (matches no rows), matching the prior empty-array
    ``?|`` semantics. Parameterized ORM — no interpolated SQL.
    """
    if not tags:
        return sa.false()
    return sa.or_(
        *(ColdMemoryModel.context_tags.bool_op("?")(tag) for tag in tags)
    )


class ColdMemory:
    """pgvector-backed cold memory for episodic knowledge.

    Stores long-term memories as vector embeddings, enabling
    semantic similarity search for context retrieval.
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
        episode_type: str,
        content: str,
        importance: float = 0.5,
        context_tags: list[str] | None = None,
        embedding: list[float] | None = None,
    ) -> str:
        """Store an episodic memory with optional embedding.

        When no ``embedding`` is passed but a generator was injected, one is
        generated from ``content`` so the ``ColdMemory.embedding`` column is
        populated and semantic recall (``search_by_embedding``) works (§10.2).

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

        if embedding is None and self._generator is not None:
            try:
                embedding = await self._generator.generate(content)
            except Exception as e:  # embedding is non-critical; cold store must not fail
                logger.debug(f"Cold memory embedding generation failed: {e}")
                embedding = None

        # Q81 — store-time near-duplicate merge (opt-in). Before inserting, look
        # for an existing episode within the dedup cosine-similarity threshold; if
        # found, skip the insert and return the existing id so the cold tier
        # doesn't accumulate near-identical episodes. Default off
        # (MEMORY_DEDUP_ENABLED); skipped when no embedding is available.
        if embedding is not None:
            from src.config import get_settings  # noqa: PLC0415

            ms = get_settings().memory
            if ms.dedup_enabled:
                existing_id = await self._find_similar_episode(
                    query_embedding=embedding, threshold=ms.dedup_threshold
                )
                if existing_id is not None:
                    logger.info(
                        f"Cold memory dedup hit → existing {existing_id[:8]} "
                        f"(>= {ms.dedup_threshold}); skipped insert"
                    )
                    return existing_id

        memory_id = str(uuid.uuid4())
        entry = ColdMemoryModel(
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

    async def _find_similar_episode(
        self,
        *,
        query_embedding: list[float],
        threshold: float,
    ) -> str | None:
        """Return the id of an existing episode within ``threshold`` similarity.

        Q81 dedup helper. Cosine distance is converted to similarity
        (``1 - distance``); an existing episode at ``>= threshold`` similarity
        means the incoming episode is a near-duplicate. Returns the most-similar
        match's id, else ``None``. Parameterized ORM query (pgvector ``<=>``),
        no interpolation.
        """
        max_distance = 1.0 - threshold
        distance = ColdMemoryModel.embedding.cosine_distance(query_embedding)
        stmt = (
            sa.select(ColdMemoryModel.id)
            .where(
                ColdMemoryModel.embedding.isnot(None),
                distance <= max_distance,
            )
            .order_by(distance)
            .limit(1)
        )
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()
        return str(row) if row is not None else None

    async def search_by_embedding(
        self,
        query_embedding: list[float],
        limit: int = 5,
        min_importance: float = 0.0,
        episode_type: str | None = None,
        min_similarity: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Search cold memory using vector similarity.

        Args:
            query_embedding: The query vector.
            limit: Maximum results to return.
            min_importance: Minimum importance threshold.
            episode_type: Optional type filter.
            min_similarity: Drop results whose cosine similarity
                (``1 - distance``) is below this. Default ``0.0`` = keep all
                (Q82 recall threshold; complementary to per-tier ranking).

        Returns:
            List of similar memories with similarity scores.
        """
        # Use cosine distance via pgvector
        distance = ColdMemoryModel.embedding.cosine_distance(query_embedding)
        query = (
            sa.select(
                ColdMemoryModel,
                distance.label("distance"),
            )
            .where(
                ColdMemoryModel.embedding.isnot(None),
                ColdMemoryModel.importance >= min_importance,
            )
            .order_by(distance)
            .limit(limit)
        )

        if episode_type:
            query = query.where(ColdMemoryModel.episode_type == episode_type)

        result = await self._session.execute(query)
        rows = result.all()

        # Q82 — post-query relevance filter. Rows are ordered by distance asc
        # (similarity desc), so dropping items below ``min_similarity`` keeps a
        # contiguous high-quality prefix. DB-agnostic (pgvector distance already
        # selected). Default 0.0 keeps everything.
        return [
            {
                "id": str(row[0].id),
                "episode_type": row[0].episode_type,
                "content": row[0].content,
                "importance": row[0].importance,
                "context_tags": row[0].context_tags,
                "similarity": 1.0 - float(row[1]),
            }
            for row in rows
            if (1.0 - float(row[1])) >= min_similarity
        ]

    async def search_by_query(
        self,
        query: str,
        limit: int = 5,
        min_importance: float = 0.0,
        episode_type: str | None = None,
        min_similarity: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Semantic search: embed a text query and rank memories by similarity.

        Thin wrapper over :meth:`search_by_embedding` that owns query embedding,
        so callers (e.g. ``MemoryManager.retrieve_context``) need no separate
        generator reference. Semantic recall is best-effort: returns ``[]`` when
        no generator is wired, the query is empty, or embedding fails — it must
        never block context retrieval.

        Args:
            query: Natural-language query text.
            limit: Maximum results to return.
            min_importance: Minimum importance threshold.
            episode_type: Optional type filter.
            min_similarity: Drop results below this cosine similarity (Q82).
                Default ``0.0`` = keep all.

        Returns:
            List of similar memories with similarity scores.
        """
        if not query or self._generator is None:
            return []
        try:
            query_embedding = await self._generator.generate(query)
        except Exception as e:  # embedding failure must not break retrieval
            logger.debug(f"Cold memory query embedding failed: {e}")
            return []
        return await self.search_by_embedding(
            query_embedding=query_embedding,
            limit=limit,
            min_importance=min_importance,
            episode_type=episode_type,
            min_similarity=min_similarity,
        )

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
        query = (
            sa.select(ColdMemoryModel)
            .where(_tags_any_predicate(tags))
            .order_by(ColdMemoryModel.importance.desc())
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
            sa.select(ColdMemoryModel).where(
                ColdMemoryModel.created_at < cutoff,
                ColdMemoryModel.importance > min_importance,
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
