"""Warm memory — PostgreSQL-backed skills, procedures, and workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import sqlalchemy as sa
from loguru import logger
from sqlalchemy.dialects import postgresql
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
        extra_data: dict[str, Any] | None = None,
        embed_text: str | None = None,
    ) -> str:
        """Store a skill or procedure in warm memory.

        When a generator is injected, a real embedding is also written to the
        ``memory_embeddings`` table (its FK-correct home —
        ``memory_embeddings.memory_id`` → ``warm_memories.id``), so the table
        the §10.2 review found empty is populated on store.

        Args:
            memory_type: Type of memory (skill, procedure, workflow, fact).
            name: Unique name for the memory entry.
            content: The actual content (code, prompt, etc.).
            tags: Optional tags for categorization.
            fitness_score: Initial fitness score (0.0-1.0).
            extra_data: Optional structured payload stored in the JSONB
                ``extra_data`` column (e.g. fact source/confidence). Defaults to
                an empty dict.
            embed_text: Optional override text used to build the embedding;
                defaults to ``content``. Facts embed ``"<key>: <value>"`` for
                better semantic recall than the bare value.

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
            extra_data=extra_data or {},
            access_count=0,
        )
        self._session.add(entry)

        # Persist an embedding row alongside the warm memory (§10.2). Both rows
        # are committed together; SQLAlchemy inserts the FK parent first.
        if self._generator is not None:
            try:
                embedding = await self._generator.generate(embed_text or content)
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

    async def store_fact(
        self,
        key: str,
        value: str,
        *,
        source: str = "extraction",
        confidence: float = 0.5,
        tags: list[str] | None = None,
    ) -> str:
        """Store a durable fact in the semantic/fact tier, deduped on ``fact_key``.

        Facts are entity-ish knowledge (``memory_type="fact"``) distinct from
        skills/procedures. A real ``fact_key`` column + partial unique index backs
        an ``ON CONFLICT (fact_key) DO UPDATE`` upsert, so re-running a goal
        updates the existing fact rather than duplicating it — the race-free form
        (concurrent memory folds can both extract the same fact). The surviving
        row's id is RETURNING-ed so its embedding is refreshed in place; source,
        confidence, content, tags, and fitness all update.

        The dedup index is partial (``memory_type='fact' AND expires_at IS NULL``)
        so a retired fact never shadows a freshly-extracted one (``retrieve_facts``
        already filters ``expires_at IS NULL``). Non-fact rows carry NULL
        ``fact_key`` (NULLs are distinct) and never collide.

        Args:
            key: Short stable identifier / entity name (== fact_key).
            value: The durable fact.
            source: Provenance label (e.g. "fold_2_episode").
            confidence: Extraction confidence 0.0-1.0 (also the fitness seed).
            tags: Extra tags; "fact" is always prepended.

        Returns:
            The UUID of the surviving (inserted-or-updated) fact.
        """
        import uuid

        clamped = max(0.0, min(1.0, float(confidence)))
        tags_list = ["fact", *(tags or [])]
        extra_data: dict[str, Any] = {
            "source": source,
            "confidence": clamped,
            "fact_key": key,
        }
        new_id = uuid.uuid4()

        # Upsert on fact_key. set_ omits id/title/fact_key so the surviving row
        # keeps its identity; content/provenance/fitness/tags/updated_at refresh.
        stmt = (
            postgresql.insert(WarmMemory)
            .values(
                id=new_id,
                memory_type="fact",
                title=key,
                content=value,
                tags=tags_list,
                fitness_score=clamped,
                extra_data=extra_data,
                fact_key=key,
                access_count=0,
            )
            .on_conflict_do_update(
                index_elements=["fact_key"],
                index_where=sa.text("memory_type = 'fact' AND expires_at IS NULL"),
                set_={
                    "content": value,
                    "fitness_score": clamped,
                    "extra_data": extra_data,
                    "tags": tags_list,
                    "updated_at": sa.func.now(),
                },
            )
            .returning(WarmMemory.id)
        )
        result = await self._session.execute(stmt)
        memory_id: uuid.UUID = result.scalar_one()

        # Refresh the embedding on the surviving row (delete-then-add). Best-effort
        # — an embedding failure must never block the fact write (mirrors store()).
        if self._generator is not None:
            try:
                embedding = await self._generator.generate(f"{key}: {value}")
                await self._session.execute(
                    sa.delete(MemoryEmbedding).where(
                        MemoryEmbedding.memory_id == memory_id
                    )
                )
                self._session.add(
                    MemoryEmbedding(
                        memory_id=memory_id,
                        embedding=embedding,
                        embedding_model=self._generator.model,
                    )
                )
            except Exception as e:  # embedding is non-critical; fact store must not fail
                logger.debug(f"Fact embedding skipped: {e}")

        await self._session.commit()
        logger.info(f"Fact stored (upserted): {key} (id={str(memory_id)[:8]})")
        return str(memory_id)

    async def retrieve_facts(
        self,
        query: str = "",
        limit: int = 5,
        min_confidence: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Recall durable facts, ranked by semantic similarity when possible.

        When a query and an injected generator are available, ranks facts by
        cosine distance against their ``memory_embeddings`` vectors (semantic
        recall). Otherwise falls back to fitness-ordered retrieval. Facts are
        strictly ``memory_type="fact"`` — never skills/procedures.

        Args:
            query: Natural-language query; empty → fitness-ordered fallback.
            limit: Maximum facts to return.
            min_confidence: Minimum fitness_score (== extraction confidence).

        Returns:
            List of fact dicts: ``{id, key, value, source, confidence,
            similarity?}``.
        """
        # Semantic path: embed the query and rank facts by vector similarity.
        # Requires a generator (so the query embeds) AND query text. Best-effort
        # — an embedding failure drops to the fitness fallback below.
        if query and self._generator is not None:
            try:
                query_embedding = await self._generator.generate(query)
            except Exception as e:  # embedding failure must not block recall
                logger.debug(f"Fact query embedding failed: {e}")
            else:
                distance = MemoryEmbedding.embedding.cosine_distance(query_embedding)
                stmt = (
                    sa.select(WarmMemory, distance.label("distance"))
                    .join(
                        MemoryEmbedding,
                        MemoryEmbedding.memory_id == WarmMemory.id,
                    )
                    .where(
                        WarmMemory.memory_type == "fact",
                        WarmMemory.fitness_score >= min_confidence,
                        WarmMemory.expires_at.is_(None),
                        MemoryEmbedding.embedding.isnot(None),
                    )
                    .order_by(distance)
                    .limit(limit)
                )
                result = await self._session.execute(stmt)
                rows = result.all()
                if rows:
                    return [
                        {
                            "id": str(row[0].id),
                            "key": row[0].title,
                            "value": row[0].content,
                            "source": row[0].extra_data.get("source", "extraction"),
                            "confidence": row[0].fitness_score,
                            "similarity": 1.0 - float(row[1]),
                        }
                        for row in rows
                    ]
                # No embedded facts matched — fall through to fitness fallback.

        # Fallback: fitness-ordered facts (no generator, empty query, or no
        # embedded facts). Reuses the generic retrieve() filter set.
        return [
            {
                "id": e["id"],
                "key": e["name"],
                "value": e["content"],
                "source": "extraction",
                "confidence": e["fitness_score"],
            }
            for e in await self.retrieve(
                memory_type="fact",
                min_fitness=min_confidence,
                limit=limit,
            )
        ]

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

    async def retrieve_skills(
        self,
        query: str = "",
        limit: int = 5,
        min_fitness: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Recall skills/procedures/workflows, ranked semantically when possible.

        Mirrors :meth:`retrieve_facts` but spans the three capability-shaped
        memory types (skill/procedure/workflow). When a query and an injected
        generator are available, ranks by cosine distance against their
        ``memory_embeddings`` vectors (semantic recall — the recall layer
        findings-05 found dead: ``capability_embedding`` and the warm embedding
        index fed dedup/curation only, never ranked recall). Otherwise falls
        back to fitness-ordered retrieval across the same three types.

        Args:
            query: Natural-language query; empty → fitness-ordered fallback.
            limit: Maximum skills to return.
            min_fitness: Minimum fitness_score threshold.

        Returns:
            List of skill dicts: ``{id, type, name, content, tags,
            fitness_score, access_count, similarity?}``.
        """
        skill_types = ["skill", "procedure", "workflow"]

        # Semantic path: embed the query and rank skills by vector similarity.
        # Requires a generator (so the query embeds) AND query text. Best-effort
        # — an embedding failure drops to the fitness fallback below.
        if query and self._generator is not None:
            try:
                query_embedding = await self._generator.generate(query)
            except Exception as e:  # embedding failure must not block recall
                logger.debug(f"Skill query embedding failed: {e}")
            else:
                distance = MemoryEmbedding.embedding.cosine_distance(query_embedding)
                stmt = (
                    sa.select(WarmMemory, distance.label("distance"))
                    .join(
                        MemoryEmbedding,
                        MemoryEmbedding.memory_id == WarmMemory.id,
                    )
                    .where(
                        WarmMemory.memory_type.in_(skill_types),
                        WarmMemory.fitness_score >= min_fitness,
                        WarmMemory.expires_at.is_(None),
                        MemoryEmbedding.embedding.isnot(None),
                    )
                    .order_by(distance)
                    .limit(limit)
                )
                result = await self._session.execute(stmt)
                rows = result.all()
                if rows:
                    return [
                        {
                            "id": str(row[0].id),
                            "type": row[0].memory_type,
                            "name": row[0].title,
                            "content": row[0].content,
                            "tags": row[0].tags,
                            "fitness_score": row[0].fitness_score,
                            "access_count": row[0].access_count,
                            "similarity": 1.0 - float(row[1]),
                        }
                        for row in rows
                    ]
                    # No embedded skills matched — fall through to fitness fallback.

        # Fallback: fitness-ordered skills (no generator, empty query, or no
        # embedded skills). ``retrieve()`` takes a single memory_type, so query
        # across all three capability types in one fitness-ordered pass.
        fallback = (
            sa.select(WarmMemory)
            .where(
                WarmMemory.memory_type.in_(skill_types),
                WarmMemory.fitness_score >= min_fitness,
                WarmMemory.expires_at.is_(None),
            )
            .order_by(WarmMemory.fitness_score.desc())
            .limit(limit)
        )
        result = await self._session.execute(fallback)
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
