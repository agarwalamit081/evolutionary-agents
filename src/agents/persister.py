"""DB persistence for sub-agent definitions and run records.

Stores SubAgentSpec definitions in SubAgentModel table and individual
execution records in SubAgentRunModel. Loads active sub-agents at
startup into SubAgentRegistry.

Follows the same pattern as src/tools/dynamic/persister.py (ToolPersister).
"""

from __future__ import annotations

import math
import uuid
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from src.agents.registry import SubAgentRegistry
    from src.config.settings import AgentSettings
    from src.graph.models import SubAgentSpec


class SubAgentPersister:
    """Persist and load sub-agent definitions from the database."""

    async def persist(
        self,
        spec: SubAgentSpec,
        capability_embedding: list[float] | None = None,
        capability_text: str | None = None,
    ) -> uuid.UUID | None:
        """Write or update a SubAgentModel in the database.

        If a sub-agent with the same name already exists, increments
        the version and deactivates the old row. Otherwise creates a
        new row with version 1.

        Args:
            spec: SubAgentSpec to persist.
            capability_embedding: Optional 768-d capability vector (B3 dedup),
                stored so future gaps can reuse this agent via
                :meth:`find_similar`. Pass None when no real embedding is
                available.
            capability_text: The text ``capability_embedding`` was derived from.

        Returns:
            UUID of the SubAgentModel row, or None on failure.
        """
        try:
            from sqlalchemy import select, update

            from src.db.models import SubAgentModel
            from src.db.session import get_session

            async with get_session() as session:
                # Check for existing agent with same name
                stmt = select(SubAgentModel).where(
                    SubAgentModel.name == spec.name
                )
                result = await session.execute(stmt)
                existing = result.scalar_one_or_none()

                if existing is not None:
                    # Increment version on existing
                    new_version = existing.version + 1
                    await session.execute(
                        update(SubAgentModel)
                        .where(SubAgentModel.id == existing.id)
                        .values(
                            is_active=False,
                            updated_at=_utcnow(),
                        )
                    )
                    await session.flush()

                    # Create new version row
                    model = _spec_to_model(
                        spec,
                        version=new_version,
                        capability_embedding=capability_embedding,
                        capability_text=capability_text,
                    )
                    session.add(model)
                    await session.flush()

                    logger.info(
                        f"Updated sub-agent '{spec.name}' to version {new_version}"
                    )
                    await self._sync_subagent_graph(spec)
                    return model.id

                # Create new sub-agent
                model = _spec_to_model(
                    spec,
                    version=1,
                    capability_embedding=capability_embedding,
                    capability_text=capability_text,
                )
                session.add(model)
                await session.flush()

                logger.info(f"Persisted new sub-agent '{spec.name}' (version 1)")
                await self._sync_subagent_graph(spec)
                return model.id

        except Exception as e:
            logger.warning(f"Failed to persist sub-agent '{spec.name}': {e}")
            return None

    async def _sync_subagent_graph(self, spec: SubAgentSpec) -> None:
        """Mirror a persisted sub-agent def into the Neo4j graph (I3).

        Best-effort structured sync — no extraction. Lazy ``get_settings()``
        + a fresh :class:`Neo4jGraph` per call (sub-agent creation is rare,
        ≤3/run) so a missing package / unreachable Neo4j / any driver error is
        caught inside the store and never re-raises. Default-off
        (``GRAPH_ENABLED=False`` ⇒ no-op). Called AFTER the DB row is written,
        so a graph hiccup can never abort the persistence.
        """
        try:
            from src.config.settings import get_settings
            from src.memory.graph import Neo4jGraph

            neo4j_settings = get_settings().neo4j
            if not neo4j_settings.enabled:
                return
            tier = (
                spec.model_tier.value
                if hasattr(spec.model_tier, "value")
                else str(spec.model_tier)
            )
            graph = Neo4jGraph(neo4j_settings)
            try:
                await graph.sync_subagent(
                    spec.name,
                    spec.description,
                    tool_scope=list(spec.tool_subset),
                    model_tier=tier,
                )
            finally:
                await graph.close()
        except Exception as exc:  # noqa: BLE001 — non-fatal observability-only
            logger.debug(f"Sub-agent graph sync skipped for '{spec.name}': {exc}")

    async def find_similar(
        self,
        embedding: list[float],
        threshold: float = 0.85,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Cosine-search active sub-agent capability embeddings (B3 dedup).

        Mirrors :meth:`src.tools.dynamic.persister.ToolPersister.find_similar`.
        Returns agents at/above ``threshold``, most-similar first, as
        ``[{"name", "description", "similarity"}]``. Best-effort: ``[]`` on DB
        error so dedup degrades to "spawn" instead of blocking.
        """
        try:
            from sqlalchemy import select

            from src.db.models import SubAgentModel
            from src.db.session import get_session

            async with get_session() as session:
                distance = SubAgentModel.capability_embedding.cosine_distance(
                    embedding
                )
                stmt = (
                    select(
                        SubAgentModel.name,
                        SubAgentModel.description,
                        distance.label("distance"),
                    )
                    .where(
                        SubAgentModel.capability_embedding.isnot(None),
                        SubAgentModel.is_active.is_(True),
                    )
                    .order_by(distance)
                    .limit(limit)
                )
                result = await session.execute(stmt)
                matches: list[dict[str, Any]] = []
                for name, desc, dist in result.all():
                    similarity = 1.0 - float(dist)
                    if similarity >= threshold:
                        matches.append(
                            {
                                "name": name,
                                "description": desc,
                                "similarity": similarity,
                            }
                        )
                return matches
        except Exception as e:
            logger.debug(f"Sub-agent capability find_similar failed: {e}")
            return []

    async def retrieve_agents_with_scores(
        self,
        names: list[str],
        embedding: list[float],
        limit: int = 8,
    ) -> list[tuple[str, float]]:
        """RECALL similarities over a NAMED subset (F1): top-limit ``(name, cosine)``.

        Unlike :meth:`find_similar` (dedup gate: threshold ≥ 0.85, over ALL
        active agents), this RANKS only the named spawned agents by cosine to a
        subtask embedding, at threshold 0.0 (recall, not dedup) — so the F1
        selection layer can keep the most-relevant spawned agents and prune the
        fan-out. Best-effort: ``[]`` on any DB error (the caller falls back to
        all-spawned). ``names`` may include unknown/retired entries — they are
        simply absent from the result (``name.in_(names)`` + active filter).
        """
        if not names:
            return []
        try:
            from sqlalchemy import select

            from src.db.models import SubAgentModel
            from src.db.session import get_session

            async with get_session() as session:
                distance = SubAgentModel.capability_embedding.cosine_distance(
                    embedding
                )
                stmt = (
                    select(
                        SubAgentModel.name,
                        distance.label("distance"),
                    )
                    .where(
                        SubAgentModel.capability_embedding.isnot(None),
                        SubAgentModel.is_active.is_(True),
                        SubAgentModel.name.in_(names),
                    )
                    .order_by(distance)
                    .limit(limit)
                )
                result = await session.execute(stmt)
                return [(name, 1.0 - float(dist)) for name, dist in result.all()]
        except Exception as e:
            logger.debug(f"retrieve_agents_with_scores failed: {e}")
            return []

    async def retire(self, names: list[str]) -> int:
        """Mark named sub-agents ``is_active=False`` in the DB.

        Used by :meth:`load_active_agents` to persist in-memory retirements
        decided by ``SubAgentRegistry.enforce_caps`` / :meth:`retire_redundant`,
        so a capability retired on one run is not reloaded on the next.

        Returns:
            Number of agents retired (best-effort; logs on DB error).
        """
        if not names:
            return 0
        try:
            from sqlalchemy import update

            from src.db.models import SubAgentModel
            from src.db.session import get_session

            async with get_session() as session:
                await session.execute(
                    update(SubAgentModel)
                    .where(SubAgentModel.name.in_(names))
                    .values(is_active=False, updated_at=_utcnow())
                )
            logger.info(f"Retired {len(names)} sub-agents: {', '.join(names)}")
            return len(names)
        except Exception as e:
            logger.warning(f"Failed to retire sub-agents {names}: {e}")
            return 0

    async def _active_capability_rows(self) -> list[dict[str, Any]]:
        """Fetch active sub-agents' capability vectors + scoring signals.

        Powers :meth:`retire_redundant`'s pairwise cosine without N DB
        round-trips. Each row: ``{"name", "embedding" (list|None),
        "success_rate", "total_runs", "quality_score"}``.
        """
        from sqlalchemy import select

        from src.db.models import SubAgentModel
        from src.db.session import get_session

        rows: list[dict[str, Any]] = []
        async with get_session() as session:
            stmt = select(
                SubAgentModel.name,
                SubAgentModel.capability_embedding,
                SubAgentModel.success_rate,
                SubAgentModel.total_runs,
                SubAgentModel.quality_score,
            ).where(SubAgentModel.is_active.is_(True))
            result = await session.execute(stmt)
            for name, emb, sr, tr, qs in result.all():
                vector = (
                    [float(x) for x in emb] if emb is not None else None
                )
                sr_f = float(sr or 0.0)
                tr_i = int(tr or 0)
                qs_f = float(qs or 0.5)
                rows.append(
                    {
                        "name": name,
                        "embedding": vector,
                        "success_rate": sr_f,
                        "total_runs": tr_i,
                        "quality_score": qs_f,
                        # Self-describing sort key so callers (retire_redundant,
                        # governance consolidation) stay shape-agnostic.
                        "score": (sr_f, tr_i, qs_f),
                    }
                )
        return rows

    async def retire_redundant(self, threshold: float) -> list[str]:
        """Retire semantically-duplicate active sub-agents (B3 de-bloat).

        Loads every active agent's capability embedding, then retires the
        lower-scoring twin of any pair whose cosine similarity >= ``threshold``
        (the stricter consolidation cutoff, distinct from the creation-time
        ``capability_dedup_threshold``). Higher ``(success_rate, total_runs,
        quality_score)`` tuple wins. Pure-Python pairwise cosine over the
        fetched vectors — the active set is small (<100), so this avoids N DB
        round-trips. Best-effort: any error degrades to no retirement.

        Returns:
            Names of agents retired (sorted).
        """
        try:
            rows = await self._active_capability_rows()
        except Exception as e:
            logger.debug(f"Sub-agent redundancy scan failed: {e}")
            return []

        retired: set[str] = set()
        for i, a in enumerate(rows):
            if a["name"] in retired or a["embedding"] is None:
                continue
            for b in rows[i + 1:]:
                if b["name"] in retired or b["embedding"] is None:
                    continue
                sim = _cosine(a["embedding"], b["embedding"])
                if sim >= threshold:
                    loser = (
                        a["name"]
                        if a["score"] < b["score"]
                        else b["name"]
                    )
                    retired.add(loser)
        if retired:
            await self.retire(sorted(retired))
            logger.info(
                f"Retired {len(retired)} redundant sub-agents "
                f"(threshold={threshold}): {', '.join(sorted(retired))}"
            )
        return sorted(retired)

    async def load_active_agents(
        self,
        registry: SubAgentRegistry,
        settings: AgentSettings | None = None,
    ) -> list[str]:
        """Load all active sub-agents from DB and register them.

        When ``settings`` is provided, two B3 de-bloat passes run around the
        load: :meth:`retire_redundant` first (DB-level, marks semantic
        duplicates inactive so they are never loaded), then
        ``registry.enforce_caps`` (in-memory retirement of chronic low
        performers / stale / over-cap agents), whose decisions are persisted via
        :meth:`retire`. Passing ``settings=None`` (the existing behavior) skips
        both passes — used by recall tests that assert the raw load path.

        Args:
            registry: SubAgentRegistry to register loaded agents into.
            settings: AgentSettings enabling cumulative caps/retirement. None
                disables enforcement (backward compatible).

        Returns:
            List of loaded sub-agent names (excludes anything retired here).
        """
        # Pass 1: retire semantic duplicates before load (best-effort).
        if settings is not None:
            try:
                await self.retire_redundant(
                    settings.capability_redundancy_threshold
                )
            except Exception as e:
                logger.debug(f"Sub-agent redundancy retirement skipped: {e}")

        loaded: list[str] = []

        try:
            from sqlalchemy import select

            from src.db.models import SubAgentModel
            from src.db.session import get_session

            async with get_session() as session:
                stmt = select(SubAgentModel).where(
                    SubAgentModel.is_active.is_(True)
                )
                result = await session.execute(stmt)
                models = result.scalars().all()

                for model in models:
                    try:
                        spec = _model_to_spec(model)
                        registry.register(spec)
                        loaded.append(spec.name)
                    except Exception as e:
                        logger.warning(
                            f"Failed to load sub-agent '{model.name}': {e}"
                        )

        except Exception as e:
            logger.debug(f"Could not load sub-agents from DB: {e}")

        # Pass 2: enforce cumulative cap + retire bad/stale (in-memory),
        # then persist so the decision survives to the next run.
        if settings is not None:
            retired = registry.enforce_caps(
                max_active=settings.max_active_sub_agents,
                min_runs=settings.retire_min_runs,
                success_floor=settings.retire_success_floor,
                recency_days=settings.retire_recency_days,
            )
            if retired:
                await self.retire(retired)
                retired_set = set(retired)
                loaded = [n for n in loaded if n not in retired_set]

        if loaded:
            logger.info(
                f"Loaded {len(loaded)} sub-agents from DB: "
                f"{', '.join(loaded)}"
            )

        return loaded

    async def record_run_and_update_metrics(
        self,
        sub_agent_id: uuid.UUID,
        run_result: dict[str, Any],
        parent_task_id: uuid.UUID | None = None,
        parent_thread_id: str = "",
    ) -> uuid.UUID | None:
        """Record a sub-agent execution run and recalculate rolling metrics.

        Args:
            sub_agent_id: UUID of the SubAgentModel.
            run_result: Dict with 'success', 'result', 'tokens_used',
                'cost_usd', 'latency_ms', 'iterations', 'errors'.
            parent_task_id: Optional parent TaskExecution ID.
            parent_thread_id: Parent's thread ID for tracking.

        Returns:
            UUID of the SubAgentRunModel row, or None on failure.
        """
        run_id: uuid.UUID | None = None

        try:


            from src.db.models import SubAgentRunModel
            from src.db.session import get_session

            async with get_session() as session:
                # Create run record
                now = _utcnow()
                run = SubAgentRunModel(
                    sub_agent_id=sub_agent_id,
                    parent_task_id=parent_task_id,
                    parent_thread_id=parent_thread_id,
                    goal_text=run_result.get("goal", ""),
                    result_summary=run_result.get("result", "")[:2000],
                    status="completed" if run_result.get("success") else "failed",
                    iterations_used=run_result.get("iterations", 0),
                    tokens_used=run_result.get("tokens_used", 0),
                    cost_usd=run_result.get("cost_usd", 0.0),
                    latency_ms=run_result.get("latency_ms", 0),
                    quality_rating=run_result.get("quality_rating"),
                    extra_data=run_result.get("extra_data", {}),
                    created_at=now,
                    completed_at=now,
                )
                session.add(run)
                await session.flush()
                run_id = run.id

                # Recalculate rolling metrics from last 100 runs
                await self._update_rolling_metrics(session, sub_agent_id)

                logger.debug(
                    f"Recorded run for sub-agent {sub_agent_id}: "
                    f"success={run_result.get('success')}"
                )

        except Exception as e:
            logger.warning(
                f"Failed to record sub-agent run for {sub_agent_id}: {e}"
            )

        return run_id

    async def _update_rolling_metrics(
        self,
        session: Any,
        sub_agent_id: uuid.UUID,
    ) -> None:
        """Recalculate rolling metrics from recent runs.

        Computes success_rate, avg_cost, avg_latency_ms, and quality_score
        from the last 100 runs and updates the SubAgentModel row.
        """
        from sqlalchemy import case, func, select, update

        from src.db.models import SubAgentModel, SubAgentRunModel

        # Aggregate from last 100 runs
        metrics_stmt = select(
            func.count(SubAgentRunModel.id).label("total"),
            func.sum(
                case(
                    (SubAgentRunModel.status == "completed", 1),  # noqa: E712
                    else_=0,
                )
            ).label("successes"),
            func.avg(SubAgentRunModel.cost_usd).label("avg_cost"),
            func.avg(SubAgentRunModel.latency_ms).label("avg_latency"),
            func.avg(SubAgentRunModel.quality_rating).label("avg_quality"),
        ).where(
            SubAgentRunModel.sub_agent_id == sub_agent_id,
        )
        metrics_result = await session.execute(metrics_stmt)
        row = metrics_result.one()

        total = row.total or 0
        successes = row.successes or 0
        success_rate = successes / total if total > 0 else 0.0
        avg_cost = float(row.avg_cost or 0)
        avg_latency = int(row.avg_latency or 0)
        avg_quality = float(row.avg_quality or 0.5)

        await session.execute(
            update(SubAgentModel)
            .where(SubAgentModel.id == sub_agent_id)
            .values(
                total_runs=total,
                success_count=successes,
                success_rate=success_rate,
                avg_cost=avg_cost,
                avg_latency_ms=avg_latency,
                quality_score=avg_quality,
                updated_at=_utcnow(),
            )
        )


# ── Conversion Helpers ──────────────────────────────────────────────────


def _spec_to_model(
    spec: SubAgentSpec,
    version: int = 1,
    capability_embedding: list[float] | None = None,
    capability_text: str | None = None,
) -> Any:
    """Convert a SubAgentSpec to a SubAgentModel ORM instance."""
    import uuid as _uuid

    from src.db.models import SubAgentModel

    # Propagate spec.id so the DB primary key matches the Python-side UUID.
    # Without this, SubAgentModel.id is auto-generated by PostgreSQL and differs
    # from spec.id, causing FK violations in SubAgentRunModel.
    try:
        model_id = _uuid.UUID(spec.id)
    except (ValueError, AttributeError):
        model_id = _uuid.uuid4()

    return SubAgentModel(
        id=model_id,
        name=spec.name,
        description=spec.description,
        template_type=spec.template_type,
        tool_scope=spec.tool_scope,
        tool_subset=spec.tool_subset,
        budget_mode=spec.budget_mode,
        budget_limit=spec.budget_limit,
        model_tier=spec.model_tier.value
        if hasattr(spec.model_tier, "value")
        else spec.model_tier,
        max_iterations=spec.max_iterations,
        depth_limit=spec.depth_limit,
        node_config=spec.node_config,
        system_prompt_override=spec.system_prompt_override,
        is_active=spec.is_active,
        version=version,
        capability_embedding=capability_embedding,
        capability_text=capability_text,
        total_runs=spec.total_runs,
        success_count=int(spec.success_rate * spec.total_runs),
        success_rate=spec.success_rate,
        avg_cost=spec.avg_cost,
        avg_latency_ms=spec.avg_latency_ms,
        quality_score=spec.quality_score,
    )


def _model_to_spec(model: Any) -> SubAgentSpec:
    """Convert a SubAgentModel ORM instance to a SubAgentSpec."""
    from src.graph.enums import TaskComplexity
    from src.graph.models import SubAgentSpec

    return SubAgentSpec(
        id=str(model.id),
        name=model.name,
        goal="",  # Goal is set per-invocation, not persisted
        description=model.description,
        model_tier=TaskComplexity(model.model_tier),
        parent_thread_id="",  # Set at runtime
        max_iterations=model.max_iterations,
        template_type=model.template_type,
        tool_scope=model.tool_scope,
        tool_subset=model.tool_subset or [],
        budget_mode=model.budget_mode,
        budget_limit=float(model.budget_limit or 0),
        depth_limit=model.depth_limit,
        node_config=model.node_config or {},
        system_prompt_override=model.system_prompt_override,
        version=model.version,
        is_active=model.is_active,
        total_runs=model.total_runs,
        success_rate=float(model.success_rate or 0),
        avg_cost=float(model.avg_cost or 0),
        avg_latency_ms=int(model.avg_latency_ms or 0),
        quality_score=float(model.quality_score or 0.5),
        # updated_at is bumped on every run-record (see _update_rolling_metrics),
        # so it is a faithful "last used" proxy without a dedicated column.
        last_used_at=getattr(model, "updated_at", None),
    )


def _utcnow() -> Any:
    """Return timezone-aware UTC now."""
    import datetime as dt

    return dt.datetime.now(dt.timezone.utc)


def _cosine(u: list[float], v: list[float]) -> float:
    """Cosine similarity; 0.0 for zero-norm vectors (no direction).

    ``u`` and ``v`` must be equal-length (768-d capability embeddings).
    """
    dot = 0.0
    nu = 0.0
    nv = 0.0
    for a, b in zip(u, v, strict=True):
        dot += a * b
        nu += a * a
        nv += b * b
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return dot / (math.sqrt(nu) * math.sqrt(nv))
