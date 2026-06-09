"""DB persistence for sub-agent definitions and run records.

Stores SubAgentSpec definitions in SubAgentModel table and individual
execution records in SubAgentRunModel. Loads active sub-agents at
startup into SubAgentRegistry.

Follows the same pattern as src/tools/dynamic/persister.py (ToolPersister).
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from src.agents.registry import SubAgentRegistry
    from src.graph.models import SubAgentSpec


class SubAgentPersister:
    """Persist and load sub-agent definitions from the database."""

    async def persist(self, spec: SubAgentSpec) -> uuid.UUID | None:
        """Write or update a SubAgentModel in the database.

        If a sub-agent with the same name already exists, increments
        the version and deactivates the old row. Otherwise creates a
        new row with version 1.

        Args:
            spec: SubAgentSpec to persist.

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
                    model = _spec_to_model(spec, version=new_version)
                    session.add(model)
                    await session.flush()

                    logger.info(
                        f"Updated sub-agent '{spec.name}' to version {new_version}"
                    )
                    return model.id

                # Create new sub-agent
                model = _spec_to_model(spec, version=1)
                session.add(model)
                await session.flush()

                logger.info(f"Persisted new sub-agent '{spec.name}' (version 1)")
                return model.id

        except Exception as e:
            logger.warning(f"Failed to persist sub-agent '{spec.name}': {e}")
            return None

    async def load_active_agents(
        self,
        registry: SubAgentRegistry,
    ) -> list[str]:
        """Load all active sub-agents from DB and register them.

        Args:
            registry: SubAgentRegistry to register loaded agents into.

        Returns:
            List of loaded sub-agent names.
        """
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
        from sqlalchemy import func, select, update

        from src.db.models import SubAgentModel, SubAgentRunModel

        # Aggregate from last 100 runs
        metrics_stmt = select(
            func.count(SubAgentRunModel.id).label("total"),
            func.sum(
                SubAgentRunModel.status == "completed"  # noqa: E712
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


def _spec_to_model(spec: SubAgentSpec, version: int = 1) -> Any:
    """Convert a SubAgentSpec to a SubAgentModel ORM instance."""
    from src.db.models import SubAgentModel

    return SubAgentModel(
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
    )


def _utcnow() -> Any:
    """Return timezone-aware UTC now."""
    import datetime as dt

    return dt.datetime.now(dt.timezone.utc)
