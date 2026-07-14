"""DB persistence for self-evolution cycles.

Records every evolution cycle as a ``MutationChain`` (one per cycle), its
generated ``Mutation`` rows, and fine-grained ``EvolutionTelemetry`` events
(generation attempts, validation results, deploy/reject outcomes). Every method
swallows DB errors and returns a safe sentinel (``None``/``False``) so a
persistence failure can never abort an evolution cycle — the engine's
in-memory result is the source of truth for control flow.

Follows the same pattern as ``src/agents/persister.py`` (SubAgentPersister):
async methods, one ``get_session()`` context per method (autocommit on exit,
rollback on exception), try/except + loguru.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from loguru import logger


def _json_safe(obj: Any) -> Any:
    """Round-trip a value through JSON so it is safe for a JSONB column.

    Coerces non-serializable values (enums, Pydantic models, datetimes) to
    ``str`` so asyncpg/SQLAlchemy never fails to bind a JSONB payload. Primitive
    values pass through unchanged.
    """
    try:
        return json.loads(json.dumps(obj, default=str))
    except (TypeError, ValueError) as e:
        logger.debug(f"_json_safe fallback on non-serializable value: {e}")
        return {"_unserializable": str(obj)}


def _coerce_mutation_type(value: Any) -> str:
    """Coerce a mutation_type value (enum / str / None) to a plain string.

    Enums yield their ``.value`` (avoids storing ``"MutationType.CODE"``);
    ``None`` becomes ``"unknown"``; anything else is ``str()``-ed. Uses
    ``getattr`` with a default so it is safe on every input type.
    """
    if value is None:
        return "unknown"
    candidate = getattr(value, "value", None)
    return str(candidate) if candidate is not None else str(value)


def _utcnow() -> Any:
    """Return timezone-aware UTC now (deferred import keeps the module light)."""
    import datetime as dt

    return dt.datetime.now(dt.timezone.utc)


class EvolutionPersister:
    """Persist evolution chains, mutations, and telemetry to PostgreSQL.

    All methods are async, take no ``session`` parameter (each opens its own),
    and never raise — failures are logged and mapped to ``None``/``False``.
    """

    async def create_chain(
        self,
        trigger_reason: str,
        extra_data: dict[str, Any] | None = None,
    ) -> uuid.UUID | None:
        """Create a new ``MutationChain`` row marked ``in_progress``.

        Args:
            trigger_reason: Why this cycle ran (e.g. the opportunity description).
            extra_data: Optional structured context (priority, generation).

        Returns:
            UUID of the new chain row, or None on failure.
        """
        try:
            from src.db.models import MutationChain
            from src.db.session import get_session

            async with get_session() as session:
                chain = MutationChain(
                    trigger_reason=trigger_reason[:1000] if trigger_reason else "evolution_cycle",
                    status="in_progress",
                    extra_data=_json_safe(extra_data) if extra_data else {},
                    created_at=_utcnow(),
                )
                session.add(chain)
                await session.flush()
                chain_id = chain.id
                logger.info(f"Created mutation chain {chain_id}: {trigger_reason[:80]}")
                return chain_id
        except Exception as e:
            logger.warning(f"Failed to create mutation chain: {e}")
            return None

    async def record_mutation(
        self,
        chain_id: uuid.UUID | None,
        proposal: dict[str, Any],
        status: str = "generated",
        diff_content: str | None = None,
    ) -> uuid.UUID | None:
        """Record a generated mutation proposal against a chain.

        Args:
            chain_id: Parent chain UUID. ``None`` is a no-op (the Mutation.chain_id
                FK is NOT NULL), returning None — callers may proceed when the chain
                failed to persist.
            proposal: The mutation proposal dict from ``SelfEvolutionEngine.generate``.
            status: Mutation status (generated/rejected/deployed).
            diff_content: Optional unified diff text.

        Returns:
            UUID of the new mutation row, or None on failure / missing chain.
        """
        if chain_id is None:
            # Mutation.chain_id is NOT NULL — cannot record without a parent chain.
            return None
        try:
            from src.db.models import Mutation
            from src.db.session import get_session

            # Coerce enum → .value for the Text column (avoids "MutationType.X").
            mutation_type = _coerce_mutation_type(proposal.get("mutation_type"))
            model_used = proposal.get("model_used")

            async with get_session() as session:
                mutation = Mutation(
                    chain_id=chain_id,
                    mutation_type=mutation_type,
                    target_path=proposal.get("target_path"),
                    description=str(proposal.get("description", "Unknown improvement"))[:5000],
                    original_content=proposal.get("original_content"),
                    mutated_content=proposal.get("mutated_content", ""),
                    diff_content=diff_content,
                    model_used=model_used if isinstance(model_used, str) else None,
                    tokens_used=int(proposal.get("tokens_used", 0) or 0),
                    status=status,
                    created_at=_utcnow(),
                )
                session.add(mutation)
                await session.flush()
                mutation_id = mutation.id
                logger.debug(
                    f"Recorded mutation {mutation_id} (chain={chain_id}, "
                    f"type={mutation_type}, status={status})"
                )
                return mutation_id
        except Exception as e:
            logger.warning(f"Failed to record mutation for chain {chain_id}: {e}")
            return None

    async def update_mutation_status(
        self,
        mutation_id: uuid.UUID | None,
        status: str,
    ) -> bool:
        """Transition an existing mutation's status (e.g. generated → deployed).

        Args:
            mutation_id: Mutation row UUID.
            status: New status string.

        Returns:
            True if the update succeeded, False otherwise.
        """
        if mutation_id is None:
            return False
        try:
            from sqlalchemy import update

            from src.db.models import Mutation
            from src.db.session import get_session

            async with get_session() as session:
                await session.execute(
                    update(Mutation)
                    .where(Mutation.id == mutation_id)
                    .values(status=status)
                )
                await session.flush()
                logger.debug(f"Updated mutation {mutation_id} status → {status}")
                return True
        except Exception as e:
            logger.warning(f"Failed to update mutation {mutation_id} status: {e}")
            return False

    async def set_active_config_version(
        self, version_id: uuid.UUID
    ) -> bool:
        """Atomically mark ``version_id`` as the single active config version.

        One transaction: clear ``is_active`` on every other
        ``agent_config_versions`` row, then set it on the target — so the
        one-active invariant holds even mid-swap (Phase 3b). A CONFIG mutation's
        rollback re-points here to the prior version id instead of mutating data.

        Returns True on success, False on any failure (non-fatal — caller logs).
        """
        try:
            from sqlalchemy import update

            from src.db.models import AgentConfigVersion
            from src.db.session import get_session

            async with get_session() as session:
                # Clear all, then activate the target — within one tx so the
                # partial unique index (one is_active=true) is never violated.
                await session.execute(
                    update(AgentConfigVersion).values(is_active=False)
                )
                await session.execute(
                    update(AgentConfigVersion)
                    .where(AgentConfigVersion.id == version_id)
                    .values(is_active=True)
                )
                await session.flush()
            logger.debug(f"Activated config version {version_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to activate config version {version_id}: {e}")
            return False

    async def get_active_config_version(self) -> uuid.UUID | None:
        """Return the currently-active config version id, or None if none/unknown.

        Reads the single ``is_active=True`` row (the partial unique index
        guarantees at most one). Best-effort: any DB error returns None.
        """
        try:
            from sqlalchemy import select

            from src.db.models import AgentConfigVersion
            from src.db.session import get_session

            async with get_session() as session:
                row = (
                    await session.execute(
                        select(AgentConfigVersion.id).where(
                            AgentConfigVersion.is_active.is_(True)
                        )
                    )
                ).scalar_one_or_none()
                return row
        except Exception as e:
            logger.debug(f"Could not read active config version: {e}")
            return None

    async def record_ab_test_result(
        self,
        mutation_id: uuid.UUID | None,
        ab_result: dict[str, Any],
    ) -> None:
        """Persist one ``ab_test_results`` row (Phase 4 — A/B rigor).

        Carries the p-value / significance / confidence the engine computed,
        keyed to the mutation. Non-fatal: a DB hiccup is logged and swallowed
        (an evolution cycle must never abort on a telemetry write). No-op when
        ``mutation_id`` is None or the result lacks paired-test stats (the
        sandbox-skip paths set no ``tested`` flag and so are not persisted).
        """
        if mutation_id is None or not ab_result.get("tested"):
            return
        try:
            from src.db.models import ABTestResult
            from src.db.session import get_session

            async with get_session() as session:
                session.add(
                    ABTestResult(
                        mutation_id=mutation_id,
                        metric_name=ab_result.get(
                            "metric_name", "sandbox_duration_seconds"
                        ),
                        control_value=ab_result.get("control_value"),
                        treatment_value=ab_result.get("treatment_value"),
                        sample_size=ab_result.get("sample_size"),
                        p_value=ab_result.get("p_value"),
                        is_significant=ab_result.get("is_significant"),
                        confidence=ab_result.get("confidence"),
                    )
                )
                await session.flush()
            logger.debug(f"Recorded A/B result for mutation {mutation_id}")
        except Exception as e:  # noqa: BLE001 — telemetry write is best-effort
            logger.warning(f"Failed to record A/B result: {e}")

    async def record_event(
        self,
        chain_id: uuid.UUID | None,
        event_type: str,
        event_data: dict[str, Any] | None = None,
    ) -> None:
        """Append a telemetry event for a chain.

        ``chain_id`` is nullable (EvolutionTelemetry.chain_id allows NULL) so an
        event can be recorded even when the chain row failed to persist.

        Args:
            chain_id: Parent chain UUID (or None).
            event_type: Event discriminator (generation_attempt, validation_result,
                deployed, rejected, ...).
            event_data: Optional structured payload.
        """
        try:
            from src.db.models import EvolutionTelemetry
            from src.db.session import get_session

            async with get_session() as session:
                event = EvolutionTelemetry(
                    chain_id=chain_id,
                    event_type=event_type,
                    event_data=_json_safe(event_data) if event_data else {},
                    created_at=_utcnow(),
                )
                session.add(event)
                await session.flush()
        except Exception as e:
            # Telemetry is strictly non-critical: never let it abort a cycle.
            logger.debug(f"Failed to record evolution event '{event_type}': {e}")

    async def complete_chain(
        self,
        chain_id: uuid.UUID | None,
        status: str,
    ) -> bool:
        """Mark a chain complete with a terminal status and completion timestamp.

        Args:
            chain_id: Chain UUID. ``None`` is a no-op returning False.
            status: Terminal status (deployed/rejected/validation_failed/sandbox_failed).

        Returns:
            True if the chain was closed, False otherwise.
        """
        if chain_id is None:
            return False
        try:
            from sqlalchemy import update

            from src.db.models import MutationChain
            from src.db.session import get_session

            async with get_session() as session:
                await session.execute(
                    update(MutationChain)
                    .where(MutationChain.id == chain_id)
                    .values(status=status, completed_at=_utcnow())
                )
                await session.flush()
                logger.info(f"Completed mutation chain {chain_id}: {status}")
                return True
        except Exception as e:
            logger.warning(f"Failed to complete chain {chain_id}: {e}")
            return False
