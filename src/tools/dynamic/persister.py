"""DB persistence for dynamically generated tools.

Stores validated tools in ToolRegistration + ToolVersion tables
and loads them back into ToolRegistry at startup.
"""

from __future__ import annotations

import datetime as dt
import math
import uuid
from typing import TYPE_CHECKING, Any

from loguru import logger


if TYPE_CHECKING:
    from src.config.settings import AgentSettings
    from src.tools.registry import ToolRegistry


class ToolPersister:
    """Persist and load dynamically generated tools from the database."""

    async def persist(
        self,
        tool_name: str,
        description: str,
        input_schema: dict[str, Any],
        handler_code: str,
        test_code: str = "",
        capability_embedding: list[float] | None = None,
        capability_text: str | None = None,
    ) -> uuid.UUID | None:
        """Write ToolRegistration + ToolVersion to DB.

        Args:
            tool_name: Unique snake_case identifier.
            description: Human-readable tool description.
            input_schema: JSON Schema for tool parameters.
            handler_code: Complete async function source code.
            test_code: Optional test code for the tool.
            capability_embedding: Optional 768-d capability vector (B3 dedup).
                Stored on the registration so future gaps can reuse this tool
                via :meth:`find_similar`. Pass None when no real embedding is
                available (hash fallback) — never overwrite a stored vector
                with None on a version bump.
            capability_text: The text ``capability_embedding`` was derived from.

        Returns:
            UUID of the ToolRegistration row, or None on failure.
        """
        try:
            from src.db.models import ToolRegistration, ToolVersion
            from src.db.session import get_session

            async with get_session() as session:
                # Check if tool already exists (update instead of duplicate)
                from sqlalchemy import select

                stmt = select(ToolRegistration).where(
                    ToolRegistration.tool_name == tool_name
                )
                result = await session.execute(stmt)
                existing = result.scalar_one_or_none()

                if existing is not None:
                    # Refresh the capability embedding only when a real vector
                    # is supplied — a version bump without one must not clobber
                    # a previously stored embedding with NULL. An explicit UPDATE
                    # (not attribute assignment) hands pgvector the raw list and
                    # keeps pyright clean (Mapped[Vector] attrs are typed as the
                    # SQLCore wrapper, not plain list[float]).
                    if capability_embedding is not None:
                        from sqlalchemy import update as _update

                        await session.execute(
                            _update(ToolRegistration)
                            .where(ToolRegistration.id == existing.id)
                            .values(
                                capability_embedding=capability_embedding,
                                capability_text=capability_text,
                            )
                        )

                    # Create a new version of the existing tool
                    version_stmt = select(ToolVersion).where(
                        ToolVersion.tool_id == existing.id
                    ).order_by(ToolVersion.version.desc())
                    ver_result = await session.execute(version_stmt)
                    latest_ver = ver_result.scalars().first()
                    next_version = (latest_ver.version + 1) if latest_ver else 1

                    # Deactivate old versions
                    from sqlalchemy import update

                    await session.execute(
                        update(ToolVersion)
                        .where(ToolVersion.tool_id == existing.id)
                        .values(is_active=False)
                    )

                    new_version = ToolVersion(
                        tool_id=existing.id,
                        version=next_version,
                        code_content=handler_code,
                        test_content=test_code or None,
                        is_active=True,
                    )
                    session.add(new_version)
                    await session.flush()

                    logger.info(
                        f"Updated tool '{tool_name}' to version {next_version}"
                    )
                    return existing.id

                # Create new tool registration
                registration = ToolRegistration(
                    tool_name=tool_name,
                    tool_type="generated",
                    description=description,
                    input_schema=input_schema,
                    is_active=True,
                    capability_embedding=capability_embedding,
                    capability_text=capability_text,
                )
                session.add(registration)
                await session.flush()

                # Create initial version
                version = ToolVersion(
                    tool_id=registration.id,
                    version=1,
                    code_content=handler_code,
                    test_content=test_code or None,
                    is_active=True,
                )
                session.add(version)

                logger.info(f"Persisted new tool '{tool_name}' (version 1)")
                return registration.id

        except Exception as e:
            logger.warning(f"Failed to persist tool '{tool_name}': {e}")
            return None

    # ---- D10: operator edit → review → approve lifecycle -------------------

    async def submit_pending_version(
        self,
        tool_name: str,
        description: str,
        input_schema: dict[str, Any],
        handler_code: str,
        test_code: str = "",
        capability_embedding: list[float] | None = None,
        capability_text: str | None = None,
    ) -> uuid.UUID | None:
        """Stage an operator-edited tool version for HITL approval (D10).

        Like :meth:`persist` but the new ToolVersion is parked at
        ``status='pending_review', is_active=False`` so :meth:`load_active_tools`
        does NOT materialize it until :meth:`approve_pending` flips it. Every
        prior version is left untouched, so the current live tool keeps running
        unchanged while the edit is under review.

        The ToolRegistration description/input_schema are updated to the new
        values now (the registration is the tool's identity + latest-known
        description, not version-pinned — the same framing :meth:`find_similar`
        already uses). The materialized handler still comes from the
        approved+active version's ``code_content``, so during the review window
        the LLM-visible description may briefly precede the executing code (an
        acceptable HITL artifact; :meth:`approve_pending` closes the gap).

        Returns:
            UUID of the staged ``ToolVersion`` row, or None on failure.
        """
        try:
            from sqlalchemy import update

            from src.db.models import ToolRegistration, ToolVersion
            from src.db.session import get_session

            async with get_session() as session:
                existing = await self._get_registration(session, tool_name)

                if existing is not None:
                    # Refresh the capability embedding only with a real vector
                    # (a None must never clobber a stored one — see persist()).
                    values: dict[str, Any] = {
                        "description": description,
                        "input_schema": input_schema,
                    }
                    if capability_embedding is not None:
                        values["capability_embedding"] = capability_embedding
                        values["capability_text"] = capability_text
                    await session.execute(
                        update(ToolRegistration)
                        .where(ToolRegistration.id == existing.id)
                        .values(**values)
                    )
                    registration_id = existing.id
                    latest = await self._latest_version(session, existing.id)
                    next_version = (latest.version + 1) if latest else 1
                else:
                    registration = ToolRegistration(
                        tool_name=tool_name,
                        tool_type="generated",
                        description=description,
                        input_schema=input_schema,
                        is_active=True,
                        capability_embedding=capability_embedding,
                        capability_text=capability_text,
                    )
                    session.add(registration)
                    await session.flush()
                    registration_id = registration.id
                    next_version = 1

                pending = ToolVersion(
                    tool_id=registration_id,
                    version=next_version,
                    code_content=handler_code,
                    test_content=test_code or None,
                    is_active=False,
                    status="pending_review",
                )
                session.add(pending)
                await session.flush()

                logger.info(
                    f"Staged pending version {next_version} for tool '{tool_name}'"
                )
                return pending.id

        except Exception as e:
            logger.warning(
                f"Failed to stage pending version for tool '{tool_name}': {e}"
            )
            return None

    async def approve_pending(self, tool_name: str) -> dict[str, Any] | None:
        """Promote the latest ``pending_review`` version to live (D10).

        Sets the tool's latest ``pending_review`` ToolVersion to
        ``status='approved', is_active=True`` and deactivates every OTHER
        version (prior live + any older pending). The ToolRegistration
        description/input_schema were synced at :meth:`submit_pending_version`
        time, so :meth:`load_active_tools` materializes the approved code with a
        matching description.

        Returns:
            A dict (``tool_name``, ``version``, ``status``) describing the
            approved version, or None if there was no ``pending_review``
            version or the write failed.
        """
        try:
            from sqlalchemy import update

            from src.db.models import ToolVersion
            from src.db.session import get_session

            async with get_session() as session:
                reg = await self._get_registration(session, tool_name)
                if reg is None:
                    logger.debug(f"approve_pending: no tool '{tool_name}'")
                    return None

                pending = await self._latest_version(
                    session, reg.id, status="pending_review"
                )
                if pending is None:
                    logger.debug(
                        f"approve_pending: no pending_review version for '{tool_name}'"
                    )
                    return None

                # Deactivate every version of this tool, then activate + approve
                # the staged one. Two statements so the staged row's is_active
                # flip is unconditional regardless of its prior state.
                await session.execute(
                    update(ToolVersion)
                    .where(ToolVersion.tool_id == reg.id)
                    .values(is_active=False)
                )
                await session.execute(
                    update(ToolVersion)
                    .where(ToolVersion.id == pending.id)
                    .values(status="approved", is_active=True)
                )
                logger.info(
                    f"Approved pending version {pending.version} for tool '{tool_name}'"
                )
                return {
                    "tool_name": tool_name,
                    "version": pending.version,
                    "status": "approved",
                }

        except Exception as e:
            logger.warning(
                f"Failed to approve pending version for tool '{tool_name}': {e}"
            )
            return None

    async def reject_pending(self, tool_name: str, *, reason: str | None = None) -> bool:
        """Dismiss the latest ``pending_review`` version (D10).

        Marks the tool's latest ``pending_review`` ToolVersion
        ``status='rejected'`` (``is_active`` stays False — it was never live).
        Prior approved versions are untouched, so the live tool is unaffected.

        Returns:
            True if a pending version was rejected, False otherwise.
        """
        try:
            from sqlalchemy import update

            from src.db.models import ToolVersion
            from src.db.session import get_session

            async with get_session() as session:
                reg = await self._get_registration(session, tool_name)
                if reg is None:
                    logger.debug(f"reject_pending: no tool '{tool_name}'")
                    return False

                pending = await self._latest_version(
                    session, reg.id, status="pending_review"
                )
                if pending is None:
                    logger.debug(
                        f"reject_pending: no pending_review version for '{tool_name}'"
                    )
                    return False

                await session.execute(
                    update(ToolVersion)
                    .where(ToolVersion.id == pending.id)
                    .values(status="rejected")
                )
                logger.info(
                    f"Rejected pending version {pending.version} for tool '{tool_name}'"
                    + (f": {reason}" if reason else "")
                )
                return True

        except Exception as e:
            logger.warning(
                f"Failed to reject pending version for tool '{tool_name}': {e}"
            )
            return False

    async def list_tools(self) -> list[dict[str, Any]]:
        """List generated tool registrations + their latest version (D10).

        Returns one dict per registration: ``tool_name``, ``description``,
        ``is_active`` (registration), and the latest version's ``version``,
        ``status`` (``approved``/``pending_review``/``rejected``),
        ``version_active``. Best-effort: ``[]`` on DB error.
        """
        try:
            from sqlalchemy import select

            from src.db.models import ToolRegistration
            from src.db.session import get_session

            tools: list[dict[str, Any]] = []
            async with get_session() as session:
                result = await session.execute(
                    select(ToolRegistration).where(
                        ToolRegistration.tool_type == "generated"
                    )
                )
                for reg in result.scalars().all():
                    latest = await self._latest_version(session, reg.id)
                    tools.append(
                        {
                            "tool_name": reg.tool_name,
                            "description": reg.description,
                            "is_active": reg.is_active,
                            "version": latest.version if latest else None,
                            "status": latest.status if latest else None,
                            "version_active": latest.is_active if latest else None,
                        }
                    )
            return tools
        except Exception as e:
            logger.debug(f"list_tools failed: {e}")
            return []

    async def get_tool(self, tool_name: str) -> dict[str, Any] | None:
        """Inspect a single generated tool + its latest version (D10).

        Returns ``None`` when no registration exists. Best-effort: ``None`` on
        DB error.
        """
        try:
            from sqlalchemy import select

            from src.db.models import ToolVersion
            from src.db.session import get_session

            async with get_session() as session:
                reg = await self._get_registration(session, tool_name)
                if reg is None:
                    return None
                latest = await self._latest_version(session, reg.id)
                versions = await session.execute(
                    select(ToolVersion)
                    .where(ToolVersion.tool_id == reg.id)
                    .order_by(ToolVersion.version.desc())
                )
                history = [
                    {
                        "version": v.version,
                        "status": v.status,
                        "is_active": v.is_active,
                        "created_at": v.created_at.isoformat()
                        if v.created_at
                        else None,
                    }
                    for v in versions.scalars().all()
                ]
                return {
                    "tool_name": reg.tool_name,
                    "description": reg.description,
                    "input_schema": reg.input_schema,
                    "is_active": reg.is_active,
                    "version": latest.version if latest else None,
                    "status": latest.status if latest else None,
                    "code_content": latest.code_content if latest else None,
                    "test_content": latest.test_content if latest else None,
                    "history": history,
                }
        except Exception as e:
            logger.debug(f"get_tool '{tool_name}' failed: {e}")
            return None

    # ---- D10 helpers -------------------------------------------------------

    async def _get_registration(self, session: Any, tool_name: str) -> Any:
        """Fetch a ToolRegistration by name (or None)."""
        from sqlalchemy import select

        from src.db.models import ToolRegistration

        result = await session.execute(
            select(ToolRegistration).where(ToolRegistration.tool_name == tool_name)
        )
        return result.scalar_one_or_none()

    async def _latest_version(
        self, session: Any, tool_id: uuid.UUID, *, status: str | None = None
    ) -> Any:
        """Latest ToolVersion for a tool, optionally filtered by ``status``.

        Ordered by version desc — the newest version matching ``status`` (or the
        newest overall when ``status`` is None).
        """
        from sqlalchemy import select

        from src.db.models import ToolVersion

        stmt = select(ToolVersion).where(ToolVersion.tool_id == tool_id)
        if status is not None:
            stmt = stmt.where(ToolVersion.status == status)
        stmt = stmt.order_by(ToolVersion.version.desc()).limit(1)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    async def find_similar(
        self,
        embedding: list[float],
        threshold: float = 0.85,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Cosine-search active tool capability embeddings (B3 dedup).

        Args:
            embedding: Query capability vector (e.g. of a capability gap).
            threshold: Minimum cosine similarity to report (1 - cosine distance).
            limit: Max candidates to scan from the HNSW index.

        Returns:
            Tools at/above ``threshold``, most-similar first, as
            ``[{"tool_name", "description", "similarity"}]``. Filters rows with
            no stored embedding. Best-effort: returns ``[]`` on any DB error so
            dedup degrades to "create" rather than blocking the run.
        """
        try:
            from sqlalchemy import select

            from src.db.models import ToolRegistration
            from src.db.session import get_session

            async with get_session() as session:
                distance = ToolRegistration.capability_embedding.cosine_distance(
                    embedding
                )
                stmt = (
                    select(
                        ToolRegistration.tool_name,
                        ToolRegistration.description,
                        distance.label("distance"),
                    )
                    .where(
                        ToolRegistration.capability_embedding.isnot(None),
                        ToolRegistration.is_active.is_(True),
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
                                "tool_name": name,
                                "description": desc,
                                "similarity": similarity,
                            }
                        )
                return matches
        except Exception as e:
            logger.debug(f"Tool capability find_similar failed: {e}")
            return []

    async def retrieve_tools(
        self,
        query_embedding: list[float],
        top_k: int = 8,
    ) -> list[str]:
        """RECALL (findings-05): the top-k active tool NAMES ranked by capability
        cosine similarity to a query embedding — the recall counterpart to the
        dedup-only ``find_similar``.

        ``threshold=0.0`` so every embedded tool is eligible; the HNSW order +
        ``limit`` do the ranking. Returns names only (the caller resolves them
        via ``ToolRegistry.list_tools(names)``). Built-in tools are NOT here
        (they have no stored embedding — only ``tool_create``/``agent_spawn``
        persist embeddings) so the selection layer adds them unconditionally.
        Best-effort: ``[]`` on any DB error (the caller then falls back to the
        full tool set, never starving the run).
        """
        matches = await self.find_similar(
            query_embedding, threshold=0.0, limit=top_k
        )
        return [m["tool_name"] for m in matches]

    async def retrieve_tools_with_scores(
        self,
        query_embedding: list[float],
        top_k: int = 8,
    ) -> list[tuple[str, float]]:
        """RECALL with similarities (E2): top-k ``(tool_name, cosine_similarity)``.

        Same HNSW ranking as :meth:`retrieve_tools` but keeps the similarity so
        the E2 score-blend can multiply ``cosine · f(success)`` before re-ranking.
        ``threshold=0.0`` so every embedded tool is eligible. Best-effort:
        ``[]`` on any DB error (the caller falls back to names-only / full set).
        """
        matches = await self.find_similar(
            query_embedding, threshold=0.0, limit=top_k
        )
        return [(m["tool_name"], float(m["similarity"])) for m in matches]

    async def tool_success_metrics(
        self, names: list[str]
    ) -> dict[str, dict[str, float]]:
        """Running success aggregates for ``names`` (E2 blend signal).

        Returns ``{name: {"success_rate", "empty_output_rate", "calls"}}`` for
        the named generated tools, keyed by ``tool_name``. A name with no row
        (or a DB error) is simply absent — the caller treats a missing name as a
        cold-start tool (success_rate=1.0, empty_output_rate=0.0) so an untested
        tool is never starved by the blend. Best-effort: ``{}`` on any DB error.
        """
        if not names:
            return {}
        try:
            from sqlalchemy import select

            from src.db.models import ToolRegistration
            from src.db.session import get_session

            metrics: dict[str, dict[str, float]] = {}
            async with get_session() as session:
                stmt = select(
                    ToolRegistration.tool_name,
                    ToolRegistration.success_rate,
                    ToolRegistration.empty_output_rate,
                    ToolRegistration.calls,
                ).where(ToolRegistration.tool_name.in_(names))
                result = await session.execute(stmt)
                for name, sr, eor, calls in result.all():
                    metrics[name] = {
                        "success_rate": float(sr) if sr is not None else 1.0,
                        "empty_output_rate": float(eor) if eor is not None else 0.0,
                        "calls": float(calls) if calls is not None else 0.0,
                    }
            return metrics
        except Exception as e:
            logger.debug(f"tool_success_metrics failed: {e}")
            return {}

    async def retire(self, names: list[str]) -> int:
        """Mark named generated tools ``is_active=False`` in the DB.

        Returns:
            Number of tools retired (best-effort; logs on DB error).
        """
        if not names:
            return 0
        try:
            from sqlalchemy import update

            from src.db.models import ToolRegistration
            from src.db.session import get_session

            async with get_session() as session:
                await session.execute(
                    update(ToolRegistration)
                    .where(ToolRegistration.tool_name.in_(names))
                    .values(is_active=False)
                )
            logger.info(f"Retired {len(names)} tools: {', '.join(names)}")
            return len(names)
        except Exception as e:
            logger.warning(f"Failed to retire tools {names}: {e}")
            return 0

    async def merge_alias(self, source: str, target: str) -> int:
        """Re-point ``tool_subset`` references from ``source`` to ``target``.

        When consolidation retires a redundant tool, sub-agents scoped to it
        (``tool_scope='inherit_subset'`` whose ``tool_subset`` lists ``source``)
        must be re-pointed to the surviving twin ``target`` or delegation loses
        the capability. Loads affected SubAgentModel rows, rewrites the list in
        Python (dedup, order-preserving), and updates. Returns rows updated;
        0 when nothing references ``source`` or ``source == target``.
        """
        if not source or source == target:
            return 0
        try:
            from sqlalchemy import select, update

            from src.db.models import SubAgentModel
            from src.db.session import get_session

            updated = 0
            async with get_session() as session:
                # JSONB containment: tool_subset @> '["source"]'.
                stmt = select(SubAgentModel).where(
                    SubAgentModel.tool_subset.contains([source])
                )
                result = await session.execute(stmt)
                for model in result.scalars().all():
                    subset = list(model.tool_subset or [])
                    if source not in subset:
                        continue
                    seen: set[str] = set()
                    deduped = [
                        t
                        for t in (target if t == source else t for t in subset)
                        if not (t in seen or seen.add(t))  # type: ignore[func-returns-value]
                    ]
                    await session.execute(
                        update(SubAgentModel)
                        .where(SubAgentModel.id == model.id)
                        .values(tool_subset=deduped)
                    )
                    updated += 1
            if updated:
                logger.info(
                    f"Re-pointed {updated} sub-agent tool_subset refs "
                    f"{source}→{target}"
                )
            return updated
        except Exception as e:
            logger.warning(f"merge_alias {source}→{target} failed: {e}")
            return 0

    async def _active_tool_capability_rows(self) -> list[dict[str, Any]]:
        """Fetch active generated tools' embeddings + scoring signals.

        Per-tool success metrics now exist (M4 — ``calls``/``success_rate`` on
        ``ToolRegistration``), but the redundancy tie-break deliberately still
        uses ``(max_version, created_at)``: redundancy retirement is about
        duplicate *capability*, while chronic low *performance* is a separate
        signal handled by :meth:`retire_underperforming`. Each row:
        ``{"name", "embedding" (list|None), "version", "created_ts"}``.
        """
        from sqlalchemy import func, select

        from src.db.models import ToolRegistration, ToolVersion
        from src.db.session import get_session

        maxver = (
            select(
                ToolVersion.tool_id,
                func.max(ToolVersion.version).label("maxver"),
            )
            .group_by(ToolVersion.tool_id)
            .subquery()
        )
        stmt = (
            select(
                ToolRegistration.tool_name,
                ToolRegistration.capability_embedding,
                func.coalesce(maxver.c.maxver, 1).label("version"),
                ToolRegistration.created_at,
            )
            .outerjoin(maxver, maxver.c.tool_id == ToolRegistration.id)
            .where(
                ToolRegistration.is_active.is_(True),
                ToolRegistration.tool_type == "generated",
            )
        )
        rows: list[dict[str, Any]] = []
        async with get_session() as session:
            result = await session.execute(stmt)
            for name, emb, ver, created in result.all():
                vector = [float(x) for x in emb] if emb is not None else None
                version_i = int(ver or 1)
                ts = created.timestamp() if created is not None else 0.0
                rows.append(
                    {
                        "name": name,
                        "embedding": vector,
                        "version": version_i,
                        "created_ts": ts,
                        # Self-describing sort key: a more-evolved (higher
                        # version) and newer tool wins a redundancy tie.
                        "score": (version_i, ts),
                    }
                )
        return rows

    async def retire_redundant(self, threshold: float) -> list[str]:
        """Retire semantically-duplicate active tools (B3 de-bloat).

        Loads every active tool's capability embedding, then retires the
        lower-ranked twin of any pair whose cosine similarity >= ``threshold``
        (the stricter consolidation cutoff). Without success metrics, ranking is
        ``(version, created_ts)`` — a more-evolved/newer tool survives.
        Best-effort: any error degrades to no retirement.

        Returns:
            Names of tools retired (sorted).
        """
        try:
            rows = await self._active_tool_capability_rows()
        except Exception as e:
            logger.debug(f"Tool redundancy scan failed: {e}")
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
                f"Retired {len(retired)} redundant tools "
                f"(threshold={threshold}): {', '.join(sorted(retired))}"
            )
        return sorted(retired)

    async def _retire_excess_tools(self, max_active: int) -> int:
        """Retire the oldest active generated tools down to ``max_active``.

        Over-cap retirement stays age-based (oldest by ``created_at`` first):
        it is a hard population cap, distinct from the metrics-driven
        :meth:`retire_underperforming` which removes chronic low performers
        regardless of age. Returns the count retired.
        """
        try:
            from sqlalchemy import select, update

            from src.db.models import ToolRegistration
            from src.db.session import get_session

            async with get_session() as session:
                # Newest first; the tail beyond the cap is the oldest excess.
                stmt = (
                    select(ToolRegistration.id)
                    .where(
                        ToolRegistration.is_active.is_(True),
                        ToolRegistration.tool_type == "generated",
                    )
                    .order_by(ToolRegistration.created_at.desc())
                )
                result = await session.execute(stmt)
                ids = [row[0] for row in result.all()]
                excess = ids[max_active:]
                if not excess:
                    return 0
                await session.execute(
                    update(ToolRegistration)
                    .where(ToolRegistration.id.in_(excess))
                    .values(is_active=False)
                )
                logger.info(
                    f"Retired {len(excess)} excess generated tools "
                    f"to enforce cap {max_active}"
                )
                return len(excess)
        except Exception as e:
            logger.debug(f"Tool cap enforcement failed: {e}")
            return 0

    async def underperforming_tools(
        self,
        min_runs: int,
        success_floor: float,
        empty_output_floor: float | None = None,
    ) -> list[str]:
        """Active generated tools that are chronic low performers (M4).

        A tool qualifies when it has been exercised enough to judge
        (``calls >= min_runs``) yet performs poorly: ``success_rate <
        success_floor`` OR, when ``empty_output_floor`` is set,
        ``empty_output_rate >= empty_output_floor`` (a tool that "succeeds" but
        returns blank output is useless). Untried tools (``calls`` 0,
        ``success_rate`` seeded 1.0) are deliberately spared — a tool is never
        retired for performance before it has had a fair chance. Best-effort:
        any DB error degrades to an empty list (no retirement).

        Args:
            min_runs: Minimum ``calls`` before a tool is eligible.
            success_floor: Retire tools with ``success_rate`` below this.
            empty_output_floor: Optional — also retire tools whose
                ``empty_output_rate`` is at/above this (Phase 4 G). ``None``
                disables the empty-output leg (back-compat for direct callers).

        Returns:
            Names of qualifying tools (sorted).
        """
        try:
            from sqlalchemy import or_, select

            from src.db.models import ToolRegistration
            from src.db.session import get_session

            # Performance failure predicate: low success, and optionally chronic
            # empty output. ``or_`` of a single column expr collapses to that
            # expr, so empty_output_floor=None reproduces the prior query exactly.
            performance = ToolRegistration.success_rate < success_floor
            if empty_output_floor is not None:
                performance = or_(
                    performance,
                    ToolRegistration.empty_output_rate >= empty_output_floor,
                )

            async with get_session() as session:
                stmt = (
                    select(ToolRegistration.tool_name)
                    .where(
                        ToolRegistration.is_active.is_(True),
                        ToolRegistration.tool_type == "generated",
                        ToolRegistration.calls >= min_runs,
                        performance,
                    )
                )
                result = await session.execute(stmt)
                return sorted(row[0] for row in result.all())
        except Exception as e:
            logger.debug(f"Underperformer scan failed: {e}")
            return []

    async def retire_underperforming(
        self,
        min_runs: int,
        success_floor: float,
        empty_output_floor: float | None = None,
    ) -> int:
        """Retire chronic low-performing generated tools (M4 performance path).

        Delegates selection to :meth:`underperforming_tools` and retirement to
        :meth:`retire`. Returns the count retired (0 when none qualify).

        Args:
            min_runs: Minimum ``calls`` before a tool is eligible (``RETIRE_MIN_RUNS``).
            success_floor: Retire tools with ``success_rate`` below this
                (``RETIRE_SUCCESS_FLOOR``).
            empty_output_floor: Optional — also retire tools whose
                ``empty_output_rate`` is at/above this
                (``RETIRE_EMPTY_OUTPUT_FLOOR``). ``None`` disables that leg.
        """
        names = await self.underperforming_tools(
            min_runs, success_floor, empty_output_floor
        )
        if not names:
            return 0
        retired = await self.retire(names)
        if retired:
            logger.info(
                f"Retired {retired} underperforming tools "
                f"(min_runs={min_runs}, floor={success_floor}, "
                f"empty_floor={empty_output_floor}): {', '.join(names)}"
            )
        return retired

    async def unused_tools(self, min_age_days: int, max_calls: int = 0) -> list[str]:
        """Active generated tools barely invoked and older than ``min_age_days``.

        Phase-4 dead-weight scan. A tool qualifies when ``calls == 0`` (never
        exercised — ``last_run_at`` is NULL) AND it was created more than
        ``min_age_days`` ago. The age gate is the safety: a freshly-spawned tool
        that no run has picked YET is NOT retired, so "never used" is a durable
        signal rather than a transient one. Distinct from
        :meth:`underperforming_tools`, which INTENTIONALLY spares untried tools
        (a tool is never retired for *performance* before a fair chance); this
        pass targets objective dead weight no run ever selected. Best-effort: any
        DB error degrades to an empty list (no retirement).

        Args:
            min_age_days: Minimum ``created_at`` age in days. ``<= 0`` disables
                the pass (returns ``[]``).
            max_calls: Upper bound (inclusive) on ``calls`` for a tool to
                qualify. ``0`` (default) preserves the original "never invoked"
                semantics; raise to also retire low-call abandonware that would
                otherwise slowly saturate the cap alongside un-deduped
                semantic duplicates.

        Returns:
            Names of qualifying tools (sorted).
        """
        if min_age_days <= 0:
            return []
        try:
            from datetime import timedelta

            from sqlalchemy import select

            from src.db.models import ToolRegistration
            from src.db.session import get_session

            cutoff = dt.datetime.now(dt.timezone.utc) - timedelta(days=min_age_days)
            async with get_session() as session:
                stmt = (
                    select(ToolRegistration.tool_name)
                    .where(
                        ToolRegistration.is_active.is_(True),
                        ToolRegistration.tool_type == "generated",
                        ToolRegistration.calls <= max_calls,
                        ToolRegistration.created_at < cutoff,
                    )
                )
                result = await session.execute(stmt)
                return sorted(row[0] for row in result.all())
        except Exception as e:
            logger.debug(f"Unused-tool scan failed: {e}")
            return []

    async def retire_unused(self, min_age_days: int, max_calls: int = 0) -> int:
        """Retire barely-invoked generated tools older than ``min_age_days``.

        The Phase-4 complement to :meth:`retire_underperforming`: that removes
        chronic low *performers* (enough calls, poor outcomes); this removes
        objective dead weight (zero calls, aged past the gate). Delegates
        selection to :meth:`unused_tools` and retirement to :meth:`retire`.
        Returns the count retired (0 when none qualify or the pass is disabled).

        Args:
            min_age_days: Minimum ``created_at`` age in days; ``<= 0`` disables.
            max_calls: Upper bound (inclusive) on ``calls`` for a tool to
                qualify ( forwarded to :meth:`unused_tools`). ``0`` (default)
                keeps the original "never invoked" semantics.
        """
        names = await self.unused_tools(min_age_days, max_calls=max_calls)
        if not names:
            return 0
        retired = await self.retire(names)
        if retired:
            logger.info(
                f"Retired {retired} unused tools (<= {max_calls} calls, older "
                f"than {min_age_days}d): {', '.join(names)}"
            )
        return retired

    async def load_active_tools(
        self,
        registry: ToolRegistry,
        settings: AgentSettings | None = None,
    ) -> list[str]:
        """Load all active generated tools from DB and register them.

        Queries ToolRegistration where is_active=True, fetches the
        active ToolVersion, materializes the handler, and registers
        each tool in the provided ToolRegistry.

        When ``settings`` is provided, two B3 de-bloat passes run first
        (best-effort, so a DB error never blocks the load):
        :meth:`retire_redundant` marks semantic duplicates inactive, then
        :meth:`_retire_excess_tools` enforces the cumulative ``max_active_tools``
        cap (oldest first — tools lack success metrics until M4). Passing
        ``settings=None`` (existing behavior) skips both passes.

        Args:
            registry: ToolRegistry to register loaded tools into.
            settings: AgentSettings enabling cumulative caps/retirement. None
                disables enforcement (backward compatible).

        Returns:
            List of loaded tool names.
        """
        if settings is not None:
            try:
                await self.retire_redundant(
                    settings.capability_redundancy_threshold
                )
            except Exception as e:
                logger.debug(f"Tool redundancy retirement skipped: {e}")
            try:
                await self._retire_excess_tools(settings.max_active_tools)
            except Exception as e:
                logger.debug(f"Tool cap enforcement skipped: {e}")
            # M4 performance retirement: retire chronic low performers that have
            # been exercised enough to judge. Runs after redundancy/cap so the
            # metrics-driven decision sees the already-debloated population.
            # Phase 4 G: also retires chronic empty-output tools.
            try:
                await self.retire_underperforming(
                    settings.retire_min_runs,
                    settings.retire_success_floor,
                    settings.retire_empty_output_floor,
                )
            except Exception as e:
                logger.debug(f"Tool performance retirement skipped: {e}")

        loaded: list[str] = []

        try:
            from src.db.models import ToolRegistration, ToolVersion
            from src.db.session import get_session
            from sqlalchemy import select

            async with get_session() as session:
                # Find all active generated tools
                stmt = select(ToolRegistration).where(
                    ToolRegistration.is_active.is_(True),
                    ToolRegistration.tool_type == "generated",
                )
                result = await session.execute(stmt)
                registrations = result.scalars().all()

                for reg in registrations:
                    try:
                        # Get the active approved version. D10: status filter
                        # means a pending_review version (operator-edited,
                        # is_active=False until approved) is never materialized
                        # — defense-in-depth alongside is_active (approve flips
                        # both, so the live version is always approved+active).
                        ver_stmt = select(ToolVersion).where(
                            ToolVersion.tool_id == reg.id,
                            ToolVersion.is_active.is_(True),
                            ToolVersion.status == "approved",
                        ).order_by(ToolVersion.version.desc()).limit(1)
                        ver_result = await session.execute(ver_stmt)
                        version = ver_result.scalar_one_or_none()

                        if version is None:
                            logger.debug(
                                f"No active version for tool '{reg.tool_name}', skipping"
                            )
                            continue

                        # Materialize and register
                        from src.tools.dynamic.generator import ToolGenerator

                        materializer = ToolGenerator.__new__(ToolGenerator)
                        handler = materializer._materialize_handler(
                            version.code_content
                        )

                        registry.register(
                            name=reg.tool_name,
                            handler=handler,
                            description=reg.description,
                            parameters=reg.input_schema,
                            # A loaded generated tool is STILL untrusted LLM
                            # output (its source is version.code_content); tag
                            # it + carry the source so a sandboxed code-exec
                            # mode routes its invocation through the sandbox on
                            # THIS run too (otherwise a tool persisted under
                            # isolation would run in-process when recalled).
                            generated=True,
                            handler_code=version.code_content,
                        )
                        loaded.append(reg.tool_name)

                    except Exception as e:
                        logger.warning(
                            f"Failed to load tool '{reg.tool_name}': {e}"
                        )

        except Exception as e:
            logger.debug(f"Could not load dynamic tools from DB: {e}")

        return loaded


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
