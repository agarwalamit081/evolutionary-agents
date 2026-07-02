"""In-memory registry for sub-agent definitions.

Loaded from DB at startup via SubAgentPersister.load_active_agents().
New sub-agents are registered here after persisting to DB.

Mirrors the pattern from src/tools/registry.py (ToolRegistry).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from loguru import logger

from src.graph.models import SubAgentSpec

if TYPE_CHECKING:
    from src.agents.runner import SubAgentRunner
    from src.llm.gateway import LLMGateway
    from src.memory.manager import MemoryManager
    from src.tools.registry import ToolRegistry


def _is_stale(
    last_used_at: datetime | None,
    recency_days: int,
    now: datetime | None,
) -> bool:
    """True when ``last_used_at`` is known and older than ``recency_days``.

    A None ``last_used_at`` (never persisted/used) is NOT stale — a brand-new
    capability should not be retired just because it has no usage history yet.
    Naive datetimes are treated as UTC for the comparison.
    """
    if last_used_at is None or recency_days <= 0:
        return False
    reference = now if now is not None else datetime.now(timezone.utc)
    last = (
        last_used_at.replace(tzinfo=timezone.utc)
        if last_used_at.tzinfo is None
        else last_used_at
    )
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    return (reference - last) > timedelta(days=recency_days)


# ── Limits ──────────────────────────────────────────────────────────────

DEPRECATION_SUCCESS_RATE_THRESHOLD: float = 0.3
DEPRECATION_MIN_RUNS: int = 10
# Stale-capability window for the runtime check_deprecation default (days). The
# cumulative-cap path (enforce_caps) uses the overridable
# AgentSettings.retire_recency_days; this constant is only the legacy default.
DEPRECATION_RECENCY_DAYS: int = 30


class SubAgentRegistry:
    """In-memory registry for sub-agent definitions.

    Stores SubAgentSpec instances keyed by name. Provides lookup,
    listing, and spawn capabilities. Loaded from PostgreSQL at startup
    and updated when new sub-agents are created at runtime.
    """

    def __init__(self) -> None:
        self._agents: dict[str, SubAgentSpec] = {}

    # ── CRUD ───────────────────────────────────────────────────────────

    def register(self, spec: SubAgentSpec) -> None:
        """Register a sub-agent definition.

        Args:
            spec: Complete sub-agent specification.
        """
        name = spec.name
        if name in self._agents:
            logger.warning(f"Sub-agent '{name}' already registered, overwriting")
        self._agents[name] = spec
        logger.debug(f"Sub-agent registered: {name}")

    def get(self, name: str) -> SubAgentSpec | None:
        """Get a sub-agent by name."""
        return self._agents.get(name)

    def has(self, name: str) -> bool:
        """Check if a sub-agent is registered."""
        return name in self._agents

    def unregister(self, name: str) -> bool:
        """Remove a sub-agent from the registry.

        Returns:
            True if the agent was found and removed.
        """
        if name in self._agents:
            del self._agents[name]
            logger.debug(f"Sub-agent unregistered: {name}")
            return True
        return False

    # ── Listing ────────────────────────────────────────────────────────

    def list_agents(self) -> list[SubAgentSpec]:
        """List all registered sub-agents."""
        return list(self._agents.values())

    def list_active(self) -> list[SubAgentSpec]:
        """List only active sub-agents."""
        return [a for a in self._agents.values() if a.is_active]

    def list_names(self) -> list[str]:
        """List all registered sub-agent names."""
        return list(self._agents.keys())

    # ── Spawn ──────────────────────────────────────────────────────────

    def spawn(
        self,
        name: str,
        goal: str,
        parent_thread_id: str,
        gateway: LLMGateway,
        tools: ToolRegistry,
        memory: MemoryManager | None = None,
    ) -> SubAgentRunner | None:
        """Create a SubAgentRunner from a registered definition.

        Args:
            name: Sub-agent name to spawn.
            goal: Specific subtask goal for this invocation.
            parent_thread_id: Parent's thread ID for tracking.
            gateway: LLMGateway for LLM calls within sub-agent.
            tools: Parent's ToolRegistry (will be scoped).
            memory: Optional MemoryManager (isolated for sub-agent).

        Returns:
            SubAgentRunner ready for execution, or None if not found.
        """
        from src.agents.runner import SubAgentRunner

        spec = self._agents.get(name)
        if spec is None:
            logger.warning(f"Cannot spawn unknown sub-agent: {name}")
            return None

        if not spec.is_active:
            logger.warning(f"Cannot spawn inactive sub-agent: {name}")
            return None

        return SubAgentRunner(
            definition=spec,
            gateway=gateway,
            tools=tools,
            memory=memory,
        )

    # ── Auto-Deprecation ───────────────────────────────────────────────

    def check_deprecation(
        self,
        name: str,
        *,
        min_runs: int = DEPRECATION_MIN_RUNS,
        success_floor: float = DEPRECATION_SUCCESS_RATE_THRESHOLD,
        recency_days: int = DEPRECATION_RECENCY_DAYS,
        now: datetime | None = None,
    ) -> bool:
        """Check if a sub-agent should be auto-deprecated and retire it if so.

        A capability is retired when EITHER retirement trigger fires:

            - **Chronic low performer**: ``total_runs >= min_runs`` AND
              ``success_rate < success_floor`` (enough data to be confident it
              is bad).
            - **Stale**: ``last_used_at`` is known and older than
              ``recency_days`` (dead weight — unused for the window).

        The defaults are the legacy module constants (min_runs=10,
        success_floor=0.3, recency_days=30) so existing single-arg callers
        (e.g. ``delegate`` at runtime) behave as before *plus* the new stale
        trigger. The cumulative-cap path (:meth:`enforce_caps`) passes the
        stricter ``AgentSettings.retire_*`` values.

        Args:
            name: Sub-agent name to check.
            min_runs: Minimum runs before the success trigger is trusted.
            success_floor: success_rate below this (with enough runs) retires.
            recency_days: Unused-for-this-many-days retires (stale trigger).
            now: Override "now" (UTC) for deterministic tests.

        Returns:
            True if the agent was retired (``is_active`` set False).
        """
        spec = self._agents.get(name)
        if spec is None:
            return False

        is_bad = (
            spec.total_runs >= min_runs and spec.success_rate < success_floor
        )
        is_stale = _is_stale(spec.last_used_at, recency_days, now)
        if not (is_bad or is_stale):
            return False

        reason = "stale" if (is_stale and not is_bad) else (
            "low-performer" if is_bad and not is_stale else "low-performer+stale"
        )
        logger.warning(
            f"Retiring sub-agent '{name}' ({reason}): "
            f"success_rate={spec.success_rate:.2f} "
            f"over {spec.total_runs} runs, last_used={spec.last_used_at}"
        )
        spec.is_active = False
        return True

    def enforce_caps(
        self,
        *,
        max_active: int,
        min_runs: int = DEPRECATION_MIN_RUNS,
        success_floor: float = DEPRECATION_SUCCESS_RATE_THRESHOLD,
        recency_days: int = DEPRECATION_RECENCY_DAYS,
        now: datetime | None = None,
    ) -> list[str]:
        """Retire active sub-agents that are bad/stale, then enforce the cap.

        Runs :meth:`check_deprecation` over every active agent (retiring chronic
        low performers and stale ones), then — if still over ``max_active`` —
        retires the lowest-scoring survivors until at/under the cap. Scoring is
        ``(success_rate, total_runs, quality_score)`` compared as a tuple
        (higher is better); the lowest tuples go first. Redundancy retirement
        (semantic duplicates) is handled separately at the DB layer by
        ``SubAgentPersister.retire_redundant`` before load.

        Args:
            max_active: Maximum number of active agents to keep.
            min_runs/success_floor/recency_days: passed to check_deprecation.
            now: Override "now" (UTC) for deterministic tests.

        Returns:
            Names of agents retired by this call, in retirement order.
        """
        retired: list[str] = []
        for spec in list(self.list_active()):
            if self.check_deprecation(
                spec.name,
                min_runs=min_runs,
                success_floor=success_floor,
                recency_days=recency_days,
                now=now,
            ):
                retired.append(spec.name)

        active = self.list_active()
        if len(active) <= max_active:
            return retired

        # Lowest score first: worst success_rate, then fewest runs, then lowest
        # quality. Stable sort keeps insertion order among ties.
        overflow = sorted(
            active,
            key=lambda s: (s.success_rate, s.total_runs, s.quality_score),
        )
        for spec in overflow[: len(active) - max_active]:
            spec.is_active = False
            retired.append(spec.name)
            logger.info(
                f"Retiring sub-agent '{spec.name}' to enforce cap "
                f"{max_active} (score={spec.success_rate:.2f}/"
                f"{spec.total_runs}/{spec.quality_score:.2f})"
            )
        return retired

    @property
    def count(self) -> int:
        """Number of registered sub-agents."""
        return len(self._agents)

    @property
    def active_count(self) -> int:
        """Number of active sub-agents."""
        return len(self.list_active())
