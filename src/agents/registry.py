"""In-memory registry for sub-agent definitions.

Loaded from DB at startup via SubAgentPersister.load_active_agents().
New sub-agents are registered here after persisting to DB.

Mirrors the pattern from src/tools/registry.py (ToolRegistry).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from src.graph.models import SubAgentSpec

if TYPE_CHECKING:
    from src.agents.runner import SubAgentRunner
    from src.llm.gateway import LLMGateway
    from src.memory.manager import MemoryManager
    from src.tools.registry import ToolRegistry


# ── Limits ──────────────────────────────────────────────────────────────

MAX_SUB_AGENTS_PER_RUN: int = 3
DEPRECATION_SUCCESS_RATE_THRESHOLD: float = 0.3
DEPRECATION_MIN_RUNS: int = 10


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

    # ── Description for LLM ────────────────────────────────────────────

    def describe_agents(self) -> str:
        """Generate a text description of active agents for LLM prompts.

        Returns:
            Formatted string describing each active sub-agent's
            name, description, tool scope, and success rate.
        """
        active = self.list_active()
        if not active:
            return "No active sub-agents available."

        lines: list[str] = []
        for agent in active:
            lines.append(
                f"- **{agent.name}**: {agent.description} "
                f"(success_rate={agent.success_rate:.0%}, "
                f"runs={agent.total_runs}, "
                f"tools={agent.tool_scope})"
            )
        return "\n".join(lines)

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

    def check_deprecation(self, name: str) -> bool:
        """Check if a sub-agent should be auto-deprecated.

        Deprecation criteria:
            - At least DEPRECATION_MIN_RUNS total runs
            - success_rate < DEPRECATION_SUCCESS_RATE_THRESHOLD

        Args:
            name: Sub-agent name to check.

        Returns:
            True if the agent was deprecated.
        """
        spec = self._agents.get(name)
        if spec is None:
            return False

        if spec.total_runs < DEPRECATION_MIN_RUNS:
            return False

        if spec.success_rate < DEPRECATION_SUCCESS_RATE_THRESHOLD:
            logger.warning(
                f"Auto-deprecating sub-agent '{name}': "
                f"success_rate={spec.success_rate:.2f} "
                f"over {spec.total_runs} runs"
            )
            spec.is_active = False
            return True

        return False

    @property
    def count(self) -> int:
        """Number of registered sub-agents."""
        return len(self._agents)

    @property
    def active_count(self) -> int:
        """Number of active sub-agents."""
        return len(self.list_active())
