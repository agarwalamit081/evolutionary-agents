"""Tests for SubAgentRegistry from src.agents.registry."""

from __future__ import annotations


import pytest

from src.agents.registry import (
    SubAgentRegistry,
)
from src.graph.models import SubAgentSpec


@pytest.fixture
def sample_spec() -> SubAgentSpec:
    """Create a sample SubAgentSpec for testing."""
    return SubAgentSpec(
        name="test_agent",
        description="Test sub-agent for unit tests",
        goal="test goal",
        parent_thread_id="test-thread-001",
        tool_scope="inherit_all",
    )


@pytest.fixture
def inactive_spec() -> SubAgentSpec:
    """Create an inactive SubAgentSpec."""
    return SubAgentSpec(
        name="inactive_agent",
        description="Inactive sub-agent",
        goal="inactive goal",
        parent_thread_id="test-thread-002",
        is_active=False,
    )


@pytest.fixture
def registry() -> SubAgentRegistry:
    """Create a fresh SubAgentRegistry for each test."""
    return SubAgentRegistry()


class TestRegisterAndGet:
    """Tests for register() and get() methods."""

    def test_register_and_get(self, registry: SubAgentRegistry, sample_spec: SubAgentSpec) -> None:
        """Register a SubAgentSpec, get it back."""
        registry.register(sample_spec)
        retrieved = registry.get("test_agent")
        assert retrieved is not None
        assert retrieved.name == "test_agent"
        assert retrieved.description == "Test sub-agent for unit tests"

    def test_register_overwrites(self, registry: SubAgentRegistry, sample_spec: SubAgentSpec) -> None:
        """Registering same name overwrites existing spec."""
        # Register first version
        registry.register(sample_spec)
        first = registry.get("test_agent")
        assert first is not None
        assert first.description == "Test sub-agent for unit tests"

        # Register second version with different description
        updated_spec = SubAgentSpec(
            name="test_agent",
            description="Updated description",
            goal="updated goal",
            parent_thread_id="test-thread-003",
            tool_scope="inherit_all",
        )
        registry.register(updated_spec)
        second = registry.get("test_agent")
        assert second is not None
        assert second.description == "Updated description"

    def test_get_unknown_returns_none(self, registry: SubAgentRegistry) -> None:
        """Getting an unknown agent returns None."""
        assert registry.get("unknown_agent") is None


class TestHas:
    """Tests for has() method."""

    def test_has_returns_true_for_registered(self, registry: SubAgentRegistry, sample_spec: SubAgentSpec) -> None:
        """has() returns True for registered agents."""
        registry.register(sample_spec)
        assert registry.has("test_agent") is True

    def test_has_returns_false_for_unknown(self, registry: SubAgentRegistry) -> None:
        """has() returns False for unknown agents."""
        assert registry.has("unknown_agent") is False


class TestListAgents:
    """Tests for list_agents() method."""

    def test_list_agents_returns_all_registered(self, registry: SubAgentRegistry) -> None:
        """list_agents() returns all registered agents."""
        spec1 = SubAgentSpec(
            name="agent_one",
            description="First agent",
            goal="goal1",
            parent_thread_id="thread-001",
            tool_scope="inherit_all",
        )
        spec2 = SubAgentSpec(
            name="agent_two",
            description="Second agent",
            goal="goal2",
            parent_thread_id="thread-002",
            tool_scope="inherit_all",
        )
        registry.register(spec1)
        registry.register(spec2)

        agents = registry.list_agents()
        assert len(agents) == 2
        names = {a.name for a in agents}
        assert names == {"agent_one", "agent_two"}


class TestListActive:
    """Tests for list_active() method."""

    def test_list_active_filters_inactive(self, registry: SubAgentRegistry, sample_spec: SubAgentSpec, inactive_spec: SubAgentSpec) -> None:
        """list_active() returns only active agents."""
        registry.register(sample_spec)
        registry.register(inactive_spec)

        active = registry.list_active()
        assert len(active) == 1
        assert active[0].name == "test_agent"
        assert active[0].is_active is True

    def test_list_active_returns_empty_when_none_active(self, registry: SubAgentRegistry, inactive_spec: SubAgentSpec) -> None:
        """list_active() returns empty list when no agents are active."""
        registry.register(inactive_spec)
        assert registry.list_active() == []


class TestUnregister:
    """Tests for unregister() method."""

    def test_unregister_removes_agent(self, registry: SubAgentRegistry, sample_spec: SubAgentSpec) -> None:
        """unregister() removes agent, returns True."""
        registry.register(sample_spec)
        assert registry.has("test_agent") is True

        result = registry.unregister("test_agent")
        assert result is True
        assert registry.has("test_agent") is False

    def test_unregister_returns_false_for_unknown(self, registry: SubAgentRegistry) -> None:
        """unregister() returns False for unknown agent."""
        assert registry.unregister("unknown_agent") is False


class TestCheckDeprecation:
    """Tests for check_deprecation() method."""

    def test_check_deprecation_no_runs(self, registry: SubAgentRegistry) -> None:
        """check_deprecation() returns False when total_runs < threshold."""
        spec = SubAgentSpec(
            name="new_agent",
            description="New agent",
            goal="new goal",
            parent_thread_id="thread-001",
            tool_scope="inherit_all",
            total_runs=5,  # Below DEPRECATION_MIN_RUNS (10)
            success_rate=0.1,  # Low success rate
        )
        registry.register(spec)

        result = registry.check_deprecation("new_agent")
        assert result is False
        assert registry.get("new_agent").is_active is True  # Still active

    def test_check_deprecation_below_threshold(self, registry: SubAgentRegistry) -> None:
        """check_deprecation() returns True when success_rate < threshold and runs >= threshold."""
        spec = SubAgentSpec(
            name="failing_agent",
            description="Failing agent",
            goal="failing goal",
            parent_thread_id="thread-002",
            tool_scope="inherit_all",
            total_runs=15,  # Above DEPRECATION_MIN_RUNS
            success_rate=0.2,  # Below DEPRECATION_SUCCESS_RATE_THRESHOLD (0.3)
        )
        registry.register(spec)

        result = registry.check_deprecation("failing_agent")
        assert result is True
        assert registry.get("failing_agent").is_active is False  # Now inactive

    def test_check_deprecation_above_threshold(self, registry: SubAgentRegistry) -> None:
        """check_deprecation() returns False when success_rate >= threshold."""
        spec = SubAgentSpec(
            name="good_agent",
            description="Good agent",
            goal="good goal",
            parent_thread_id="thread-003",
            tool_scope="inherit_all",
            total_runs=20,
            success_rate=0.8,  # Above threshold
        )
        registry.register(spec)

        result = registry.check_deprecation("good_agent")
        assert result is False
        assert registry.get("good_agent").is_active is True  # Still active

    def test_check_deprecation_unknown_agent(self, registry: SubAgentRegistry) -> None:
        """check_deprecation() returns False for unknown agent."""
        result = registry.check_deprecation("unknown_agent")
        assert result is False

    def test_check_deprecation_at_threshold_boundary(self, registry: SubAgentRegistry) -> None:
        """check_deprecation() handles boundary case at exact threshold."""
        spec = SubAgentSpec(
            name="boundary_agent",
            description="Boundary agent",
            goal="boundary goal",
            parent_thread_id="thread-004",
            tool_scope="inherit_all",
            total_runs=10,  # Exactly DEPRECATION_MIN_RUNS
            success_rate=0.3,  # Exactly DEPRECATION_SUCCESS_RATE_THRESHOLD
        )
        registry.register(spec)

        result = registry.check_deprecation("boundary_agent")
        assert result is False  # At threshold, not deprecated


class TestCountProperties:
    """Tests for count and active_count properties."""

    def test_count_property(self, registry: SubAgentRegistry) -> None:
        """count property returns number of registered agents."""
        assert registry.count == 0

        for i in range(5):
            spec = SubAgentSpec(
                name=f"agent_{i}",
                description=f"Agent {i}",
                goal=f"goal_{i}",
                parent_thread_id=f"thread-{i}",
                tool_scope="inherit_all",
                is_active=(i % 2 == 0),  # Half active
            )
            registry.register(spec)

        assert registry.count == 5

    def test_active_count_property(self, registry: SubAgentRegistry) -> None:
        """active_count property returns number of active agents."""
        assert registry.active_count == 0

        # Register 5 agents, 3 active and 2 inactive
        for i in range(5):
            spec = SubAgentSpec(
                name=f"agent_{i}",
                description=f"Agent {i}",
                goal=f"goal_{i}",
                parent_thread_id=f"thread-{i}",
                tool_scope="inherit_all",
                is_active=(i < 3),  # First 3 active
            )
            registry.register(spec)

        assert registry.active_count == 3

    def test_active_count_updates_on_deactivation(self, registry: SubAgentRegistry) -> None:
        """active_count updates when agent is deprecated."""
        spec = SubAgentSpec(
            name="to_deprecate",
            description="Will be deprecated",
            goal="deprecation goal",
            parent_thread_id="thread-001",
            tool_scope="inherit_all",
            total_runs=20,
            success_rate=0.2,
        )
        registry.register(spec)
        assert registry.active_count == 1

        registry.check_deprecation("to_deprecate")
        assert registry.active_count == 0


class TestSpawn:
    """Tests for spawn() method."""

    def test_spawn_returns_none_for_unknown(self, registry: SubAgentRegistry) -> None:
        """spawn() returns None for unknown agent."""
        from unittest.mock import MagicMock

        gateway = MagicMock()
        tools = MagicMock()

        result = registry.spawn(
            name="unknown",
            goal="test goal",
            parent_thread_id="thread-001",
            gateway=gateway,
            tools=tools,
        )
        assert result is None

    def test_spawn_returns_none_for_inactive(self, registry: SubAgentRegistry, inactive_spec: SubAgentSpec) -> None:
        """spawn() returns None for inactive agent."""
        from unittest.mock import MagicMock

        registry.register(inactive_spec)
        gateway = MagicMock()
        tools = MagicMock()

        result = registry.spawn(
            name="inactive_agent",
            goal="test goal",
            parent_thread_id="thread-001",
            gateway=gateway,
            tools=tools,
        )
        assert result is None

    def test_spawn_creates_runner(self, registry: SubAgentRegistry, sample_spec: SubAgentSpec) -> None:
        """spawn() creates SubAgentRunner for active agent."""
        from unittest.mock import MagicMock

        registry.register(sample_spec)
        gateway = MagicMock()
        tools = MagicMock()

        result = registry.spawn(
            name="test_agent",
            goal="subtask goal",
            parent_thread_id="thread-001",
            gateway=gateway,
            tools=tools,
        )
        assert result is not None
        assert result.definition.name == "test_agent"
