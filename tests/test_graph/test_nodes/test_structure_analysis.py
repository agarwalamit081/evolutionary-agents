"""Tests for src.graph.nodes.structure_analysis — proactive gap seeding."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.agents.registry import SubAgentRegistry
from src.config import get_settings
from src.graph.enums import Phase
from src.graph.factory import initial_state
from src.graph.models import SubAgentSpec
from src.graph.nodes.structure_analysis import structure_analysis_node


class TestStructureAnalysisToolDetection:
    """Proactive tool-creation intent extraction from the goal."""

    @pytest.mark.asyncio
    async def test_extracts_quoted_tool_names(self) -> None:
        """Quoted/backticked tool identifiers become pending_tool_gaps."""
        state = initial_state(
            "Create a custom tool 'rss_aggregator' and 'html_parser'", "thread-tool-1"
        )
        result = await structure_analysis_node(state)

        assert result["structure_analysis_done"] is True
        assert result["phase"] == Phase.STRUCTURE_ANALYSIS
        gaps = result["pending_tool_gaps"]
        assert any("rss_aggregator" in g for g in gaps)
        assert any("html_parser" in g for g in gaps)

    @pytest.mark.asyncio
    async def test_tool_gaps_capped_at_max(self) -> None:
        """More than the configured max_tools_per_run named tools are capped."""
        cap = get_settings().agent.max_tools_per_run
        names = ", ".join(f"'tool_{i}'" for i in range(cap + 5))
        state = initial_state(f"Create tools {names}", "thread-tool-cap")
        result = await structure_analysis_node(state)
        assert len(result["pending_tool_gaps"]) == cap

    @pytest.mark.asyncio
    async def test_existing_tool_skipped(self) -> None:
        """A tool already registered is not re-requested."""
        tools = MagicMock()
        tools.list_names = MagicMock(return_value=["rss_aggregator"])

        state = initial_state(
            "Create a tool 'rss_aggregator' and 'brand_new_tool'", "thread-tool-exist"
        )
        result = await structure_analysis_node(state, tools=tools)

        gaps = result["pending_tool_gaps"]
        assert not any("rss_aggregator" in g for g in gaps)
        assert any("brand_new_tool" in g for g in gaps)

    @pytest.mark.asyncio
    async def test_generic_gap_when_no_names(self) -> None:
        """Intent without an explicit name seeds a single generic gap."""
        state = initial_state(
            "Build a custom utility to help with the task", "thread-tool-generic"
        )
        result = await structure_analysis_node(state)
        assert len(result["pending_tool_gaps"]) == 1


class TestStructureAnalysisAgentDetection:
    """Proactive sub-agent / parallel intent extraction."""

    @pytest.mark.asyncio
    async def test_explicit_roles_after_keyword(self) -> None:
        """'sub-agents for X and Y' yields one agent gap per role."""
        state = initial_state(
            "Use specialized sub-agents for data gathering and report generation",
            "thread-agent-roles",
        )
        result = await structure_analysis_node(state)

        gaps = result["pending_agent_gaps"]
        assert len(gaps) >= 2
        assert any("data gathering" in g for g in gaps)
        assert any("report generation" in g for g in gaps)

    @pytest.mark.asyncio
    async def test_numbered_parallel_units(self) -> None:
        """Numbered units with 'in parallel' become sub-agent gaps, capped."""
        state = initial_state(
            "Research these topics in parallel: "
            "(1) quantum computing (2) neural networks (3) blockchain",
            "thread-agent-parallel",
        )
        result = await structure_analysis_node(state)

        gaps = result["pending_agent_gaps"]
        assert len(gaps) >= 2
        assert len(gaps) <= get_settings().agent.max_sub_agents_per_run

    @pytest.mark.asyncio
    async def test_skip_when_agents_already_spawned(self) -> None:
        """Already-spawned agents suppress proactive sub-agent gaps."""
        state = initial_state(
            "Use specialized sub-agents for data gathering and report generation",
            "thread-agent-skip",
        )
        state["sub_agents_spawned"] = [{"name": "x", "id": "1"}]
        result = await structure_analysis_node(state)
        assert "pending_agent_gaps" not in result


class TestStructureAnalysisGuards:
    """Loop-safety and configuration guards."""

    @pytest.mark.asyncio
    async def test_single_shot_no_reseed(self) -> None:
        """Once structure_analysis_done is set, no gaps are re-seeded."""
        state = initial_state("Create a tool 'my_tool'", "thread-single")
        first = await structure_analysis_node(state)
        assert "pending_tool_gaps" in first

        # Simulate a later reach (the flag now persisted in state).
        state["structure_analysis_done"] = True
        second = await structure_analysis_node(state)
        assert "pending_tool_gaps" not in second

    @pytest.mark.asyncio
    async def test_dedup_vs_attempted_tool_gaps(self) -> None:
        """Attempted tool gaps block re-detection of tools."""
        state = initial_state("Create a tool 'my_tool'", "thread-dedup")
        state["attempted_tool_gaps"] = ["custom tool 'my_tool' described in the goal"]
        result = await structure_analysis_node(state)
        assert result.get("pending_tool_gaps", []) == []

    @pytest.mark.asyncio
    async def test_dedup_vs_attempted_agent_gaps(self) -> None:
        """Attempted agent gaps block re-detection of sub-agents."""
        state = initial_state(
            "Use sub-agents for data gathering and report generation",
            "thread-dedup-agent",
        )
        state["attempted_agent_gaps"] = ["specialized sub-agent for: data gathering"]
        result = await structure_analysis_node(state)
        assert result.get("pending_agent_gaps", []) == []

    @pytest.mark.asyncio
    async def test_disabled_via_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When structure_analysis_enabled is False, no gaps are seeded."""
        from src.config import get_settings

        monkeypatch.setattr(get_settings().agent, "structure_analysis_enabled", False)
        state = initial_state("Create a tool 'my_tool'", "thread-disabled")
        result = await structure_analysis_node(state)

        assert "pending_tool_gaps" not in result
        assert "pending_agent_gaps" not in result
        assert result["structure_analysis_done"] is True

    @pytest.mark.asyncio
    async def test_no_intent_yields_empty_gaps(self) -> None:
        """A plain goal with no capability intent seeds nothing."""
        state = initial_state("Explain how quicksort works", "thread-none")
        result = await structure_analysis_node(state)

        assert "pending_tool_gaps" not in result
        assert "pending_agent_gaps" not in result
        assert result["structure_analysis_done"] is True


class TestStructureAnalysisE2EGoals:
    """Regression: the real e2e validation goal texts trigger the intended gaps.

    Encodes the plan's headline criteria — Q2 must create tools, Q1/Q5 must spawn
    sub-agents — as fast unit checks so a detection regression fails CI before the
    slow e2e run (see scripts/run_e2e_validation.py QUERIES).
    """

    @pytest.mark.asyncio
    async def test_q2_creates_two_tool_gaps(self) -> None:
        """The real Q2 goal ('Create two custom tools') seeds both tool names."""
        goal = (
            "Create two custom tools: (1) an 'rss_aggregator' tool that fetches and "
            "parses RSS feeds from multiple sources, and (2) an 'html_table_generator' "
            "tool that converts structured data into formatted HTML tables."
        )
        result = await structure_analysis_node(initial_state(goal, "thread-q2"))

        gaps = result["pending_tool_gaps"]
        assert any("rss_aggregator" in g for g in gaps)
        assert any("html_table_generator" in g for g in gaps)
        # A multi-unit tool goal must not also be misread as sub-agent intent.
        assert "pending_agent_gaps" not in result

    @pytest.mark.asyncio
    async def test_q1_seeds_parallel_sub_agent_gaps(self) -> None:
        """The real Q1 goal ('...in parallel: (1)(2)(3)') seeds ≥2 sub-agent gaps."""
        goal = (
            "Research and compare three independent topics in parallel: "
            "(1) the security implications of pickle vs JSON serialization, "
            "(2) the performance trade-offs between asyncio and threading, "
            "and (3) the scalability differences between SQL and NoSQL databases."
        )
        result = await structure_analysis_node(initial_state(goal, "thread-q1"))

        gaps = result["pending_agent_gaps"]
        assert len(gaps) >= 2
        assert len(gaps) <= get_settings().agent.max_sub_agents_per_run

    @pytest.mark.asyncio
    async def test_q5_seeds_named_sub_agent_roles(self) -> None:
        """The real Q5 goal ('sub-agents for data gathering and report generation')."""
        goal = (
            "Perform a comprehensive analysis of renewable energy trends. "
            "Use specialized sub-agents for data gathering and report generation."
        )
        result = await structure_analysis_node(initial_state(goal, "thread-q5"))

        gaps = result["pending_agent_gaps"]
        assert any("data gathering" in g for g in gaps)
        assert any("report generation" in g for g in gaps)


class TestStructureAnalysisSuppressOverSpawn:
    """battery-02 N8: a goal that references recalled sub-agents by name must NOT
    proactively spawn a redundant helper — delegate reuses the recalled ones.

    Root cause: "Using the doc_outline and python_file_inventory sub-agents
    (created earlier)..." matched the "sub-agents" keyword with no explicit
    roles, fell back to the generic "an independent subtask" gap, and needlessly
    spawned repo_map_builder though both named agents were already recalled.
    """

    @staticmethod
    def _registry_with(*names: str) -> SubAgentRegistry:
        registry = SubAgentRegistry()
        for name in names:
            registry.register(SubAgentSpec(
                name=name, goal=f"{name} task", parent_thread_id="t",
            ))
        return registry

    @pytest.mark.asyncio
    async def test_n8_goal_suppresses_spawn_when_agents_recalled(self) -> None:
        """The N8 goal names recalled agents → no proactive agent_spawn gap."""
        registry = self._registry_with("doc_outline", "python_file_inventory")
        state = initial_state(
            "Using the doc_outline and python_file_inventory sub-agents "
            "(created earlier), build a combined repo map.",
            "thread-n8",
        )
        result = await structure_analysis_node(state, sub_agent_registry=registry)
        assert result.get("pending_agent_gaps", []) == []

    @pytest.mark.asyncio
    async def test_same_goal_spawns_without_registry(self) -> None:
        """Positive control: same goal, no recalled agents → generic gap fires."""
        state = initial_state(
            "Using the doc_outline and python_file_inventory sub-agents "
            "(created earlier), build a combined repo map.",
            "thread-n8-noreg",
        )
        result = await structure_analysis_node(state)  # no registry
        assert "pending_agent_gaps" in result
        assert len(result["pending_agent_gaps"]) >= 1

    @pytest.mark.asyncio
    async def test_only_suppresses_when_named_agent_actually_recalled(self) -> None:
        """A snake_case token that is NOT a recalled agent does not suppress."""
        # Goal names doc_outline, but the registry holds a different agent.
        registry = self._registry_with("other_agent")
        state = initial_state(
            "Using the doc_outline sub-agents (created earlier), build a map.",
            "thread-n8-mismatch",
        )
        result = await structure_analysis_node(state, sub_agent_registry=registry)
        assert "pending_agent_gaps" in result
        assert len(result["pending_agent_gaps"]) >= 1

    @pytest.mark.asyncio
    async def test_explicit_new_roles_still_spawn_despite_recalled_agents(self) -> None:
        """Explicit 'sub-agents for X and Y' roles spawn even when recalled
        agents exist — they are genuinely new gaps, not the generic fallback."""
        registry = self._registry_with("doc_outline")
        state = initial_state(
            "Use specialized sub-agents for data gathering and report generation",
            "thread-roles-with-registry",
        )
        result = await structure_analysis_node(state, sub_agent_registry=registry)
        gaps = result["pending_agent_gaps"]
        assert any("data gathering" in g for g in gaps)
        assert any("report generation" in g for g in gaps)
