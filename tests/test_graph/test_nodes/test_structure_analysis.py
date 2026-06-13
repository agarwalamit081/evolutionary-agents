"""Tests for src.graph.nodes.structure_analysis — proactive gap seeding."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.agents.registry import MAX_SUB_AGENTS_PER_RUN
from src.graph.enums import Phase
from src.graph.factory import initial_state
from src.graph.nodes.structure_analysis import structure_analysis_node
from src.tools.dynamic.allowlist import MAX_TOOLS_PER_RUN


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
        """More than MAX_TOOLS_PER_RUN named tools are capped."""
        state = initial_state(
            "Create tools 'alpha_tool', 'beta_tool', 'gamma_tool', 'delta_tool'",
            "thread-tool-cap",
        )
        result = await structure_analysis_node(state)
        assert len(result["pending_tool_gaps"]) == MAX_TOOLS_PER_RUN

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
        assert len(gaps) <= MAX_SUB_AGENTS_PER_RUN

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
        assert len(gaps) <= MAX_SUB_AGENTS_PER_RUN

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
