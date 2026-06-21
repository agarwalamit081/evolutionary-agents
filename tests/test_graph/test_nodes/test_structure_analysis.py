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


class TestStructureAnalysisJsonKeySuppression:
    """battery-04 q07 regression: a goal that specifies a deliverable's JSON
    shape with quoted keys (``{"engineers": [...], "capacity_hours": number}``)
    must NOT seed phantom tools named after those keys.

    Root cause: ``_TOOL_NAME_RE`` matched ANY quoted snake_case identifier, so a
    schema-heavy goal (which legitimately says "write a script") flooded
    tool_create with tools named ``engineers``/``capacity_hours``/``tasks`` —
    each ~60-90s of wasted codegen + DB persistence — and the run never
    converged. The negative lookahead skips a quoted identifier immediately
    followed by ``:`` (a dict/JSON key).
    """

    @pytest.mark.asyncio
    async def test_json_schema_keys_not_treated_as_tools(self) -> None:
        """Quoted keys followed by ':' are JSON keys, not tool names."""
        goal = (
            "Write a Python script that emits results/x.json with shape "
            '{"engineers": [{"id": "e1", "capacity_hours": 10, "skills": ["sql"]}], '
            '"tasks": [{"id": "t1", "required_hours": 4, "required_skill": "sql", '
            '"priority": 5}]}. The script must maximize assigned priority.'
        )
        result = await structure_analysis_node(initial_state(goal, "thread-jsonkeys"))
        gaps = result.get("pending_tool_gaps", [])
        for key in (
            "engineers", "capacity_hours", "skills", "tasks",
            "required_hours", "required_skill", "priority",
        ):
            assert not any(key in g for g in gaps), f"{key!r} wrongly flagged as a tool"

    @pytest.mark.asyncio
    async def test_quoted_tool_name_not_a_key_still_extracted(self) -> None:
        """Positive control: the colon-suppression must not over-fire — a
        quoted tool name NOT followed by ':' is still extracted."""
        goal = "Create a custom tool 'csv_exporter' that writes rows to a file."
        result = await structure_analysis_node(initial_state(goal, "thread-pos"))
        assert any("csv_exporter" in g for g in result["pending_tool_gaps"])

    @pytest.mark.asyncio
    async def test_q07_goal_seeds_no_phantom_tools(self) -> None:
        """The real q07 goal (solver script + JSON schema) seeds no JSON-key tools."""
        goal = (
            "Generate a constraint-satisfaction instance and write it to "
            'results/q07/instance.json with shape {"engineers": [{"id": str, '
            '"capacity_hours": number, "skills": [str]}], "tasks": [{"id": str, '
            '"required_hours": number, "required_skill": str, "priority": int}]}. '
            "Write ONE Python solver script and execute it. objective_value must "
            'equal the assigned-priority sum. Write {"assignments": [...], '
            '"objective_value": number} to solution.json.'
        )
        result = await structure_analysis_node(initial_state(goal, "thread-q07"))
        gaps = result.get("pending_tool_gaps", [])
        for key in ("engineers", "capacity_hours", "skills", "tasks",
                    "required_hours", "required_skill", "priority",
                    "assignments", "objective_value"):
            assert not any(key in g for g in gaps), f"{key!r} wrongly flagged as a tool"


class TestStructureAnalysisColumnNameSuppression:
    """battery-04 q09 regression: a backticked CSV COLUMN / data-field name in a
    tool-creation goal must NOT seed a phantom tool.

    Root cause: ``_TOOL_NAME_RE`` matched ANY quoted snake_case identifier, so the
    q09 goal ('run the new iqr_outlier_detector tool on column `amount`') flagged
    ``amount`` as "custom tool 'amount' described in the goal", routing the run
    into a spurious tool_create -> replan cycle. Real custom-tool names in this
    codebase are always multi-word snake_case (underscored), so requiring an
    underscore — mirroring ``_AGENT_NAME_RE`` — suppresses bare column/field words
    while keeping ``iqr_outlier_detector``/``csv_exporter``/``rss_aggregator``.
    """

    @pytest.mark.asyncio
    async def test_q09_csv_column_not_treated_as_tool(self) -> None:
        """The real q09 goal flags only the underscored tool name, not `amount`."""
        goal = (
            "Create a new tool 'iqr_outlier_detector' that computes Tukey IQR "
            "fences. Run it on transactions.csv column `amount` and write the "
            "fences to anomalies.json. Also reference the `quantity` and "
            "`timestamp` columns for context."
        )
        result = await structure_analysis_node(initial_state(goal, "thread-q09-col"))
        gaps = result["pending_tool_gaps"]
        assert any("iqr_outlier_detector" in g for g in gaps)
        for col in ("amount", "quantity", "timestamp"):
            assert not any(col in g for g in gaps), (
                f"column {col!r} wrongly flagged as a tool: {gaps}"
            )

    @pytest.mark.asyncio
    async def test_bare_quoted_word_intent_falls_back_to_generic(self) -> None:
        """A tool-creation goal whose only quoted name is a bare (underscore-free)
        word seeds the generic gap (LLM derives the name), not a phantom tool."""
        goal = "Build a custom tool 'parser' that reads the input."
        result = await structure_analysis_node(initial_state(goal, "thread-bare"))
        gaps = result["pending_tool_gaps"]
        assert len(gaps) == 1
        assert "parser" not in gaps[0]


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


class TestStructureAnalysisAgentRoleScope:
    """battery-04 q07 regression: a goal that says 'sub-agent' and then, in a
    LATER sentence, 'for each confirms…' must NOT seed phantom sub-agents from
    that unrelated 'for …' clause.

    Root cause: ``_extract_roles_after_keyword`` scanned the WHOLE tail after
    the keyword for a 'for/covering' clause, across sentence boundaries. The
    q07 goal ('adversarial sub-agent that tries to BREAK the solution. … for
    each confirms the constraint checker WOULD flag it, then confirms the
    ORIGINAL solution has zero such violations.') matched 'for each confirms…'
    and seeded two phantom sub-agents ('for each confirms…' / 'then confirms…'),
    routing the run into a spurious delegate cycle that then hit the
    reflect↔execute tight loop.
    """

    @pytest.mark.asyncio
    async def test_q07_adversarial_prose_seeds_no_phantom_roles(self) -> None:
        """The real q07 adversarial instructions seed no 'confirms…' roles."""
        goal = (
            "Then SPAWN an adversarial sub-agent that tries to BREAK the "
            "solution. The adversarial sub-agent MUST itself load instance.json "
            "and solution.json and test the REAL assignments: it performs >= 2 "
            "attacks (e.g. move a task to an engineer lacking the required_skill, "
            "push an engineer past capacity_hours) and for each confirms the "
            "constraint checker WOULD flag it, then confirms the ORIGINAL "
            "solution has zero such violations."
        )
        result = await structure_analysis_node(initial_state(goal, "thread-q07-adv"))
        gaps = result.get("pending_agent_gaps", [])
        for phantom in ("confirms", "constraint checker", "ORIGINAL"):
            assert not any(phantom in g for g in gaps), (
                f"phantom sub-agent role {phantom!r} seeded from a later-sentence "
                f"'for …' clause: {gaps}"
            )

    @pytest.mark.asyncio
    async def test_same_sentence_role_still_extracted(self) -> None:
        """Positive control: a 'for …' clause in the SAME sentence as the
        keyword is still extracted — the sentence-scope fix must not over-fire."""
        goal = "Use specialized sub-agents for data gathering and report generation."
        result = await structure_analysis_node(initial_state(goal, "thread-same-sent"))
        gaps = result["pending_agent_gaps"]
        assert any("data gathering" in g for g in gaps)
        assert any("report generation" in g for g in gaps)

    @pytest.mark.asyncio
    async def test_q07_full_goal_seeds_no_constraint_phantom_roles(self) -> None:
        """The real q07 goal combines an 'adversarial sub-agent' mention with
        an unrelated (1)(2)(3) HARD-constraint list and a later-sentence
        'for …' clause. None of these may seed phantom sub-agents — only the
        explicit same-sentence role pattern (absent here) or the single
        generic gap is allowed.

        Regression for BOTH the sentence-scope fix (no 'for each confirms…'
        role) and the numbered-fallback removal (no 'each task assigned…',
        'required_skill', 'required_hours' roles from the constraint list).
        """
        goal = (
            "HARD constraints the assignment must satisfy: "
            "(1) each task assigned to at most one engineer, "
            "(2) the engineer possesses the task's required_skill, "
            "(3) the sum of required_hours assigned to an engineer never "
            "exceeds their capacity_hours. "
            "Then SPAWN an adversarial sub-agent that tries to BREAK the "
            "solution. It performs attacks and for each confirms the "
            "constraint checker WOULD flag it."
        )
        result = await structure_analysis_node(initial_state(goal, "thread-q07-full"))
        gaps = result.get("pending_agent_gaps", [])
        for phantom in (
            "each task assigned", "required_skill", "required_hours",
            "capacity_hours", "confirms", "constraint checker",
        ):
            assert not any(phantom in g for g in gaps), (
                f"phantom sub-agent role {phantom!r} seeded from goal prose: "
                f"{gaps}"
            )
