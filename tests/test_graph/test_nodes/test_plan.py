"""Tests for src.graph.nodes.plan — plan node function."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.graph.enums import GoalStatus, Phase, Strategy, TaskComplexity
from src.graph.factory import initial_state
from src.graph.models import Goal
from src.graph.nodes.plan import plan_node
from src.llm.models import LLMResponse


class TestPlanNode:
    """Tests for the plan_node async function."""

    @pytest.mark.asyncio
    async def test_plan_react_strategy(self) -> None:
        """REACT strategy generates a 3-step plan."""
        state = initial_state("implement a REST API", "thread-react")
        state["strategy"] = Strategy.REACT
        result = await plan_node(state)

        assert result["phase"] == Phase.RETRIEVE_MEMORY
        steps = result["plan_steps"]
        assert len(steps) == 3
        assert all(s.status == GoalStatus.PENDING for s in steps)
        assert result["current_step_index"] == 0

    @pytest.mark.asyncio
    async def test_plan_direct_strategy(self) -> None:
        """DIRECT strategy generates a single-step plan."""
        state = initial_state("define variable", "thread-direct")
        state["strategy"] = Strategy.DIRECT
        result = await plan_node(state)

        assert result["phase"] == Phase.RETRIEVE_MEMORY
        assert len(result["plan_steps"]) == 1

    @pytest.mark.asyncio
    async def test_plan_planning_strategy(self) -> None:
        """PLANNING strategy generates a 4-step plan."""
        state = initial_state("build end-to-end pipeline", "thread-planning")
        state["strategy"] = Strategy.PLANNING
        result = await plan_node(state)

        assert result["phase"] == Phase.RETRIEVE_MEMORY
        assert len(result["plan_steps"]) == 4

    @pytest.mark.asyncio
    async def test_plan_reflection_strategy(self) -> None:
        """REFLECTION strategy generates a 4-step plan."""
        state = initial_state("review and improve code", "thread-reflection")
        state["strategy"] = Strategy.REFLECTION
        result = await plan_node(state)

        assert result["phase"] == Phase.RETRIEVE_MEMORY
        assert len(result["plan_steps"]) == 4

    @pytest.mark.asyncio
    async def test_plan_tot_strategy(self) -> None:
        """TOT strategy generates a 3-step plan."""
        state = initial_state("compare multiple approaches", "thread-tot")
        state["strategy"] = Strategy.TOT
        result = await plan_node(state)

        assert result["phase"] == Phase.RETRIEVE_MEMORY
        assert len(result["plan_steps"]) == 3

    @pytest.mark.asyncio
    async def test_plan_debate_strategy(self) -> None:
        """DEBATE strategy generates a 3-step plan."""
        state = initial_state("argue pros and cons", "thread-debate")
        state["strategy"] = Strategy.DEBATE
        result = await plan_node(state)

        assert result["phase"] == Phase.RETRIEVE_MEMORY
        assert len(result["plan_steps"]) == 3

    @pytest.mark.asyncio
    async def test_plan_rewoo_strategy_falls_back(self) -> None:
        """REWOO strategy has no dedicated branch, falls back to single-step plan."""
        state = initial_state("some goal", "thread-rewoo")
        state["strategy"] = Strategy.REWOO
        result = await plan_node(state)

        assert result["phase"] == Phase.RETRIEVE_MEMORY
        assert len(result["plan_steps"]) == 1

    @pytest.mark.asyncio
    async def test_plan_no_goal_returns_error(self) -> None:
        """Missing goal routes to ERROR_HANDLER phase."""
        state = initial_state("some goal", "thread-nogoal")
        state["current_goal"] = None
        result = await plan_node(state)

        assert result["phase"] == Phase.ERROR_HANDLER
        assert len(result["errors"]) > 0

    @pytest.mark.asyncio
    async def test_plan_empty_goal_text_returns_error(self) -> None:
        """Empty goal text routes to ERROR_HANDLER phase."""
        state = initial_state("some goal", "thread-empty")
        state["current_goal"] = Goal(text="")
        result = await plan_node(state)

        assert result["phase"] == Phase.ERROR_HANDLER
        assert len(result["errors"]) > 0

    @pytest.mark.asyncio
    async def test_plan_with_gateway_falls_back_to_heuristic(self, mock_gateway: object) -> None:
        """When gateway returns unparseable JSON, falls back to heuristics."""
        state = initial_state("implement feature", "thread-gw")
        state["strategy"] = Strategy.REACT
        result = await plan_node(state, gateway=mock_gateway)

        # Mock gateway returns classify JSON, not plan JSON → falls back to heuristic
        assert result["phase"] == Phase.RETRIEVE_MEMORY
        assert len(result["plan_steps"]) == 3

    @pytest.mark.asyncio
    async def test_plan_caps_heuristic_plan_to_remaining_iterations(self) -> None:
        """Plan is capped to the remaining iteration budget (PLANNING=4 steps, budget=3)."""
        state = initial_state("build end-to-end pipeline", "thread-cap", max_iterations=3)
        state["strategy"] = Strategy.PLANNING
        result = await plan_node(state)

        # remaining = max(0, 3 - 0) = 3 → cap = min(planning_max_steps=10, 3) = 3
        assert result["phase"] == Phase.RETRIEVE_MEMORY
        assert len(result["plan_steps"]) == 3

    @pytest.mark.asyncio
    async def test_plan_passes_remaining_iterations_to_prompt(self, mock_gateway: object) -> None:
        """The LLM plan prompt receives the remaining/max iteration budget."""
        state = initial_state("implement feature", "thread-budget", max_iterations=10)
        state["strategy"] = Strategy.REACT
        await plan_node(state, gateway=mock_gateway)

        # The mock gateway returns classify JSON → acompletion is still called once
        # before the parse fallback; inspect the user message it received.
        messages = mock_gateway.acompletion.call_args.kwargs["messages"]
        user_content = messages[1]["content"]
        assert "Remaining iterations" in user_content
        assert "10 / 10" in user_content  # 10 remaining of 10 at iteration 0


class TestPlanComplexityThreading:
    """The classified goal complexity routes the planning LLM call (§5 C.1)."""

    @pytest.mark.asyncio
    async def test_critical_complexity_threaded_to_gateway(
        self, mock_gateway: object
    ) -> None:
        """A CRITICAL goal routes planning to a stronger model, not SIMPLE."""
        state = initial_state("critical mission-critical task", "thread-cpx")
        state["strategy"] = Strategy.REACT
        state["current_goal"] = Goal(
            text="critical mission-critical task",
            status=GoalStatus.ACTIVE,
            complexity=TaskComplexity.CRITICAL,
        )

        await plan_node(state, gateway=mock_gateway)

        assert (
            mock_gateway.acompletion.call_args_list[0].kwargs["complexity"]
            == TaskComplexity.CRITICAL
        )

    @pytest.mark.asyncio
    async def test_unclassified_goal_defaults_to_simple(
        self, mock_gateway: object
    ) -> None:
        """A goal without a classified complexity falls back to SIMPLE."""
        state = initial_state("plain task", "thread-nocpx")
        state["strategy"] = Strategy.REACT
        # initial_state builds a Goal with default complexity=SIMPLE.
        assert state["current_goal"].complexity == TaskComplexity.SIMPLE

        await plan_node(state, gateway=mock_gateway)

        assert (
            mock_gateway.acompletion.call_args_list[0].kwargs["complexity"]
            == TaskComplexity.SIMPLE
        )


class TestPlanDeliverableAwareReplan:
    """battery-04 q08 regression: a re-plan after failing correctness checks
    must tell the planner which deliverables already PASS (reuse, do not
    regenerate) and which FAILED (fix only those) — instead of regenerating the
    whole pipeline, which perpetually invalidates the manifest's content-hash.

    Root cause: ``_llm_plan`` was eval-blind + deliverable-blind. It rebuilt the
    plan from goal/strategy/memory alone, so each re-plan rewrote
    ``raw_findings.jsonl`` and the manifest's ``input_sha256`` stayed stale
    forever. The ``_correction_context`` injects a targeted directive only when
    ``state["eval_checks"]`` carries failures (fresh plans are untouched).
    """

    @staticmethod
    def _user_prompt(mock_gateway: object) -> str:
        return mock_gateway.acompletion.call_args.kwargs["messages"][1]["content"]

    @pytest.mark.asyncio
    async def test_fresh_plan_has_no_correction_directive(
        self, mock_gateway: object
    ) -> None:
        """A fresh plan (no eval_checks in state) omits the correction block."""
        state = initial_state("analyze data", "thread-fresh")
        state["strategy"] = Strategy.REACT
        await plan_node(state, gateway=mock_gateway)

        prompt = self._user_prompt(mock_gateway)
        assert "CORRECTION RE-PLAN" not in prompt

    @pytest.mark.asyncio
    async def test_all_passing_plan_has_no_correction_directive(
        self, mock_gateway: object
    ) -> None:
        """All-passing checks = nothing to correct → no directive."""
        state = initial_state("analyze data", "thread-allpass")
        state["strategy"] = Strategy.REACT
        state["eval_checks"] = [
            {"check_name": "q08_raw_jsonl_rows", "passed": True, "skipped": False,
             "score": 1.0, "evidence": {}, "error": None},
        ]
        await plan_node(state, gateway=mock_gateway)

        prompt = self._user_prompt(mock_gateway)
        assert "CORRECTION RE-PLAN" not in prompt

    @pytest.mark.asyncio
    async def test_failing_check_surfaces_correction_with_pass_and_fail(
        self, mock_gateway: object
    ) -> None:
        """A failing check (with one passing) lists both and forbids regeneration."""
        state = initial_state("multi-agent pipeline", "thread-replan")
        state["strategy"] = Strategy.REACT
        state["eval_checks"] = [
            {"check_name": "q08_raw_jsonl_rows", "passed": True, "skipped": False,
             "score": 1.0, "evidence": {}, "error": None},
            {"check_name": "q08_handoff_integrity", "passed": False, "skipped": False,
             "score": 0.0,
             "evidence": {"stdout": "stage analysis: input_sha256 mismatch "
                          "(did not read results/q08/raw_findings.jsonl)"},
             "error": None},
        ]
        await plan_node(state, gateway=mock_gateway)

        prompt = self._user_prompt(mock_gateway)
        assert "CORRECTION RE-PLAN" in prompt
        # Passing deliverable named under PASSED (reuse).
        assert "q08_raw_jsonl_rows" in prompt
        assert "PASSED" in prompt
        assert "DO NOT" in prompt
        # Failing check named + its actionable reason surfaced.
        assert "q08_handoff_integrity" in prompt
        assert "FAILED" in prompt
        assert "input_sha256 mismatch" in prompt

    @pytest.mark.asyncio
    async def test_integrity_failure_emits_recompute_directive(
        self, mock_gateway: object
    ) -> None:
        """A hash/mismatch failure tells the agent to re-read+recompute, not
        regenerate upstream — the exact q08 plateau cause."""
        state = initial_state("multi-agent pipeline", "thread-hash")
        state["strategy"] = Strategy.REACT
        state["eval_checks"] = [
            {"check_name": "q08_handoff_integrity", "passed": False, "skipped": False,
             "score": 0.0,
             "evidence": {"stdout": "stage analysis: input_sha256 mismatch"},
             "error": None},
        ]
        await plan_node(state, gateway=mock_gateway)

        prompt = self._user_prompt(mock_gateway)
        assert "CRITICAL" in prompt
        assert "recompute" in prompt.lower()
        assert "Regenerating the upstream" in prompt

    @pytest.mark.asyncio
    async def test_non_hash_failure_omits_recompute_directive(
        self, mock_gateway: object
    ) -> None:
        """A non-integrity failure (e.g. structural) does not claim the upstream
        is correct — no hash-recompute CRITICAL line."""
        state = initial_state("multi-agent pipeline", "thread-nonhash")
        state["strategy"] = Strategy.REACT
        state["eval_checks"] = [
            {"check_name": "q08_final_report", "passed": False, "skipped": False,
             "score": 0.0,
             "evidence": {"reason": "final_report.md missing required section"},
             "error": None},
        ]
        await plan_node(state, gateway=mock_gateway)

        prompt = self._user_prompt(mock_gateway)
        assert "CORRECTION RE-PLAN" in prompt
        assert "q08_final_report" in prompt
        # No integrity marker in the reason → no recompute CRITICAL line.
        assert "Regenerating the upstream" not in prompt

    @pytest.mark.asyncio
    async def test_skipped_checks_treated_as_neither_pass_nor_fail(
        self, mock_gateway: object
    ) -> None:
        """A skipped check is neither a pass to reuse nor a failure to fix."""
        state = initial_state("multi-agent pipeline", "thread-skip")
        state["strategy"] = Strategy.REACT
        state["eval_checks"] = [
            {"check_name": "q08_oracle_judge", "passed": False, "skipped": True,
             "score": 0.0, "evidence": {}, "error": None},
        ]
        await plan_node(state, gateway=mock_gateway)

        prompt = self._user_prompt(mock_gateway)
        # Skipped-only → no failed checks → no correction directive at all.
        assert "CORRECTION RE-PLAN" not in prompt


class TestTechniqueInjection:
    """§5: a CRITICAL goal injects ≥1 technique body above the JSON footer.

    Confirms the dynamic prompting layer fires end-to-end: techniques are
    selected by (complexity, node, goal-pattern), spliced into the system
    prompt above the schema footer, and the resulting messages still parse as
    a valid GeneratedPlan (StructuredOutputManager.extract succeeds).
    """

    @pytest.mark.asyncio
    async def test_critical_math_goal_injects_chain_of_thought(self) -> None:
        state = initial_state(
            "calculate and prove the convergence of the series", "thread-tech"
        )
        state["strategy"] = Strategy.REACT
        state["current_goal"] = Goal(
            text="calculate and prove the convergence of the series",
            status=GoalStatus.ACTIVE,
            complexity=TaskComplexity.CRITICAL,
        )

        plan_json = (
            '{"steps": [{"description": "solve it", "tool_name": null, '
            '"expected_output": "answer"}], "rationale": "reasons"}'
        )
        gateway = MagicMock()
        gateway.acompletion = AsyncMock(return_value=LLMResponse(
            content=plan_json,
            model="gpt-4o-mini-2024-07-18",
            provider="openai",
            input_tokens=10,
            output_tokens=50,
            total_tokens=60,
            cost_usd=0.0001,
        ))

        result = await plan_node(state, gateway=gateway)

        # extract() succeeded on the spliced prompt → real plan steps returned.
        assert result["phase"] == Phase.RETRIEVE_MEMORY
        assert len(result["plan_steps"]) >= 1

        # The technique block was injected, above the intact JSON footer.
        system = gateway.acompletion.call_args.kwargs["messages"][0]["content"]
        assert "Reasoning techniques to apply:" in system
        assert "Respond with a JSON object matching this schema:" in system
        # chain_of_thought is the top CRITICAL+math technique → its body present.
        assert "step by step" in system.lower()
        # The block must precede the schema footer (so extract() finds it last).
        assert (
            system.find("Reasoning techniques to apply:")
            < system.find("Respond with a JSON object")
        )
