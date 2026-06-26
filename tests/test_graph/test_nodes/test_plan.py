"""Tests for src.graph.nodes.plan — plan node function."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config import get_settings
from src.graph.enums import GoalStatus, Phase, Strategy, TaskComplexity
from src.graph.factory import initial_state
from src.graph.models import Goal, PlanStep
from src.graph.nodes.plan import _split_coarse_step, _validate_step_atomicity, plan_node
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


class TestPlanMissingDeliverableReplan:
    """complex-arxiv-stats-3 regression: a re-plan after verify found a
    missing/empty goal deliverable must target ONLY that deliverable — write
    it minimally via file_writer — instead of regenerating the whole pipeline.

    Root cause: the planner was missing-deliverable-blind. Each re-plan rebuilt
    the full pipeline (re-deriving papers.jsonl/stats.json), so it never
    reached the missing-deliverable step before the memory-folding checkpoint
    interrupted it mid-plan — observed looping 40 iterations never writing
    attention_report.md. ``_missing_deliverable_context`` injects a targeted
    directive only when ``state["missing_deliverables"]`` is non-empty (fresh
    plans are untouched). Mirrors ``TestPlanDeliverableAwareReplan`` but for
    the VERIFY-driven (not eval-driven) missing case.
    """

    @staticmethod
    def _user_prompt(mock_gateway: object) -> str:
        return mock_gateway.acompletion.call_args.kwargs["messages"][1]["content"]

    @pytest.mark.asyncio
    async def test_fresh_plan_has_no_missing_directive(
        self, mock_gateway: object
    ) -> None:
        """A fresh plan (no missing_deliverables in state) omits the directive."""
        state = initial_state("summarize arxiv papers", "thread-md-fresh")
        state["strategy"] = Strategy.REACT
        await plan_node(state, gateway=mock_gateway)

        prompt = self._user_prompt(mock_gateway)
        assert "MISSING-DELIVERABLE RE-PLAN" not in prompt

    @pytest.mark.asyncio
    async def test_empty_missing_list_omits_directive(
        self, mock_gateway: object
    ) -> None:
        """An empty missing list (all deliverables now present) = no directive."""
        state = initial_state("summarize arxiv papers", "thread-md-empty")
        state["strategy"] = Strategy.REACT
        state["missing_deliverables"] = []
        await plan_node(state, gateway=mock_gateway)

        prompt = self._user_prompt(mock_gateway)
        assert "MISSING-DELIVERABLE RE-PLAN" not in prompt

    @pytest.mark.asyncio
    async def test_missing_deliverable_surfaces_targeted_directive(
        self, mock_gateway: object
    ) -> None:
        """A missing deliverable names the file and demands a minimal file_writer
        plan — the exact complex-arxiv-stats-3 fix."""
        state = initial_state("summarize arxiv papers", "thread-md-targeted")
        state["strategy"] = Strategy.REACT
        state["missing_deliverables"] = ["results/attention_report.md"]
        await plan_node(state, gateway=mock_gateway)

        prompt = self._user_prompt(mock_gateway)
        assert "MISSING-DELIVERABLE RE-PLAN" in prompt
        # The missing file is named so the plan targets it, not the pipeline.
        assert "results/attention_report.md" in prompt
        # The plan must be minimal (finish before the fold checkpoint interrupts).
        assert "MINIMAL" in prompt
        # The producing step must persist via file_writer (not narrate in text).
        assert "file_writer" in prompt
        assert "Do NOT regenerate" in prompt

    @pytest.mark.asyncio
    async def test_missing_deliverable_list_capped_at_eight(
        self, mock_gateway: object
    ) -> None:
        """More than 8 missing deliverables lists only the first 8 (the [:8]
        cap) so the directive stays bounded."""
        state = initial_state("many deliverables", "thread-md-cap")
        state["strategy"] = Strategy.REACT
        state["missing_deliverables"] = [
            f"results/file_{i}.md" for i in range(1, 11)
        ]
        await plan_node(state, gateway=mock_gateway)

        prompt = self._user_prompt(mock_gateway)
        assert "results/file_8.md" in prompt
        # The 9th/10th are beyond the cap → not enumerated in the directive.
        assert "results/file_9.md" not in prompt
        assert "results/file_10.md" not in prompt


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


class TestGeneratedStepFieldPreservation:
    """Regression: GeneratedStep→PlanStep conversion must preserve
    tool_name, expected_output, AND depends_on.

    Previously plan_node only carried `description` into PlanStep, silently
    discarding the LLM-emitted tool_name/expected_output (and depends_on was
    never populated even though PlanStep carried the field). These are exactly
    the plumbing the atomicity / decomposition passes need, so they must round-trip.
    """

    @pytest.mark.asyncio
    async def test_tool_name_expected_output_and_depends_on_preserved(self) -> None:
        state = initial_state(
            "gather data then compute the answer", "thread-fields"
        )
        state["strategy"] = Strategy.PLANNING
        state["current_goal"] = Goal(
            text="gather data then compute the answer",
            status=GoalStatus.ACTIVE,
            complexity=TaskComplexity.COMPLEX,
        )

        plan_json = (
            '{"steps": ['
            '{"description": "Gather data", "tool_name": "web_search", '
            '"expected_output": "search results", "depends_on": []}, '
            '{"description": "Compute the answer", "tool_name": "code_executor", '
            '"expected_output": "computed answer", '
            '"depends_on": ["Gather data"]}'
            '], "rationale": "dependency-ordered"}'
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

        assert result["phase"] == Phase.RETRIEVE_MEMORY
        steps = result["plan_steps"]
        assert len(steps) == 2

        # First step: tool_name + expected_output preserved, no dependencies.
        assert steps[0].tool_name == "web_search"
        assert steps[0].expected_output == "search results"
        assert steps[0].depends_on == []

        # Second step: every field preserved, including the dependency edge.
        assert steps[1].tool_name == "code_executor"
        assert steps[1].expected_output == "computed answer"
        assert steps[1].depends_on == ["Gather data"]


def _mock_gateway_for_plan(plan_json: str) -> object:
    """Build a gateway mock whose acompletion returns the given plan JSON."""
    gw = MagicMock()
    gw.acompletion = AsyncMock(return_value=LLMResponse(
        content=plan_json,
        model="gpt-4o-mini-2024-07-18",
        provider="openai",
        input_tokens=10,
        output_tokens=50,
        total_tokens=60,
        cost_usd=0.0001,
    ))
    return gw


class TestPlanAtomicityValidator:
    """Feature C — the pure-heuristic per-step atomicity validator."""

    def test_too_coarse_flag(self) -> None:
        """A step with >=2 conjunction/clause markers is too_coarse."""
        steps = [PlanStep(description="Fetch the data and clean it then write to disk")]
        quality = _validate_step_atomicity(steps)
        assert quality.per_step[0].flag == "too_coarse"
        assert quality.too_coarse_count == 1
        assert quality.atomic is False

    def test_too_fine_flag(self) -> None:
        """A sub-3-word step with no expected_output is too_fine."""
        steps = [PlanStep(description="run it")]  # 2 words, no expected_output
        quality = _validate_step_atomicity(steps)
        assert quality.per_step[0].flag == "too_fine"
        assert quality.too_fine_count == 1
        assert quality.atomic is False

    def test_too_fine_rescued_by_expected_output(self) -> None:
        """A short step WITH an expected_output is atomic (not too_fine)."""
        steps = [PlanStep(description="run it", expected_output="exit code 0")]
        quality = _validate_step_atomicity(steps)
        assert quality.per_step[0].flag == "atomic"

    def test_atomic_flag(self) -> None:
        """A single, well-scoped step is atomic."""
        steps = [PlanStep(description="Compute the Fibonacci sequence")]
        quality = _validate_step_atomicity(steps)
        assert quality.per_step[0].flag == "atomic"
        assert quality.atomic is True
        assert quality.too_coarse_count == 0
        assert quality.too_fine_count == 0

    def test_split_coarse_step_decomposes(self) -> None:
        """_split_coarse_step breaks a coarse step into one PlanStep per clause."""
        step = PlanStep(
            description="Fetch the data and clean it then write to disk",
            tool_name="code_executor",
            expected_output="file",
        )
        sub = _split_coarse_step(step)
        assert len(sub) == 3
        assert [s.description for s in sub] == [
            "Fetch the data", "clean it", "write to disk",
        ]
        # Context inherited.
        for s in sub:
            assert s.tool_name == "code_executor"
            assert s.expected_output == "file"

    def test_split_returns_unchanged_when_single_clause(self) -> None:
        """A non-coarse description (one clause) is returned unchanged."""
        step = PlanStep(description="Compute the Fibonacci sequence")
        assert _split_coarse_step(step) == [step]


class TestPlanAtomicityNodeIntegration:
    """Feature C — plan_node attaches plan_quality + enforce-gated split."""

    _COARSE_PLAN_JSON = (
        '{"steps": ['
        '{"description": "Fetch the data and clean it then write to disk", '
        '"tool_name": "code_executor", "expected_output": "file", '
        '"depends_on": []}'
        '], "rationale": "one coarse step"}'
    )

    def _state_with_coarse_goal(self, thread: str) -> dict:
        state = initial_state(
            "fetch the data and clean it then write to disk", thread,
        )
        state["strategy"] = Strategy.PLANNING
        state["current_goal"] = Goal(
            text="fetch the data and clean it then write to disk",
            status=GoalStatus.ACTIVE,
            complexity=TaskComplexity.COMPLEX,
        )
        return state

    @pytest.mark.asyncio
    async def test_plan_quality_always_attached(self) -> None:
        """Even with enforce off, plan_quality is attached as advisory telemetry."""
        state = self._state_with_coarse_goal("thread-quality")
        result = await plan_node(state, gateway=_mock_gateway_for_plan(self._COARSE_PLAN_JSON))
        # The coarse step is preserved (enforce is off by default).
        assert len(result["plan_steps"]) == 1
        pq = result["plan_quality"]
        assert pq["too_coarse_count"] == 1
        assert pq["atomic"] is False
        assert pq["per_step"][0]["flag"] == "too_coarse"
        # No enforcement attempted -> no replan marker.
        assert "atomicity_replan_done" not in result

    @pytest.mark.asyncio
    async def test_enforce_off_preserves_coarse_step(self) -> None:
        """Default-off: the coarse step is NOT split (plan unchanged)."""
        state = self._state_with_coarse_goal("thread-nosplit")
        result = await plan_node(state, gateway=_mock_gateway_for_plan(self._COARSE_PLAN_JSON))
        assert len(result["plan_steps"]) == 1
        assert "fetch the data" in result["plan_steps"][0].description.lower()

    @pytest.mark.asyncio
    async def test_enforce_on_splits_coarse_step(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """enforce on + coarse step -> one bounded heuristic split."""
        monkeypatch.setattr(get_settings().agent, "plan_atomicity_enforce", True)
        state = self._state_with_coarse_goal("thread-split")
        result = await plan_node(state, gateway=_mock_gateway_for_plan(self._COARSE_PLAN_JSON))
        # The single coarse step decomposed into 3 atomic clauses.
        assert len(result["plan_steps"]) == 3
        assert result["atomicity_replan_done"] is True
        # Re-validated quality now reports the plan as atomic.
        assert result["plan_quality"]["atomic"] is True
        assert result["plan_quality"]["too_coarse_count"] == 0

    @pytest.mark.asyncio
    async def test_replan_done_guard_prevents_resplit(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """atomicity_replan_done set in state -> no re-split even with enforce on."""
        monkeypatch.setattr(get_settings().agent, "plan_atomicity_enforce", True)
        state = self._state_with_coarse_goal("thread-guard")
        state["atomicity_replan_done"] = True
        result = await plan_node(state, gateway=_mock_gateway_for_plan(self._COARSE_PLAN_JSON))
        # Guard held: the coarse step is NOT decomposed.
        assert len(result["plan_steps"]) == 1
        assert result["plan_quality"]["too_coarse_count"] == 1
