"""Tests for src.graph.routers — conditional edge routing functions."""

from __future__ import annotations

from typing import Any

import pytest


from src.config.settings import get_settings
from src.graph.enums import Confidence, TaskComplexity
from src.graph.iteration_cap import effective_max_iterations
from src.graph.models import PlanStep, ReflectionResult, ToolResult
from src.graph.routers import (
    route_after_error,
    route_after_evolve,
    route_after_execute,
    route_after_hitl,
    route_after_reflect,
    route_after_store,
    route_after_structure_analysis,
    route_after_verify,
)


class TestRouteAfterExecute:
    """Tests for route_after_execute routing function."""

    def test_route_after_execute_to_reflect(self, state_with_plan: dict[str, Any]) -> None:
        """When all plan steps are executed, route to reflect."""
        # Exhaust all steps
        state_with_plan["current_step_index"] = 3  # equal to len(plan_steps)
        result = route_after_execute(state_with_plan)
        assert result == "reflect"

    def test_route_after_execute_to_reflect_on_max_iterations(self, sample_state: dict[str, Any]) -> None:
        """When max iterations is reached, route to reflect."""
        sample_state["iteration_count"] = 25
        sample_state["max_iterations"] = 25
        result = route_after_execute(sample_state)
        assert result == "reflect"

    def test_route_after_execute_is_complexity_aware(self) -> None:
        """B1: when state omits an explicit cap, routing terminates at the goal's
        complexity tier (via ``effective_max_iterations``), not a flat cap.

        A TRIVIAL goal reflects once iteration_count reaches its low tier cap; a
        COMPLEX goal at the same count keeps executing (its tier cap is higher).
        The cap is derived from settings, not a hardcoded literal.
        """
        from types import SimpleNamespace

        def _state(complexity: TaskComplexity, iters: int) -> dict[str, Any]:
            # No max_iterations key — exercises the complexity-tier fallback.
            return {
                "iteration_count": iters,
                "errors": [],
                "tool_results": [],
                "plan_steps": [],
                "current_step_index": 0,
                "messages": [],
                "current_goal": SimpleNamespace(complexity=complexity),
            }

        trivial_cap = effective_max_iterations(_state(TaskComplexity.TRIVIAL, 0))
        complex_cap = effective_max_iterations(_state(TaskComplexity.COMPLEX, 0))
        # The tiers genuinely differ — a TRIVIAL goal caps lower than a COMPLEX one.
        assert trivial_cap < complex_cap
        # At the TRIVIAL cap → reflect; a COMPLEX goal at the same count → execute.
        assert route_after_execute(_state(TaskComplexity.TRIVIAL, trivial_cap)) == "reflect"
        assert route_after_execute(_state(TaskComplexity.COMPLEX, trivial_cap)) == "execute"

    def test_route_after_execute_to_error(self, sample_state: dict[str, Any]) -> None:
        """When authentication errors are present, route to error_handler."""
        sample_state["errors"] = ["authentication failed for provider"]
        result = route_after_execute(sample_state)
        assert result == "error_handler"

    def test_route_after_execute_retries_on_persistent_tool_error(self, sample_state: dict[str, Any]) -> None:
        """A non-timeout tool failure (unknown tool, permission denied, …) is
        recoverable: retry execute so the LLM sees the failure in context and
        picks a valid tool. Previously this routed to error_handler, which
        falsely completed the run because tool failures live in tool_results,
        not errors (F14)."""
        sample_state["tool_results"] = [
            ToolResult(tool_name="code_executor", success=False, output="", error="permission denied"),
        ]
        result = route_after_execute(sample_state)
        assert result == "execute"

    def test_route_after_execute_at_cap_with_tool_failure_routes_to_reflect(
        self, sample_state: dict[str, Any]
    ) -> None:
        """A persistently-failing tool retries only until the iteration cap,
        then reflects — never loops execute→execute into LangGraph's recursion
        limit. The max-iterations guard must run before the recoverable-tool
        retry (F14)."""
        sample_state["tool_results"] = [
            ToolResult(tool_name="find", success=False, output="", error="Unknown tool: find"),
        ]
        sample_state["iteration_count"] = 10
        sample_state["max_iterations"] = 10
        assert route_after_execute(sample_state) == "reflect"

    def test_route_after_execute_loops_on_retriable_tool_error(self, sample_state: dict[str, Any]) -> None:
        """When a retriable tool error occurs (timeout/rate), loop back to execute."""
        sample_state["tool_results"] = [
            ToolResult(tool_name="web_search", success=False, output="", error="timeout exceeded"),
        ]
        result = route_after_execute(sample_state)
        assert result == "execute"

    def test_route_after_execute_continues_with_remaining_steps(self, state_with_plan: dict[str, Any]) -> None:
        """When there are remaining steps, route back to execute."""
        state_with_plan["current_step_index"] = 0  # 3 steps total, only at index 0
        result = route_after_execute(state_with_plan)
        assert result == "execute"

    def test_route_after_execute_forces_reflect_on_message_floor_midplan(self, sample_state: dict[str, Any]) -> None:
        """Mid-plan with >=10 messages and past the fold window → reflect (fold checkpoint)."""
        sample_state["plan_steps"] = [
            PlanStep(id="s1", description="Step 1", status="pending"),
            PlanStep(id="s2", description="Step 2", status="pending"),
        ]
        sample_state["current_step_index"] = 0
        sample_state["iteration_count"] = 12
        sample_state["max_iterations"] = 25
        sample_state["messages"] = list(range(12))  # 12 messages >= floor
        sample_state["last_fold_iteration"] = 0
        assert route_after_execute(sample_state) == "reflect"

    def test_route_after_execute_no_force_reflect_within_cooldown(self, sample_state: dict[str, Any]) -> None:
        """Mid-plan within the fold cooldown → execute (checkpoint suppressed)."""
        sample_state["plan_steps"] = [
            PlanStep(id="s1", description="Step 1", status="pending"),
            PlanStep(id="s2", description="Step 2", status="pending"),
        ]
        sample_state["current_step_index"] = 0
        sample_state["iteration_count"] = 12
        sample_state["max_iterations"] = 25
        sample_state["messages"] = list(range(12))
        sample_state["last_fold_iteration"] = 8  # 12 - 8 = 4 < 6 → cooldown
        assert route_after_execute(sample_state) == "execute"

    def test_route_after_execute_no_force_reflect_when_few_messages_post_fold(self, sample_state: dict[str, Any]) -> None:
        """Mid-plan with few messages (post-fold reset) → execute (checkpoint inactive)."""
        sample_state["plan_steps"] = [
            PlanStep(id="s1", description="Step 1", status="pending"),
            PlanStep(id="s2", description="Step 2", status="pending"),
        ]
        sample_state["current_step_index"] = 0
        sample_state["iteration_count"] = 12
        sample_state["max_iterations"] = 25
        sample_state["messages"] = list(range(3))  # < floor after a fold reset
        sample_state["last_fold_iteration"] = 0
        assert route_after_execute(sample_state) == "execute"

    def test_route_after_execute_no_force_reflect_when_folds_exhausted(
        self, sample_state: dict[str, Any]
    ) -> None:
        """Past the fold window AND >=10 messages, but max_folds consumed → execute.

        Regression for the loop amplifier: once ``len(fold_history) >=
        memory_folding_max_folds`` the fold never executes, so
        ``last_fold_iteration`` stops advancing. Without the fold-availability
        gate, ``(iteration - last_fold) >= 6`` stayed permanently true and the
        checkpoint fired EVERY iteration, churning reflect→verify→replan and
        starving multi-deliverable plans of the uninterrupted execute steps
        they need to finish (observed on q4 under deepseek-v4-flash: 3 folds
        consumed by iter 19, then a checkpoint every iter 25→34).
        """
        sample_state["plan_steps"] = [
            PlanStep(id="s1", description="Step 1", status="pending"),
            PlanStep(id="s2", description="Step 2", status="pending"),
        ]
        sample_state["current_step_index"] = 0
        sample_state["iteration_count"] = 34
        sample_state["max_iterations"] = 60
        sample_state["messages"] = list(range(20))  # well past the floor
        sample_state["last_fold_iteration"] = 19  # 34 - 19 = 15 >= 6
        max_folds = get_settings().agent.memory_folding_max_folds
        sample_state["fold_history"] = [{"iteration": i} for i in range(max_folds)]
        assert route_after_execute(sample_state) == "execute"


class TestRouteAfterVerify:
    """Tests for route_after_verify routing function."""

    def test_route_after_verify_to_evolve(self, sample_state: dict[str, Any]) -> None:
        """When is_complete and should_evolve is True, route to evolve."""
        sample_state["is_complete"] = True
        sample_state["reflection"] = ReflectionResult(
            summary="Task complete",
            should_evolve=True,
        )
        result = route_after_verify(sample_state)
        assert result == "evolve"

    def test_route_after_verify_no_evolution_skips_evolve(
        self, sample_state: dict[str, Any]
    ) -> None:
        """--no-evolution in state short-circuits evolve even when reflection requests it.

        Invoked single-arg (no config) — exactly how LangGraph calls the
        router. The original Phase-4 bug shipped because tests passed a config
        dict that the production graph never forwards (config is None at
        conditional-edge routers). State is now the source of truth.
        """
        sample_state["is_complete"] = True
        sample_state["no_evolution"] = True
        sample_state["reflection"] = ReflectionResult(
            summary="Task complete",
            should_evolve=True,
        )
        result = route_after_verify(sample_state)
        assert result == "store_memory"

    def test_route_after_verify_evolution_enabled_routes_to_evolve(
        self, sample_state: dict[str, Any]
    ) -> None:
        """no_evolution absent (False default) allows evolve (the positive path)."""
        sample_state["is_complete"] = True
        sample_state["no_evolution"] = False
        sample_state["reflection"] = ReflectionResult(
            summary="Task complete",
            should_evolve=True,
        )
        result = route_after_verify(sample_state)
        assert result == "evolve"

    def test_route_after_verify_no_evolution_key_routes_to_evolve(
        self, sample_state: dict[str, Any]
    ) -> None:
        """With no no_evolution key in state, evolve still fires (False default)."""
        sample_state["is_complete"] = True
        sample_state.pop("no_evolution", None)
        sample_state["reflection"] = ReflectionResult(
            summary="Task complete",
            should_evolve=True,
        )
        result = route_after_verify(sample_state)
        assert result == "evolve"

    def test_route_after_verify_to_store(self, sample_state: dict[str, Any]) -> None:
        """When is_complete and no evolution needed, route to store_memory."""
        sample_state["is_complete"] = True
        sample_state["reflection"] = ReflectionResult(
            summary="Task complete",
            should_evolve=False,
        )
        result = route_after_verify(sample_state)
        assert result == "store_memory"

    def test_route_after_verify_to_store_when_no_reflection(self, sample_state: dict[str, Any]) -> None:
        """When is_complete with no reflection, route to store_memory (no evolve)."""
        sample_state["is_complete"] = True
        sample_state["reflection"] = None
        result = route_after_verify(sample_state)
        assert result == "store_memory"

    def test_route_after_verify_evolve_when_reflection_missing_but_succeeded(
        self, sample_state: dict[str, Any]
    ) -> None:
        """Folding-bypass regression: reflect's folding path early-returns before
        computing reflection (reflect.py:87-92), so a successful multi-step run can
        reach verify with reflection=None. route_after_verify must still ground
        should_evolve from objective evidence (>=3 completed steps, HIGH confidence,
        no errors) and route to evolve — otherwise evolution never fires on goals
        that fold mid-run (battery: 0 mutations across all 10 queries)."""
        sample_state["is_complete"] = True
        sample_state["reflection"] = None  # simulates the folding bypass
        sample_state["completed_steps"] = [
            PlanStep(id=f"s{i}", description=f"step {i}", status="completed", result="ok")
            for i in range(5)
        ]
        sample_state["errors"] = []
        sample_state["confidence"] = Confidence.HIGH
        result = route_after_verify(sample_state)
        assert result == "evolve"

    def test_route_after_verify_retries_on_low_confidence(self, sample_state: dict[str, Any]) -> None:
        """When not complete with low confidence and steps remaining, route back to execute."""
        sample_state["is_complete"] = False
        sample_state["confidence"] = Confidence.LOW
        sample_state["plan_steps"] = [
            PlanStep(id="s1", description="pending step", status="pending"),
            PlanStep(id="s2", description="another step", status="pending"),
        ]
        sample_state["current_step_index"] = 0
        result = route_after_verify(sample_state)
        assert result == "execute"

    def test_route_after_verify_retries_on_medium_confidence(self, sample_state: dict[str, Any]) -> None:
        """When not complete with medium confidence and steps remaining, still retries execute."""
        sample_state["is_complete"] = False
        sample_state["confidence"] = Confidence.MEDIUM
        sample_state["plan_steps"] = [
            PlanStep(id="s1", description="pending step", status="pending"),
            PlanStep(id="s2", description="another step", status="pending"),
        ]
        sample_state["current_step_index"] = 0
        result = route_after_verify(sample_state)
        assert result == "execute"

    def test_route_after_verify_to_store_when_budget_exhausted(self, sample_state: dict[str, Any]) -> None:
        """When not complete but budget exhausted, accept partial → store_memory (loop guard)."""
        sample_state["is_complete"] = False
        sample_state["confidence"] = Confidence.MEDIUM
        sample_state["iteration_count"] = 10
        sample_state["max_iterations"] = 10
        result = route_after_verify(sample_state)
        assert result == "store_memory"

    def test_route_after_verify_to_plan_when_no_remaining_steps(self, sample_state: dict[str, Any]) -> None:
        """When partial with no remaining steps and budget remains, re-plan to address gaps."""
        sample_state["is_complete"] = False
        sample_state["confidence"] = Confidence.MEDIUM
        sample_state["plan_steps"] = []
        sample_state["current_step_index"] = 0
        sample_state["iteration_count"] = 0
        sample_state["max_iterations"] = 10
        result = route_after_verify(sample_state)
        assert result == "plan"

    def test_route_after_verify_terminates_on_stable_fingerprint(self, sample_state: dict[str, Any]) -> None:
        """Convergence early-exit (B3): when verify emits an identical output
        fingerprint across ``convergence_stable_threshold`` consecutive passes AND
        the plan is exhausted (well under the hard cap), accept the partial
        result via ``store_memory`` instead of looping verify→plan→execute to the
        cap. This must fire BEFORE the budget-exhausted branch — iteration_count
        is kept below the cap to prove the terminator is the stable fingerprint,
        not the iteration budget."""
        threshold = get_settings().agent.convergence_stable_threshold
        sample_state["is_complete"] = False
        sample_state["confidence"] = Confidence.MEDIUM
        sample_state["consecutive_stable_verifies"] = threshold
        sample_state["last_verify_fingerprint"] = "abc123"
        # Plan exhausted: all steps consumed (step_index == len(plan_steps)).
        sample_state["plan_steps"] = [
            PlanStep(id="s1", description="step 1", status="completed", result="ok"),
            PlanStep(id="s2", description="step 2", status="completed", result="ok"),
        ]
        sample_state["current_step_index"] = 2
        # Well below the cap so the budget branch (also store_memory) cannot mask
        # the convergence terminator.
        sample_state["iteration_count"] = 0
        sample_state["max_iterations"] = 10
        result = route_after_verify(sample_state)
        assert result == "store_memory"

    def test_route_after_verify_stable_fingerprint_continues_with_steps_remaining(
        self, sample_state: dict[str, Any]
    ) -> None:
        """B3 guard: a stable fingerprint must NOT terminate when steps remain —
        the run may still make forward progress, so it continues (retries
        execute). The convergence branch only fires with the plan exhausted."""
        threshold = get_settings().agent.convergence_stable_threshold
        sample_state["is_complete"] = False
        sample_state["confidence"] = Confidence.HIGH
        sample_state["consecutive_stable_verifies"] = threshold
        sample_state["last_verify_fingerprint"] = "abc123"
        # Steps remain: step_index < len(plan_steps).
        sample_state["plan_steps"] = [
            PlanStep(id="s1", description="step 1", status="completed", result="ok"),
            PlanStep(id="s2", description="step 2", status="pending"),
        ]
        sample_state["current_step_index"] = 1
        sample_state["iteration_count"] = 0
        sample_state["max_iterations"] = 10
        result = route_after_verify(sample_state)
        assert result != "store_memory"
        assert result == "execute"

    def test_route_after_verify_below_threshold_falls_through_to_plan(
        self, sample_state: dict[str, Any]
    ) -> None:
        """B3 threshold guard: below ``convergence_stable_threshold`` consecutive
        stable passes, no early termination — a single transient repeat is
        normal. With the plan exhausted, it falls through to the re-plan branch
        (giving the agent a chance to address gaps) rather than accepting the
        partial result."""
        threshold = get_settings().agent.convergence_stable_threshold
        sample_state["is_complete"] = False
        sample_state["confidence"] = Confidence.MEDIUM
        # Strictly below the threshold.
        sample_state["consecutive_stable_verifies"] = max(threshold - 1, 0)
        sample_state["last_verify_fingerprint"] = "abc123"
        sample_state["plan_steps"] = []
        sample_state["current_step_index"] = 0
        sample_state["iteration_count"] = 0
        sample_state["max_iterations"] = 10
        result = route_after_verify(sample_state)
        assert result == "plan"


class TestRouteAfterError:
    """Tests for route_after_error routing function."""

    def test_route_after_error_to_execute(self, sample_state: dict[str, Any]) -> None:
        """When error is retryable (generic), route back to execute."""
        sample_state["errors"] = ["something went wrong"]
        sample_state["iteration_count"] = 5
        sample_state["max_iterations"] = 25
        result = route_after_error(sample_state)
        assert result == "execute"

    def test_route_after_error_to_execute_on_rate_limit(self, sample_state: dict[str, Any]) -> None:
        """When rate limited and under max iterations, retry execute."""
        sample_state["errors"] = ["rate limit exceeded, try again"]
        sample_state["iteration_count"] = 5
        sample_state["max_iterations"] = 25
        result = route_after_error(sample_state)
        assert result == "execute"

    def test_route_after_error_to_classify_on_auth(self, sample_state: dict[str, Any]) -> None:
        """When auth error, route to classify (try different provider)."""
        sample_state["errors"] = ["401 unauthorized access"]
        result = route_after_error(sample_state)
        assert result == "classify"

    def test_route_after_error_to_hitl_on_budget(self, sample_state: dict[str, Any]) -> None:
        """When budget exhausted, escalate to human via hitl_gate."""
        sample_state["errors"] = ["budget limit exceeded"]
        result = route_after_error(sample_state)
        assert result == "hitl_gate"

    def test_route_after_error_to_complete_on_max_iterations(self, sample_state: dict[str, Any]) -> None:
        """When max iterations exceeded with errors, abort to complete."""
        sample_state["errors"] = ["persistent failure"]
        sample_state["iteration_count"] = 25
        sample_state["max_iterations"] = 25
        result = route_after_error(sample_state)
        assert result == "complete"

    def test_route_after_error_to_verify_when_no_errors(self, sample_state: dict[str, Any]) -> None:
        """No errors to handle is an anomaly, not a success — route to verify so
        the actual state is judged instead of falsely completing the run (F14)."""
        sample_state["errors"] = []
        result = route_after_error(sample_state)
        assert result == "verify"


class TestRouteAfterStore:
    """Tests for route_after_store routing function."""

    def test_route_after_store_to_complete(self, sample_state: dict[str, Any]) -> None:
        """When is_complete is True, route to complete."""
        sample_state["is_complete"] = True
        result = route_after_store(sample_state)
        assert result == "complete"

    def test_route_after_store_to_execute_when_incomplete(self, sample_state: dict[str, Any]) -> None:
        """When is_complete is False and budget remains, route back to execute."""
        sample_state["is_complete"] = False
        result = route_after_store(sample_state)
        assert result == "execute"

    def test_route_after_store_to_complete_when_budget_exhausted(self, sample_state: dict[str, Any]) -> None:
        """When not complete but budget exhausted, complete with partial result (loop guard)."""
        sample_state["is_complete"] = False
        sample_state["iteration_count"] = 10
        sample_state["max_iterations"] = 10
        result = route_after_store(sample_state)
        assert result == "complete"


class TestRouteAfterReflect:
    """Tests for route_after_reflect routing function."""

    def test_route_after_reflect_to_verify(self, sample_state: dict[str, Any]) -> None:
        """Medium confidence → route to verify."""
        sample_state["confidence"] = Confidence.MEDIUM
        sample_state["reflection"] = ReflectionResult(summary="ok", should_replan=False)
        result = route_after_reflect(sample_state)
        assert result == "verify"

    def test_route_after_reflect_to_plan(self, sample_state: dict[str, Any]) -> None:
        """should_replan=True on an in-progress-but-stuck plan (steps remain but
        none completed) → route to plan for a genuine replan."""
        sample_state["confidence"] = Confidence.HIGH
        sample_state["reflection"] = ReflectionResult(summary="replan", should_replan=True)
        sample_state["plan_steps"] = [
            PlanStep(id="s1", description="step 1", status="pending"),
            PlanStep(id="s2", description="step 2", status="pending"),
        ]
        sample_state["current_step_index"] = 0
        sample_state["completed_steps"] = []  # stuck: nothing completed
        sample_state["errors"] = []
        result = route_after_reflect(sample_state)
        assert result == "plan"

    def test_route_after_reflect_replan_in_progress_progressing_continues_execute(
        self, sample_state: dict[str, Any]
    ) -> None:
        """should_replan=True while a plan is in-progress AND progressing must
        NOT discard completed work — continue executing so the plan finishes and
        verify can judge it. (Q9: the folding checkpoint interrupted a
        succeeding plan mid-way every ~6 iterations; each replan regenerated a
        fresh plan that got interrupted the same way, looping at step 1/3.)"""
        sample_state["confidence"] = Confidence.HIGH
        sample_state["reflection"] = ReflectionResult(summary="replan", should_replan=True)
        sample_state["plan_steps"] = [
            PlanStep(id="s1", description="step 1", status="completed"),
            PlanStep(id="s2", description="step 2", status="pending"),
            PlanStep(id="s3", description="step 3", status="pending"),
        ]
        sample_state["current_step_index"] = 1  # 1 step done, 2 remain
        sample_state["completed_steps"] = [
            PlanStep(id="s1", description="step 1", status="completed", result="done")
        ]
        sample_state["errors"] = []
        result = route_after_reflect(sample_state)
        assert result == "execute"

    def test_route_after_reflect_replan_exhausted_routes_verify(
        self, sample_state: dict[str, Any]
    ) -> None:
        """should_replan=True with the plan exhausted routes to verify (the
        objective deliverable check) before regenerating — if the deliverable
        is present verify completes instead of looping plan->execute->plan."""
        sample_state["confidence"] = Confidence.HIGH
        sample_state["reflection"] = ReflectionResult(summary="replan", should_replan=True)
        sample_state["plan_steps"] = [
            PlanStep(id="s1", description="step 1", status="completed"),
            PlanStep(id="s2", description="step 2", status="completed"),
        ]
        sample_state["current_step_index"] = 2  # all steps done
        sample_state["completed_steps"] = [
            PlanStep(id="s1", description="step 1", status="completed", result="done"),
            PlanStep(id="s2", description="step 2", status="completed", result="done"),
        ]
        sample_state["errors"] = []
        result = route_after_reflect(sample_state)
        assert result == "verify"

    def test_route_after_reflect_low_confidence_to_execute(self, sample_state: dict[str, Any]) -> None:
        """LOW confidence with remaining steps → route back to execute."""
        sample_state["confidence"] = Confidence.LOW
        sample_state["reflection"] = ReflectionResult(summary="low", should_replan=False)
        sample_state["plan_steps"] = [
            PlanStep(id="s1", description="pending step", status="pending"),
        ]
        sample_state["current_step_index"] = 0
        result = route_after_reflect(sample_state)
        assert result == "execute"

    def test_route_after_reflect_no_reflection_to_verify(self, sample_state: dict[str, Any]) -> None:
        """No reflection → default to verify."""
        sample_state["reflection"] = None
        result = route_after_reflect(sample_state)
        assert result == "verify"


class TestRouteAfterReflectLoopGuards:
    """Regression tests for F12: reflect↔execute must not loop forever.

    T2 crashed with GraphRecursionError because a low-confidence reflection at
    the iteration cap routed back to execute; execute had no remaining steps and
    bounced to reflect, looping until LangGraph's recursion limit. The cap (and
    the no-remaining-steps pre-cap case) now route to verify instead.
    """

    def test_at_cap_routes_to_verify_even_when_low_confidence(
        self, sample_state: dict[str, Any]
    ) -> None:
        """At the iteration cap, low confidence must NOT loop back to execute."""
        sample_state["iteration_count"] = 10
        sample_state["max_iterations"] = 10
        sample_state["confidence"] = Confidence.LOW
        sample_state["reflection"] = ReflectionResult(summary="low at cap", should_replan=False)
        assert route_after_reflect(sample_state) == "verify"

    def test_at_cap_routes_to_verify_even_with_pending_gaps(
        self, sample_state: dict[str, Any]
    ) -> None:
        """At the cap, gap resolution is skipped — gaps would only re-loop."""
        sample_state["iteration_count"] = 10
        sample_state["max_iterations"] = 10
        sample_state["confidence"] = Confidence.LOW
        sample_state["pending_tool_gaps"] = ["some_missing_tool"]
        sample_state["reflection"] = ReflectionResult(summary="cap", should_replan=False)
        assert route_after_reflect(sample_state) == "verify"

    def test_low_confidence_no_remaining_steps_routes_to_verify(
        self, sample_state: dict[str, Any]
    ) -> None:
        """Pre-cap: low confidence with an exhausted plan must not retry execute."""
        sample_state["iteration_count"] = 3
        sample_state["max_iterations"] = 10
        sample_state["confidence"] = Confidence.LOW
        sample_state["plan_steps"] = []  # exhausted
        sample_state["current_step_index"] = 0
        sample_state["reflection"] = ReflectionResult(summary="stuck", should_replan=False)
        assert route_after_reflect(sample_state) == "verify"


class TestRouteAfterEvolve:
    """Tests for route_after_evolve routing function."""

    def test_route_after_evolve_to_store_memory(self, sample_state: dict[str, Any]) -> None:
        """No evolution errors → route to store_memory."""
        sample_state["errors"] = []
        result = route_after_evolve(sample_state)
        assert result == "store_memory"

    def test_route_after_evolve_to_error_handler(self, sample_state: dict[str, Any]) -> None:
        """Evolution error → route to error_handler."""
        sample_state["errors"] = ["evolution mutation failed"]
        result = route_after_evolve(sample_state)
        assert result == "error_handler"

    def test_route_after_evolve_to_execute_when_reexecute_offered(
        self, sample_state: dict[str, Any]
    ) -> None:
        """Phase 4 E: a live-registered TOOL mutation signals one execute pass."""
        sample_state["errors"] = []
        sample_state["evolve_reexecute_offered"] = True
        result = route_after_evolve(sample_state)
        assert result == "execute"

    def test_route_after_evolve_errors_precede_reexecute(
        self, sample_state: dict[str, Any]
    ) -> None:
        """Errors win even when a re-execution was offered (fail-closed)."""
        sample_state["errors"] = ["evolution mutation failed"]
        sample_state["evolve_reexecute_offered"] = True
        result = route_after_evolve(sample_state)
        assert result == "error_handler"

    def test_route_after_evolve_no_offer_defaults_to_store(
        self, sample_state: dict[str, Any]
    ) -> None:
        """No offer flag in state → store_memory (regression guard for E)."""
        sample_state["errors"] = []
        sample_state.pop("evolve_reexecute_offered", None)
        result = route_after_evolve(sample_state)
        assert result == "store_memory"


class TestRouteAfterHitl:
    """Tests for route_after_hitl routing function."""

    def test_route_after_hitl_approved_to_complete(self, sample_state: dict[str, Any]) -> None:
        """is_complete=True → route to complete."""
        sample_state["is_complete"] = True
        result = route_after_hitl(sample_state)
        assert result == "complete"

    def test_route_after_hitl_rejected_to_execute(self, sample_state: dict[str, Any]) -> None:
        """is_complete=False → route to execute for revision."""
        sample_state["is_complete"] = False
        result = route_after_hitl(sample_state)
        assert result == "execute"


class TestRouteAfterReflectAgentGaps:
    """Tests for agent gap routing in route_after_reflect."""

    def test_route_after_reflect_agent_gaps(self, sample_state: dict[str, Any]) -> None:
        """Routes to agent_spawn when pending_agent_gaps present."""
        sample_state["pending_agent_gaps"] = ["Need data analysis specialist"]
        result = route_after_reflect(sample_state)
        assert result == "agent_spawn"

    def test_route_after_reflect_agent_gaps_before_tool_gaps(self, sample_state: dict[str, Any]) -> None:
        """Agent gaps take priority over tool gaps."""
        sample_state["pending_agent_gaps"] = ["Need specialist"]
        sample_state["pending_tool_gaps"] = ["missing_tool"]
        result = route_after_reflect(sample_state)
        # Agent gaps have higher priority
        assert result == "agent_spawn"

    def test_route_after_reflect_tool_gaps_when_no_agent_gaps(self, sample_state: dict[str, Any]) -> None:
        """Routes to tool_create when only tool gaps present."""
        sample_state["pending_agent_gaps"] = []
        sample_state["pending_tool_gaps"] = ["missing_tool"]
        result = route_after_reflect(sample_state)
        assert result == "tool_create"


class TestCapabilityCapLoopBreak:
    """q09 run-control B: route_after_reflect must break the capability-cap
    spawn<->create ping-pong once consecutive cap-blocks reach the threshold,
    routing to ``verify`` (accept partial) instead of re-routing into the
    unfillable gaps forever (the q09 loop that had to be halted via a container
    restart). The counter is reset to 0 on real progress by the spawn/create
    nodes themselves; here we assert the router reads it correctly."""

    def test_breaks_to_verify_at_threshold(
        self, sample_state: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """At threshold (2 = one fully-saturated spawn+create cycle) with agent gaps present → verify, NOT agent_spawn."""
        monkeypatch.setattr(get_settings().agent, "cap_loop_break_threshold", 2)
        sample_state["pending_agent_gaps"] = ["Need specialist"]
        sample_state["consecutive_cap_blocks"] = 2
        assert route_after_reflect(sample_state) == "verify"

    def test_breaks_to_verify_above_threshold(
        self, sample_state: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Above threshold with tool gaps present → verify, NOT tool_create."""
        monkeypatch.setattr(get_settings().agent, "cap_loop_break_threshold", 2)
        sample_state["pending_tool_gaps"] = ["missing_tool"]
        sample_state["consecutive_cap_blocks"] = 5
        assert route_after_reflect(sample_state) == "verify"

    def test_does_not_break_below_threshold(
        self, sample_state: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Below threshold (1 < 2) still routes to gap resolution (agent_spawn)."""
        monkeypatch.setattr(get_settings().agent, "cap_loop_break_threshold", 2)
        sample_state["pending_agent_gaps"] = ["Need specialist"]
        sample_state["consecutive_cap_blocks"] = 1
        assert route_after_reflect(sample_state) == "agent_spawn"

    def test_default_counter_routes_to_gaps(self, sample_state: dict[str, Any]) -> None:
        """No counter key (default 0) → backward compatible: routes to tool_create."""
        sample_state["pending_tool_gaps"] = ["missing_tool"]
        # consecutive_cap_blocks intentionally absent (default 0 via state.get).
        assert route_after_reflect(sample_state) == "tool_create"

    def test_disabled_when_threshold_zero(
        self, sample_state: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """threshold=0 disables the break (opt-out) even at a high counter."""
        monkeypatch.setattr(get_settings().agent, "cap_loop_break_threshold", 0)
        sample_state["pending_agent_gaps"] = ["Need specialist"]
        sample_state["consecutive_cap_blocks"] = 99
        assert route_after_reflect(sample_state) == "agent_spawn"


class TestRouteAfterStructureAnalysis:
    """Tests for route_after_structure_analysis routing function."""

    def test_routes_to_agent_spawn(self, sample_state: dict[str, Any]) -> None:
        """Pending sub-agent gaps route to agent_spawn."""
        sample_state["pending_agent_gaps"] = ["specialized sub-agent for: data gathering"]
        assert route_after_structure_analysis(sample_state) == "agent_spawn"

    def test_routes_to_tool_create(self, sample_state: dict[str, Any]) -> None:
        """Pending tool gaps (and no agent gaps) route to tool_create."""
        sample_state["pending_tool_gaps"] = ["custom tool 'rss_aggregator'"]
        assert route_after_structure_analysis(sample_state) == "tool_create"

    def test_routes_to_execute_when_no_gaps(self, sample_state: dict[str, Any]) -> None:
        """No gaps route to the execute loop."""
        assert route_after_structure_analysis(sample_state) == "execute"

    def test_agent_gaps_skip_when_already_spawned(self, sample_state: dict[str, Any]) -> None:
        """Agent gaps are suppressed once sub-agents are spawned → execute."""
        sample_state["pending_agent_gaps"] = ["specialized sub-agent for: data gathering"]
        sample_state["sub_agents_spawned"] = [{"name": "a", "id": "1"}]
        assert route_after_structure_analysis(sample_state) == "execute"

    def test_agent_gaps_take_priority_over_tool_gaps(self, sample_state: dict[str, Any]) -> None:
        """Agent gaps take priority over tool gaps (mirrors route_after_reflect)."""
        sample_state["pending_agent_gaps"] = ["specialized sub-agent for: data gathering"]
        sample_state["pending_tool_gaps"] = ["custom tool 'rss_aggregator'"]
        assert route_after_structure_analysis(sample_state) == "agent_spawn"


class TestRouteAfterAgentSpawn:
    """Tests for route_after_agent_spawn routing function."""

    def test_route_after_agent_spawn_with_spawned(self, sample_state: dict[str, Any]) -> None:
        """Routes to delegate when sub_agents_spawned non-empty."""
        from src.graph.routers import route_after_agent_spawn

        sample_state["sub_agents_spawned"] = [
            {"name": "agent1", "id": "id1"},
            {"name": "agent2", "id": "id2"},
        ]
        result = route_after_agent_spawn(sample_state)
        assert result == "delegate"

    def test_route_after_agent_spawn_empty(self, sample_state: dict[str, Any]) -> None:
        """Routes to plan when no agents spawned."""
        from src.graph.routers import route_after_agent_spawn

        sample_state["sub_agents_spawned"] = []
        result = route_after_agent_spawn(sample_state)
        assert result == "plan"


class TestRouteAfterDelegate:
    """Tests for route_after_delegate routing function."""

    def test_route_after_delegate_all_success(self, sample_state: dict[str, Any]) -> None:
        """Routes to verify when all delegation_results successful."""
        from src.graph.routers import route_after_delegate

        sample_state["delegation_results"] = [
            {"success": True, "result": "Done"},
            {"success": True, "result": "Also done"},
        ]
        result = route_after_delegate(sample_state)
        assert result == "verify"

    def test_route_after_delegate_some_failure(self, sample_state: dict[str, Any]) -> None:
        """Routes to execute when any delegation fails."""
        from src.graph.routers import route_after_delegate

        sample_state["delegation_results"] = [
            {"success": True, "result": "Done"},
            {"success": False, "errors": ["Failed"]},
        ]
        result = route_after_delegate(sample_state)
        assert result == "execute"

    def test_route_after_delegate_empty_results(self, sample_state: dict[str, Any]) -> None:
        """Routes to verify when delegation_results is empty."""
        from src.graph.routers import route_after_delegate

        sample_state["delegation_results"] = []
        result = route_after_delegate(sample_state)
        assert result == "verify"
