"""Tests for src.graph.prompts.technique_selector — §5 selection layer.

Covers: goal-pattern inference, complexity/node filtering, goal-pattern
narrowing, determinism (same input → same output), and budget capping
(including the guarantee that the top qualifier always survives).
"""

from __future__ import annotations

from src.graph.enums import TaskComplexity
from src.graph.prompts import select_techniques_for_node
from src.graph.prompts.technique_selector import (
    NODE_EXECUTE,
    NODE_PLAN,
    NODE_REFLECT,
    NODE_VERIFY,
    Technique,
    TechniqueSelector,
)


class TestInferGoalPattern:
    """infer_goal_pattern maps goal text to a technique family."""

    def test_detects_math(self) -> None:
        assert TechniqueSelector.infer_goal_pattern("calculate the sum of the series") == "math"

    def test_detects_code(self) -> None:
        assert TechniqueSelector.infer_goal_pattern("refactor the function for clarity") == "code"

    def test_detects_verification(self) -> None:
        assert TechniqueSelector.infer_goal_pattern("verify the migration is complete") == "verification"

    def test_no_match_returns_none(self) -> None:
        assert TechniqueSelector.infer_goal_pattern("a vague unstructured sentence") is None

    def test_empty_or_none_returns_none(self) -> None:
        assert TechniqueSelector.infer_goal_pattern("") is None
        assert TechniqueSelector.infer_goal_pattern(None) is None  # type: ignore[arg-type]


class TestSelectByComplexity:
    """Selection varies by classified complexity (§5 requirement)."""

    def test_critical_plan_yields_reasoning_techniques(self) -> None:
        selected = TechniqueSelector().select(
            complexity=TaskComplexity.CRITICAL, node=NODE_PLAN, goal_pattern="math",
        )
        names = [t.name for t in selected]
        # chain_of_thought is the highest-priority math/reasoning technique.
        assert "chain_of_thought" in names
        # zero_shot is TRIVIAL/SIMPLE-only, must NOT appear for CRITICAL.
        assert "zero_shot" not in names

    def test_simple_plan_yields_direct_technique(self) -> None:
        selected = TechniqueSelector().select(
            complexity=TaskComplexity.SIMPLE, node=NODE_PLAN, goal_pattern=None,
        )
        names = [t.name for t in selected]
        assert "zero_shot" in names
        # chain_of_thought is COMPLEX/CRITICAL-only, must NOT appear for SIMPLE.
        assert "chain_of_thought" not in names


class TestSelectByGoalPattern:
    """Goal-pattern narrowing keeps only matching techniques when any match."""

    def test_math_pattern_keeps_math_techniques(self) -> None:
        selected = TechniqueSelector().select(
            complexity=TaskComplexity.CRITICAL, node=NODE_PLAN, goal_pattern="math",
        )
        # Every returned technique must either match 'math' or be the broad
        # fallback set — here narrowing activates, so all match 'math'.
        assert all("math" in t.goal_patterns for t in selected)

    def test_unmatched_pattern_falls_back_to_broad(self) -> None:
        """An inferred pattern with no specific matches falls back to broad."""
        selected = TechniqueSelector().select(
            complexity=TaskComplexity.COMPLEX, node=NODE_PLAN, goal_pattern="writing",
        )
        # No plan-node technique lists 'writing', so the broad COMPLEX+plan
        # candidates survive rather than returning empty.
        assert len(selected) >= 1


class TestSelectByNode:
    """Selection respects node applicability."""

    def test_reflect_node_only_returns_reflect_techniques(self) -> None:
        selected = TechniqueSelector().select(
            complexity=TaskComplexity.COMPLEX, node=NODE_REFLECT, goal_pattern=None,
        )
        names = {t.name for t in selected}
        assert names <= {"reflection", "self_refine"}

    def test_verify_node_only_returns_verify_techniques(self) -> None:
        selected = TechniqueSelector().select(
            complexity=TaskComplexity.CRITICAL, node=NODE_VERIFY, goal_pattern=None,
        )
        names = {t.name for t in selected}
        assert names <= {"chain_of_verification", "checklist_prompting"}

    def test_execute_node_eligible(self) -> None:
        selected = TechniqueSelector().select(
            complexity=TaskComplexity.COMPLEX, node=NODE_EXECUTE, goal_pattern="code",
        )
        assert len(selected) >= 1


class TestDeterminism:
    """Same inputs always yield the same technique list (§5 requirement)."""

    def test_identical_inputs_identical_output(self) -> None:
        sel = TechniqueSelector()
        a = sel.select(TaskComplexity.CRITICAL, NODE_PLAN, "math")
        b = sel.select(TaskComplexity.CRITICAL, NODE_PLAN, "math")
        assert [t.name for t in a] == [t.name for t in b]


class TestBudgetCap:
    """Token-budget capping bounds the number of techniques injected."""

    def test_default_budget_fits_multiple(self) -> None:
        selected = TechniqueSelector().select(
            complexity=TaskComplexity.CRITICAL, node=NODE_PLAN, goal_pattern="math",
        )
        total = sum(t.token_cost_estimate for t in selected)
        assert total <= 512
        assert len(selected) >= 1

    def test_tight_budget_still_returns_top_qualifier(self) -> None:
        """A tiny budget never zeroes out selection — the strongest survives."""
        selected = TechniqueSelector().select(
            complexity=TaskComplexity.CRITICAL, node=NODE_PLAN, goal_pattern="math",
            budget_tokens=1,
        )
        assert len(selected) == 1
        # Highest-priority math/reasoning technique survives.
        assert selected[0].name == "chain_of_thought"

    def test_custom_registry_empty_budget_returns_top(self) -> None:
        """With a custom registry and tiny budget, the top qualifier is returned."""
        custom = [
            Technique(
                name="big",
                body="x" * 1000,
                applies_to_complexities=frozenset({TaskComplexity.COMPLEX}),
                nodes=frozenset({NODE_PLAN}),
                token_cost_estimate=5000,
                priority=10,
            ),
        ]
        selected = TechniqueSelector(registry=custom).select(
            complexity=TaskComplexity.COMPLEX, node=NODE_PLAN, budget_tokens=1,
        )
        assert len(selected) == 1
        assert selected[0].name == "big"


class TestSelectTechniquesForNode:
    """The centralized wiring helper used by plan/execute/reflect/verify nodes."""

    def test_none_complexity_returns_empty(self) -> None:
        """A None complexity (heuristic-fallback path) never applies techniques —
        the helper must not raise and must return an empty list."""
        assert select_techniques_for_node(complexity=None, node=NODE_PLAN) == []

    def test_infers_goal_pattern_from_text(self) -> None:
        """The helper infers the goal pattern internally from goal_text, so a
        math goal selects math/reasoning techniques without the caller passing a
        goal_pattern explicitly."""
        selected = select_techniques_for_node(
            complexity=TaskComplexity.CRITICAL,
            node=NODE_PLAN,
            goal_text="calculate the sum of the convergent series",
        )
        names = {t.name for t in selected}
        assert "chain_of_thought" in names  # top math/reasoning qualifier

    def test_consistent_with_direct_select(self) -> None:
        """The wrapper delegates to TechniqueSelector.select, so for identical
        inputs it returns the same technique names as a direct select() call."""
        via_wrapper = select_techniques_for_node(
            complexity=TaskComplexity.CRITICAL, node=NODE_PLAN, goal_text="factor 91",
        )
        via_direct = TechniqueSelector().select(
            complexity=TaskComplexity.CRITICAL,
            node=NODE_PLAN,
            goal_pattern=TechniqueSelector.infer_goal_pattern("factor 91"),
        )
        assert [t.name for t in via_wrapper] == [t.name for t in via_direct]
