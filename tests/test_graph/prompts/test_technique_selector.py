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


class TestInferAudience:
    """infer_audience maps goal text to a reader profile (Feature D)."""

    def test_detects_expert(self) -> None:
        assert TechniqueSelector.infer_audience("as an expert, derive the proof") == "expert"

    def test_detects_developer(self) -> None:
        assert TechniqueSelector.infer_audience("for a developer audience, explain the api") == "developer"

    def test_detects_executive(self) -> None:
        assert TechniqueSelector.infer_audience("summarize for the executive team") == "executive"

    def test_detects_enduser(self) -> None:
        assert TechniqueSelector.infer_audience("explain it to a beginner in simple terms") == "enduser"

    def test_no_match_returns_none(self) -> None:
        assert TechniqueSelector.infer_audience("calculate the sum of the series") is None

    def test_empty_or_none_returns_none(self) -> None:
        assert TechniqueSelector.infer_audience("") is None
        assert TechniqueSelector.infer_audience(None) is None  # type: ignore[arg-type]


class TestInferUncertainty:
    """infer_uncertainty maps goal text to a settledness level (Feature D)."""

    def test_detects_high(self) -> None:
        assert TechniqueSelector.infer_uncertainty("estimate roughly the value") == "high"

    def test_detects_medium(self) -> None:
        assert TechniqueSelector.infer_uncertainty("likely around ten units") == "medium"

    def test_detects_low(self) -> None:
        assert TechniqueSelector.infer_uncertainty("compute exactly the determinant") == "low"

    def test_no_match_returns_none(self) -> None:
        assert TechniqueSelector.infer_uncertainty("calculate the sum of the series") is None

    def test_empty_or_none_returns_none(self) -> None:
        assert TechniqueSelector.infer_uncertainty("") is None
        assert TechniqueSelector.infer_uncertainty(None) is None  # type: ignore[arg-type]


class TestSelectByAudience:
    """Audience narrowing is non-destructive: universal techniques are always
    kept; only techniques tagged for a DIFFERENT audience drop out (Feature D)."""

    def test_expert_audience_keeps_tagged_and_universal(self) -> None:
        """An expert reasoning goal keeps first_principles (tagged expert) AND
        the universal chain_of_thought/self_ask/step_back."""
        selected = TechniqueSelector().select(
            complexity=TaskComplexity.CRITICAL, node=NODE_PLAN,
            goal_pattern="reasoning", audience="expert",
        )
        names = {t.name for t in selected}
        assert "first_principles" in names  # tagged expert/developer
        assert "chain_of_thought" in names  # universal (empty audiences)
        assert "self_ask" in names

    def test_enduser_audience_drops_expert_tagged(self) -> None:
        """An enduser reasoning goal drops first_principles (tagged expert/
        developer, not enduser) but keeps the universal techniques."""
        selected = TechniqueSelector().select(
            complexity=TaskComplexity.CRITICAL, node=NODE_PLAN,
            goal_pattern="reasoning", audience="enduser",
        )
        names = {t.name for t in selected}
        assert "first_principles" not in names  # dropped — wrong audience
        assert "chain_of_thought" in names     # universal, retained

    def test_no_audience_is_unchanged(self) -> None:
        """Omitting audience (the default) leaves the full candidate set."""
        selected = TechniqueSelector().select(
            complexity=TaskComplexity.CRITICAL, node=NODE_PLAN, goal_pattern="reasoning",
        )
        names = {t.name for t in selected}
        assert "first_principles" in names  # would be dropped only for enduser


class TestSelectByUncertainty:
    """Uncertainty narrowing drops techniques tagged for other levels (Feature D)."""

    def test_high_uncertainty_keeps_analogy_drops_synthesis(self) -> None:
        selected = TechniqueSelector().select(
            complexity=TaskComplexity.CRITICAL, node=NODE_PLAN,
            goal_pattern="analysis", uncertainty="high",
        )
        names = {t.name for t in selected}
        assert "analogy" in names        # tagged {high, medium}
        assert "synthesis" not in names  # tagged {medium} — dropped for high


class TestNewReasoningEntries:
    """The three new reasoning entries are reachable when their factors match."""

    def test_first_principles_for_expert_reasoning(self) -> None:
        selected = TechniqueSelector().select(
            complexity=TaskComplexity.CRITICAL, node=NODE_PLAN,
            goal_pattern="reasoning", audience="expert",
        )
        assert "first_principles" in {t.name for t in selected}

    def test_synthesis_for_executive_analysis(self) -> None:
        selected = TechniqueSelector().select(
            complexity=TaskComplexity.CRITICAL, node=NODE_PLAN,
            goal_pattern="analysis", audience="executive",
        )
        assert "synthesis" in {t.name for t in selected}

    def test_analogy_for_enduser_analysis(self) -> None:
        selected = TechniqueSelector().select(
            complexity=TaskComplexity.CRITICAL, node=NODE_PLAN,
            goal_pattern="analysis", audience="enduser",
        )
        assert "analogy" in {t.name for t in selected}


class TestRefinedIntentThreading:
    """The wrapper threads refined_intent (Feature A) into audience inference."""

    def test_refined_intent_drives_audience_narrowing(self) -> None:
        """A reasoning goal whose refined_intent names a beginner audience drops
        the expert-tagged first_principles; without refined_intent it is kept."""
        with_intent = select_techniques_for_node(
            complexity=TaskComplexity.CRITICAL, node=NODE_PLAN,
            goal_text="derive why the algorithm works",
            refined_intent="explain it to a beginner in simple terms",
        )
        without_intent = select_techniques_for_node(
            complexity=TaskComplexity.CRITICAL, node=NODE_PLAN,
            goal_text="derive why the algorithm works",
        )
        assert "first_principles" not in {t.name for t in with_intent}
        assert "first_principles" in {t.name for t in without_intent}


class TestBackwardCompatSnapshot:
    """No-new-kwargs selections for patterned goals are byte-identical to the
    pre-Feature-D technique mix (the new entries carry no math/code/
    verification/writing tags, so those narrowings are untouched)."""

    def test_math_plan_snapshot(self) -> None:
        """A critical math plan still selects exactly [chain_of_thought,
        self_ask] — the new reasoning entries are excluded by the math
        goal-pattern narrowing."""
        selected = select_techniques_for_node(
            complexity=TaskComplexity.CRITICAL, node=NODE_PLAN,
            goal_text="calculate the sum of the convergent series",
        )
        assert [t.name for t in selected] == ["chain_of_thought", "self_ask"]

    def test_reflect_node_set_unchanged(self) -> None:
        """The reflect node still returns only the two reflect techniques — the
        new entries are not registered on NODE_REFLECT."""
        selected = TechniqueSelector().select(
            complexity=TaskComplexity.COMPLEX, node=NODE_REFLECT, goal_pattern=None,
        )
        assert {t.name for t in selected} <= {"reflection", "self_refine"}

    def test_verify_node_set_unchanged(self) -> None:
        """The verify node still returns only the two verify techniques."""
        selected = TechniqueSelector().select(
            complexity=TaskComplexity.CRITICAL, node=NODE_VERIFY, goal_pattern=None,
        )
        assert {t.name for t in selected} <= {"chain_of_verification", "checklist_prompting"}
