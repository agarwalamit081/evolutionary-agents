"""StateCheck — assertions over live graph STATE fields (not deliverables).

The other correctness check kinds (Structural / Execution / Golden / Oracle)
read FILES; StateCheck reads run state directly so a node whose decision lives
only in state — classify's ``current_goal.complexity`` — can be scored. That is
what makes the optimizer's real canary (``GoldenCanary``) sensitive to a
classify-prompt candidate: data-correctness specs (``q01``…) are inert to
classify prose, so without a state-asserting spec a promotion was structurally
impossible. See ``src/eval/checks.py::StateCheck``.

Hermetic: pure in-process dict + object traversal, no files / no subprocess / no
LLM. Mirrors ``test_checks.py``'s shape (a ``_cfg``/``_state`` helper + one
class per concern).
"""

from __future__ import annotations

from typing import Any

import pytest

from src.eval.checks import StateCheck, run_checks
from src.eval.models import CheckConfig, GoalSpec
from src.graph.enums import TaskComplexity
from src.graph.models import Goal


def _cfg(assertions: list[dict[str, Any]], name: str = "state_check") -> CheckConfig:
    return CheckConfig(check_type="state", name=name, params={"assertions": assertions})


def _state(**overrides: Any) -> Any:
    """Build a partial ``AgentState`` dict for the checks under test.

    Returned as ``Any`` (not ``dict[str, Any]``) so it is assignable to the
    ``AgentState`` TypedDict the check signature declares — the check only reads
    the keys it asserts over, not the full state contract.
    """
    base: dict[str, Any] = {
        "current_goal": Goal(text="g", complexity=TaskComplexity.COMPLEX),
        "refined_intent": "compute factorial and prime factorization",
        "strategy": "step-by-step arithmetic then factorize",
        "tags": ["complex"],
    }
    base.update(overrides)
    return base


class TestStateCheck:
    @pytest.mark.asyncio
    async def test_no_assertions_is_a_trivial_pass(self) -> None:
        res = await StateCheck().check(_cfg([]), [], _state())
        assert res.passed is True
        assert res.score == 1.0
        assert res.evidence == {"note": "no state assertions declared"}

    @pytest.mark.asyncio
    async def test_eq_match(self) -> None:
        res = await StateCheck().check(
            _cfg([{"field": "refined_intent", "kind": "eq", "expected": "compute factorial and prime factorization"}]),
            [],
            _state(),
        )
        assert res.passed is True
        assert res.score == 1.0

    @pytest.mark.asyncio
    async def test_eq_is_case_insensitive(self) -> None:
        res = await StateCheck().check(
            _cfg([{"field": "strategy", "kind": "eq", "expected": "STEP-BY-STEP ARITHMETIC THEN FACTORIZE"}]),
            [],
            _state(),
        )
        assert res.passed is True  # both sides lowercased before comparison

    @pytest.mark.asyncio
    async def test_eq_mismatch_fails(self) -> None:
        res = await StateCheck().check(
            _cfg([{"field": "refined_intent", "kind": "eq", "expected": "something else"}]),
            [],
            _state(),
        )
        assert res.passed is False
        assert res.score == 0.0

    @pytest.mark.asyncio
    async def test_in_match(self) -> None:
        res = await StateCheck().check(
            _cfg([{"field": "tags.0", "kind": "in", "expected": ["complex", "critical"]}]),
            [],
            _state(),
        )
        assert res.passed is True

    @pytest.mark.asyncio
    async def test_in_accepts_scalar_expected(self) -> None:
        res = await StateCheck().check(
            _cfg([{"field": "strategy", "kind": "in", "expected": "step-by-step arithmetic then factorize"}]),
            [],
            _state(),
        )
        assert res.passed is True

    @pytest.mark.asyncio
    async def test_in_mismatch_fails(self) -> None:
        res = await StateCheck().check(
            _cfg([{"field": "strategy", "kind": "in", "expected": ["a", "b"]}]),
            [],
            _state(),
        )
        assert res.passed is False

    @pytest.mark.asyncio
    async def test_contains_substring_match(self) -> None:
        res = await StateCheck().check(
            _cfg([{"field": "refined_intent", "kind": "contains", "expected": "factorial"}]),
            [],
            _state(),
        )
        assert res.passed is True

    @pytest.mark.asyncio
    async def test_contains_missing_substring_fails(self) -> None:
        res = await StateCheck().check(
            _cfg([{"field": "refined_intent", "kind": "contains", "expected": "nonexistent"}]),
            [],
            _state(),
        )
        assert res.passed is False

    @pytest.mark.asyncio
    async def test_traverses_nested_goal_with_enum_complexity(self) -> None:
        """The core case: ``current_goal.complexity`` is a TaskComplexity enum on a Goal object.

        ``_state_get`` walks the ``Goal`` via getattr (``_dotted_get`` would return
        None — dict/list only); ``_normalize_state_value`` unwraps the enum to its
        lowercase ``.value`` so it compares equal to the plain string ``"complex"``.
        """
        res = await StateCheck().check(
            _cfg([{"field": "current_goal.complexity", "kind": "in", "expected": ["complex", "critical"]}]),
            [],
            _state(current_goal=Goal(text="g", complexity=TaskComplexity.COMPLEX)),
        )
        assert res.passed is True
        assert res.evidence["assertions"][0]["got"] == "complex"

    @pytest.mark.asyncio
    async def test_enum_complexity_wrong_tier_fails(self) -> None:
        res = await StateCheck().check(
            _cfg([{"field": "current_goal.complexity", "kind": "in", "expected": ["trivial", "simple"]}]),
            [],
            _state(current_goal=Goal(text="g", complexity=TaskComplexity.COMPLEX)),
        )
        assert res.passed is False

    @pytest.mark.asyncio
    async def test_missing_field_fails(self) -> None:
        res = await StateCheck().check(
            _cfg([{"field": "nonexistent.deep.path", "kind": "eq", "expected": "x"}]),
            [],
            _state(),
        )
        assert res.passed is False
        # None normalizes to "" for evidence.
        assert res.evidence["assertions"][0]["got"] == ""

    @pytest.mark.asyncio
    async def test_unknown_kind_fails(self) -> None:
        res = await StateCheck().check(
            _cfg([{"field": "refined_intent", "kind": "regex", "expected": ".*"}]),
            [],
            _state(),
        )
        assert res.passed is False
        assert "unknown" in res.evidence["assertions"][0]["reason"]

    @pytest.mark.asyncio
    async def test_partial_score_is_fraction_passed(self) -> None:
        res = await StateCheck().check(
            _cfg(
                [
                    {"field": "current_goal.complexity", "kind": "in", "expected": ["complex", "critical"]},
                    {"field": "refined_intent", "kind": "eq", "expected": "wrong"},
                ]
            ),
            [],
            _state(),
        )
        assert res.passed is False  # not all assertions held
        assert res.score == 0.5


class TestStateCheckViaRunChecks:
    """StateCheck composes with ``run_checks`` like the other kinds (state is forwarded)."""

    @pytest.mark.asyncio
    async def test_run_checks_passes_state_to_state_check(self) -> None:
        spec = GoalSpec(
            spec_id="classify_canary",
            name="classify_canary",
            goal_text="g",
            checks=[
                _cfg([{"field": "current_goal.complexity", "kind": "in", "expected": ["complex", "critical"]}])
            ],
        )
        result = await run_checks(
            spec, [], _state(current_goal=Goal(text="g", complexity=TaskComplexity.COMPLEX))
        )
        assert result.passed is True
        assert result.overall_score == 1.0
        assert result.checks[0].check_type == "state"

    @pytest.mark.asyncio
    async def test_state_check_failure_blocks_overall_pass(self) -> None:
        spec = GoalSpec(
            spec_id="classify_canary",
            name="classify_canary",
            goal_text="g",
            checks=[
                _cfg([{"field": "current_goal.complexity", "kind": "in", "expected": ["trivial", "simple"]}])
            ],
        )
        result = await run_checks(
            spec, [], _state(current_goal=Goal(text="g", complexity=TaskComplexity.COMPLEX))
        )
        assert result.passed is False
        assert result.overall_score == 0.0
