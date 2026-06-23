"""Tests for src.graph.iteration_cap — complexity-aware runtime cap (B1)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.config.settings import AgentSettings
from src.graph.enums import TaskComplexity
from src.graph.iteration_cap import effective_max_iterations


def _state(complexity: TaskComplexity | None, max_iterations: int | None = None) -> dict[str, object]:
    """Build a minimal state carrying a goal with the given complexity.

    ``current_goal`` is a plain namespace exposing ``.complexity`` — exactly
    what ``_goal_complexity`` reads via ``getattr``.
    """
    state: dict[str, object] = {
        "current_goal": SimpleNamespace(complexity=complexity) if complexity else None,
    }
    if max_iterations is not None:
        state["max_iterations"] = max_iterations
    return state


class TestEffectiveMaxIterations:
    """``effective_max_iterations`` maps complexity → tier cap (B1)."""

    def test_trivial_caps_low(self) -> None:
        assert effective_max_iterations(_state(TaskComplexity.TRIVIAL)) == 12

    def test_simple(self) -> None:
        assert effective_max_iterations(_state(TaskComplexity.SIMPLE)) == 15

    def test_complex_keeps_headroom(self) -> None:
        assert effective_max_iterations(_state(TaskComplexity.COMPLEX)) == 60

    def test_critical_keeps_headroom(self) -> None:
        assert effective_max_iterations(_state(TaskComplexity.CRITICAL)) == 60

    def test_trivial_caps_below_complex(self) -> None:
        """The headline contract: a TRIVIAL goal stops earlier than a COMPLEX one."""
        assert effective_max_iterations(_state(TaskComplexity.TRIVIAL)) < effective_max_iterations(
            _state(TaskComplexity.COMPLEX)
        )

    def test_unclassified_goal_defaults_to_simple(self) -> None:
        """Before classify runs (or the heuristic path) complexity is unset → SIMPLE,
        the historical default cap basis."""
        assert effective_max_iterations(_state(None)) == 15

    def test_explicit_pin_wins_over_complexity(self) -> None:
        """A caller-pinned cap (CLI --max-iterations / eval spec / worker job) always
        wins — callers that pin a cap know their budget."""
        assert effective_max_iterations(_state(TaskComplexity.COMPLEX, 7)) == 7
        assert effective_max_iterations(_state(TaskComplexity.TRIVIAL, 100)) == 100

    def test_string_goal_falls_back_to_simple(self) -> None:
        """``current_goal`` may briefly be a plain string; ``getattr`` → None → SIMPLE."""
        assert effective_max_iterations({"current_goal": "some goal text"}) == 15


class TestIterationCapValidator:
    """The recursion-limit basis must cover every tier cap (B1 invariant)."""

    def test_rejects_basis_below_tier_cap(self) -> None:
        """``max_iterations`` < a tier cap would let that tier's run hit
        GraphRecursionError before its cap — rejected at startup."""
        with pytest.raises(ValueError, match="recursion_limit basis"):
            AgentSettings(max_iterations=10)

    def test_accepts_basis_at_tier_cap(self) -> None:
        """The default basis (>= every tier cap) constructs cleanly."""
        AgentSettings(max_iterations=60)

    def test_rejects_basis_below_raised_complex_cap(self) -> None:
        """Raising a tier cap above the basis is also caught."""
        with pytest.raises(ValueError, match="recursion_limit basis"):
            AgentSettings(max_iterations=60, max_iterations_critical=100)
