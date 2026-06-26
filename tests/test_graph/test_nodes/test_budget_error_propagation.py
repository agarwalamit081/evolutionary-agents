"""Regression: a budget-exhausted gateway must propagate, not degrade.

``BudgetExhaustedError`` is a TERMINAL condition — the cheapest model tier is
already spent, so there is no recovery this attempt. Before this fix every
``_llm_*`` node helper had a broad ``except Exception`` that swallowed it into
``return None`` (heuristic fallback), so a budget-dead gateway (BUDGET_HARD_STOP)
kept the run looping on heuristics instead of reaching the worker's terminal
``JobStatus.BUDGET_EXHAUSTED`` handler (complex-arxiv-stats-4: the gateway died
mid-pass and the run looped to the iteration cap).

Each helper now re-raises ``BudgetExhaustedError`` ahead of its heuristic
fallback. These tests prove the re-raise by giving each helper a gateway whose
``acompletion`` raises it and asserting the helper propagates (not returns
``None``). They would fail against the pre-fix code, which swallowed the error.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.graph.enums import Strategy
from src.graph.factory import initial_state
from src.graph.nodes.classify import _llm_classify
from src.graph.nodes.plan import _llm_plan
from src.graph.nodes.reflect import _llm_reflect
from src.graph.nodes.verify import _llm_verify
from src.llm.exceptions import BudgetExhaustedError


def _raising_gateway() -> MagicMock:
    """A gateway whose acompletion always raises BudgetExhaustedError."""
    gw = MagicMock()
    gw.acompletion = AsyncMock(side_effect=BudgetExhaustedError("budget exhausted"))
    return gw


@pytest.mark.asyncio
async def test_llm_classify_propagates_budget_error() -> None:
    """classify must not swallow a budget-exhausted gateway into heuristics."""
    with pytest.raises(BudgetExhaustedError):
        await _llm_classify(_raising_gateway(), "explain quicksort")


@pytest.mark.asyncio
async def test_llm_plan_propagates_budget_error() -> None:
    """plan must not swallow a budget-exhausted gateway into heuristics."""
    state = initial_state(goal_text="compute fibonacci", thread_id="t-plan")
    goal = state.get("current_goal")
    assert goal is not None
    with pytest.raises(BudgetExhaustedError):
        await _llm_plan(_raising_gateway(), goal, Strategy.REACT, state)


@pytest.mark.asyncio
async def test_llm_verify_propagates_budget_error() -> None:
    """verify must not swallow a budget-exhausted gateway into heuristics."""
    state = initial_state(goal_text="produce a report", thread_id="t-verify")
    with pytest.raises(BudgetExhaustedError):
        await _llm_verify(_raising_gateway(), state, "evidence text")


@pytest.mark.asyncio
async def test_llm_reflect_propagates_budget_error() -> None:
    """reflect must not swallow a budget-exhausted gateway into heuristics."""
    state = initial_state(goal_text="produce a report", thread_id="t-reflect")
    with pytest.raises(BudgetExhaustedError):
        await _llm_reflect(_raising_gateway(), state)
