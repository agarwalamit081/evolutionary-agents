"""Regression tests for the immutable submitted_goal objective anchor.

Context (#254 structural backstop): the objective-drift fix was prompt-only — it
rendered the goal as an OBJECTIVE anchor and recalled memory as ADVISORY in the
plan/execute prompts, but it left ``current_goal.text`` (a mutable ``Goal``
object) as the OBJECTIVE source AND as the recall query. If any node ever wrote a
drifted ``current_goal.text`` (a recalled skill leaking into it), the OBJECTIVE
would drift AND recall would compound the drift by re-querying on the drifted
text — pulling in ever-more-irrelevant context.

The fix: an immutable ``submitted_goal`` (set once in ``initial_state``, never
overwritten) sourced via ``objective_goal_text()`` by the recall query and the
plan/execute/verify OBJECTIVE slots. These tests lock that guarantee — a drifted
``current_goal.text`` cannot change the objective or redirect recall.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from src.graph.factory import initial_state
from src.graph.models import Goal
from src.graph.nodes.memory import retrieve_memory_node
from src.graph.state import objective_goal_text

# A submitted objective that shares nothing semantically with the drifted text.
_SUBMITTED = "Write the first 50 primes to results/primes.csv as a single column."
# A recalled skill from a PRIOR, UNRELATED run that has contaminated current_goal.
_DRIFTED = "q01 skill: normalize the e-commerce orders.csv for revenue analysis"


def _drifted_state() -> dict[str, object]:
    """A state whose current_goal.text has drifted off the submitted objective."""
    return {
        "submitted_goal": _SUBMITTED,
        "current_goal": Goal(text=_DRIFTED),
    }


def test_initial_state_seeds_submitted_goal() -> None:
    """initial_state freezes the literal goal text as the immutable anchor."""
    state = initial_state(goal_text=_SUBMITTED, thread_id="t-anchor")
    assert state["submitted_goal"] == _SUBMITTED
    # current_goal.text starts equal — they only diverge if something drifts it.
    assert state["current_goal"].text == _SUBMITTED


def test_objective_goal_text_prefers_anchor_over_drifted_current_goal() -> None:
    """The anchor wins: a drifted current_goal.text cannot change the objective."""
    assert objective_goal_text(_drifted_state()) == _SUBMITTED


def test_objective_goal_text_falls_back_to_current_goal_when_anchor_absent() -> None:
    """Back-compat: a checkpointed state predating the anchor still resolves."""
    state: dict[str, object] = {"current_goal": Goal(text=_SUBMITTED)}
    assert objective_goal_text(state) == _SUBMITTED


def test_objective_goal_text_empty_when_no_goal() -> None:
    """No anchor and no goal → empty string (not an exception)."""
    assert objective_goal_text({}) == ""


async def test_retrieve_memory_keys_recall_on_anchor_not_drifted_goal() -> None:
    """The headline contamination guard: recall queries the IMMUTABLE objective.

    Even when current_goal.text has drifted onto a foreign (e-commerce) skill,
    retrieve_memory_node must query memory with the submitted objective — never
    the drifted text — so recall cannot be redirected or compounded by drift.
    """
    memory = AsyncMock()
    memory.retrieve_context = AsyncMock(return_value=[])
    memory.retrieve_facts = AsyncMock(return_value=[])
    memory.retrieve_skills = AsyncMock(return_value=[])
    memory.warm.retrieve = AsyncMock(return_value=[])

    await retrieve_memory_node(_drifted_state(), memory=memory)

    # The recall query is the submitted objective, NOT the drifted current_goal.
    memory.retrieve_context.assert_awaited_once_with(query=_SUBMITTED, limit=5)
    memory.retrieve_facts.assert_awaited_once_with(query=_SUBMITTED, limit=3)
    memory.retrieve_skills.assert_awaited_once_with(query=_SUBMITTED, limit=3)
    # And never the drifted text.
    for call in memory.retrieve_context.await_args_list:
        assert call.kwargs.get("query") != _DRIFTED
