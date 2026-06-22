"""Regression tests for objective-drift guard (#254).

Root cause: the plan/execute prompts interpolated ``memory_context`` (semantic
recall over the goal text) as a bare ``"Relevant context"`` block right under the
goal, with no primacy on the goal and no advisory label on the memory. A goal
whose recall surfaced a *different* run's skill (e.g. a Collatz goal recalling a
q01 e-commerce skill) let the planner drift onto the recalled objective.

The fix: both prompts now render the goal as an explicit OBJECTIVE anchor and
wrap recalled memory in an ``ADVISORY ONLY`` frame that says it is NOT the
objective and must be discarded if it names a different deliverable.

These tests render the prompts directly (no LLM) and assert the submitted goal
stays in the OBJECTIVE slot while a foreign recalled skill lands ONLY inside the
advisory frame — the exact contamination the bug allowed.
"""

from __future__ import annotations

from src.graph.prompts import EXECUTE_SYSTEM, PLAN_USER

# A submitted goal that has NOTHING in common with the recalled memory below.
_COLLATZ_GOAL = (
    "Prove the Collatz conjecture for every starting value 1..100 and write the "
    "trajectory table to results/collatz/trajectories.csv"
)
# A recalled skill from a PRIOR, UNRELATED run (q01 e-commerce) — semantically
# near nothing in the Collatz goal but the kind of recall that contaminated it.
_ECOM_SKILL = (
    "q01 skill: normalize the e-commerce orders.csv via pandas.read_csv + dropna "
    "for revenue analysis"
)
_ADVISORY_MARKER = "ADVISORY ONLY"

# Mirrors the node's memory_ctx construction (plan.py / execute.py): a bare
# bulleted list with no header — the template supplies the advisory frame.
_ECOM_MEMORY_CTX = f"- {_ECOM_SKILL}"


def _split_objective_advisory(rendered: str) -> tuple[str, str]:
    """Split a rendered prompt into (objective_block, advisory_block) on the marker."""
    idx = rendered.index(_ADVISORY_MARKER)
    return rendered[:idx], rendered[idx:]


def test_plan_prompt_anchors_goal_and_labels_memory_advisory() -> None:
    rendered = PLAN_USER.format(
        goal_text=_COLLATZ_GOAL,
        strategy="complex",
        complexity="complex",
        estimated_steps="auto",
        remaining_iterations=15,
        max_iterations=20,
        memory_context=_ECOM_MEMORY_CTX,
        correction_context="",
    )
    objective, advisory = _split_objective_advisory(rendered)

    # The submitted goal is the OBJECTIVE, before any advisory memory.
    assert _COLLATZ_GOAL in objective
    # The advisory frame is present and carries the foreign skill.
    assert _ECOM_SKILL in advisory
    # Drift guard: the foreign skill does NOT leak into the OBJECTIVE slot.
    assert _ECOM_SKILL not in objective
    # The goal does not leak into the advisory block either.
    assert _COLLATZ_GOAL not in advisory


def test_execute_prompt_anchors_goal_and_labels_memory_advisory() -> None:
    rendered = EXECUTE_SYSTEM.format(
        goal_text=_COLLATZ_GOAL,
        completed_count=1,
        total_steps=4,
        step_description="compute trajectories for 1..100",
        memory_context=_ECOM_MEMORY_CTX,
        tool_results_context="",
    )
    objective, advisory = _split_objective_advisory(rendered)

    assert _COLLATZ_GOAL in objective
    assert _ECOM_SKILL in advisory
    assert _ECOM_SKILL not in objective
    assert _COLLATZ_GOAL not in advisory


def test_plan_prompt_drops_advisory_frame_when_no_memory() -> None:
    """With no recalled memory the advisory frame must not render (no empty frame)."""
    rendered = PLAN_USER.format(
        goal_text=_COLLATZ_GOAL,
        strategy="complex",
        complexity="complex",
        estimated_steps="auto",
        remaining_iterations=15,
        max_iterations=20,
        memory_context="",
        correction_context="",
    )
    assert _ADVISORY_MARKER not in rendered
    assert _COLLATZ_GOAL in rendered


def test_execute_prompt_drops_advisory_frame_when_no_memory() -> None:
    rendered = EXECUTE_SYSTEM.format(
        goal_text=_COLLATZ_GOAL,
        completed_count=0,
        total_steps=4,
        step_description="step one",
        memory_context="",
        tool_results_context="",
    )
    assert _ADVISORY_MARKER not in rendered
    assert _COLLATZ_GOAL in rendered
