"""Plan node — generates an execution plan from the classified task."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from loguru import logger

from turing_agent.graph.enums import GoalStatus, Phase, Strategy
from turing_agent.graph.models import PlanStep
from turing_agent.graph.state import AgentState


async def plan_node(state: AgentState) -> dict[str, Any]:
    """Generate an execution plan based on the classified goal and strategy.

    Creates a list of PlanSteps from the goal and strategy.
    For now uses heuristic plan generation; LLM-based planning is an enhancement.

    Args:
        state: Current agent state with classified goal.

    Returns:
        Partial state update with plan_steps and phase transition.
    """
    goal = state.get("current_goal")
    strategy = state.get("strategy", Strategy.REACT)
    iteration_count = state.get("iteration_count", 0)

    if not goal or not goal.text:
        return {
            "phase": Phase.ERROR_HANDLER,
            "errors": ["plan: No goal text available"],
        }

    logger.info(f"Planning for goal (strategy={strategy.value}): {goal.text[:80]}...")

    # Generate plan steps based on strategy
    plan_steps = _generate_plan(goal.text, strategy)

    logger.info(f"Generated {len(plan_steps)} plan steps")

    return {
        "phase": Phase.RETRIEVE_MEMORY,
        "plan_steps": plan_steps,
        "current_step_index": 0,
        "iteration_count": iteration_count + 1,
    }


def _generate_plan(goal_text: str, strategy: Strategy) -> list[PlanStep]:
    """Generate plan steps based on goal and strategy.

    This is a heuristic planner. In production, the LLM gateway would
    generate plans dynamically.
    """
    steps: list[PlanStep] = []

    if strategy == Strategy.DIRECT:
        # Single-step direct execution
        steps.append(PlanStep(
            description=f"Directly address: {goal_text}",
            status=GoalStatus.PENDING,
        ))

    elif strategy == Strategy.REACT:
        # Reasoning + Acting loop
        steps.append(PlanStep(
            id=uuid4().hex[:8],
            description="Analyze the task requirements and gather context",
            status=GoalStatus.PENDING,
        ))
        steps.append(PlanStep(
            id=uuid4().hex[:8],
            description=f"Execute core task: {goal_text[:100]}",
            status=GoalStatus.PENDING,
        ))
        steps.append(PlanStep(
            id=uuid4().hex[:8],
            description="Review results and verify against success criteria",
            status=GoalStatus.PENDING,
        ))

    elif strategy == Strategy.PLANNING:
        # Multi-step structured plan
        steps.append(PlanStep(
            id=uuid4().hex[:8],
            description="Break down the task into sub-components",
            status=GoalStatus.PENDING,
        ))
        steps.append(PlanStep(
            id=uuid4().hex[:8],
            description="Execute each sub-component sequentially",
            status=GoalStatus.PENDING,
        ))
        steps.append(PlanStep(
            id=uuid4().hex[:8],
            description="Integrate results from all sub-components",
            status=GoalStatus.PENDING,
        ))
        steps.append(PlanStep(
            id=uuid4().hex[:8],
            description="Validate the integrated result",
            status=GoalStatus.PENDING,
        ))

    elif strategy == Strategy.REFLECTION:
        # Execute + reflect cycles
        steps.append(PlanStep(
            id=uuid4().hex[:8],
            description="Initial execution attempt",
            status=GoalStatus.PENDING,
        ))
        steps.append(PlanStep(
            id=uuid4().hex[:8],
            description="Self-critique and identify improvements",
            status=GoalStatus.PENDING,
        ))
        steps.append(PlanStep(
            id=uuid4().hex[:8],
            description="Refined execution based on critique",
            status=GoalStatus.PENDING,
        ))
        steps.append(PlanStep(
            id=uuid4().hex[:8],
            description="Final review and validation",
            status=GoalStatus.PENDING,
        ))

    elif strategy == Strategy.TOT:
        # Tree of thought: explore multiple approaches
        steps.append(PlanStep(
            id=uuid4().hex[:8],
            description="Generate multiple solution approaches",
            status=GoalStatus.PENDING,
        ))
        steps.append(PlanStep(
            id=uuid4().hex[:8],
            description="Evaluate and compare approaches",
            status=GoalStatus.PENDING,
        ))
        steps.append(PlanStep(
            id=uuid4().hex[:8],
            description="Execute the best approach",
            status=GoalStatus.PENDING,
        ))

    elif strategy == Strategy.DEBATE:
        # Multi-perspective analysis
        steps.append(PlanStep(
            id=uuid4().hex[:8],
            description="Present arguments from perspective A",
            status=GoalStatus.PENDING,
        ))
        steps.append(PlanStep(
            id=uuid4().hex[:8],
            description="Present arguments from perspective B",
            status=GoalStatus.PENDING,
        ))
        steps.append(PlanStep(
            id=uuid4().hex[:8],
            description="Synthesize and resolve into final answer",
            status=GoalStatus.PENDING,
        ))

    else:
        # Fallback: simple 3-step plan
        steps.append(PlanStep(
            id=uuid4().hex[:8],
            description=f"Address: {goal_text[:100]}",
            status=GoalStatus.PENDING,
        ))

    return steps
