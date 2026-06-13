"""Plan node — generates an execution plan from the classified task."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

from loguru import logger

from src.graph.enums import GoalStatus, Phase, Strategy, TaskComplexity
from src.graph.models import Goal, PlanStep
from src.graph.state import AgentState

if TYPE_CHECKING:
    from src.llm.gateway import LLMGateway
    from src.tools.registry import ToolRegistry


async def plan_node(
    state: AgentState,
    *,
    gateway: LLMGateway | None = None,
    tools: ToolRegistry | None = None,
) -> dict[str, Any]:
    """Generate an execution plan based on the classified goal and strategy.

    Creates a list of PlanSteps from the goal and strategy.
    When a gateway is provided, LLM-based planning can generate richer plans.

    Args:
        state: Current agent state with classified goal.
        gateway: Optional LLM gateway for LLM-enhanced planning.
        tools: Optional ToolRegistry for dynamic tool list in prompts.

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

    # Try LLM planning first, fall back to heuristics
    plan_steps: list[PlanStep] | None = None
    if gateway is not None:
        plan_steps = await _llm_plan(gateway, goal, strategy, state, tools)

    if plan_steps is None:
        plan_steps = _generate_plan(goal.text, strategy)

    # Budget-aware sizing: cap the plan to fit the remaining iteration budget
    # so large multi-unit goals decompose within max_iterations instead of
    # stretching the run to exhaustion. Applies to both LLM and heuristic plans.
    from src.config import get_settings
    max_iter = state.get("max_iterations", 20)
    remaining = max(0, max_iter - iteration_count)
    max_steps = min(get_settings().agent.planning_max_steps, max(1, remaining))
    if len(plan_steps) > max_steps:
        logger.info(
            f"Capping plan from {len(plan_steps)} to {max_steps} steps "
            f"(remaining iterations: {remaining})"
        )
        plan_steps = plan_steps[:max_steps]

    logger.info(f"Generated {len(plan_steps)} plan steps")

    return {
        "phase": Phase.RETRIEVE_MEMORY,
        "plan_steps": plan_steps,
        "current_step_index": 0,
        "iteration_count": iteration_count + 1,
    }


async def _llm_plan(
    gateway: LLMGateway,
    goal: Goal,
    strategy: Strategy,
    state: AgentState,
    tools: ToolRegistry | None = None,
) -> list[PlanStep] | None:
    """Attempt LLM-based plan generation. Returns None on failure."""
    try:
        from src.graph.prompts import PLAN_SYSTEM, PLAN_USER
        from src.graph.schemas import GeneratedPlan
        from src.llm.structured_output import StructuredOutputManager

        # Build memory context if available
        memories = state.get("retrieved_memories", [])
        memory_ctx = ""
        if memories:
            memory_ctx = "\nRelevant context from memory:\n" + "\n".join(
                f"- {m}" for m in memories[:5]
            )

        max_iterations = state.get("max_iterations", 20)
        iteration_count = state.get("iteration_count", 0)
        remaining_iterations = max(0, max_iterations - iteration_count)
        user_prompt = PLAN_USER.format(
            goal_text=goal.text,
            strategy=strategy.value,
            complexity=goal.complexity.value if goal.complexity else "simple",
            estimated_steps="auto",
            remaining_iterations=remaining_iterations,
            max_iterations=max_iterations,
            memory_context=memory_ctx,
        )
        # Build dynamic tool list for the plan prompt
        if tools is not None:
            tool_names = [t["function"]["name"] for t in tools.list_tools()]
        else:
            tool_names = [
                "code_executor", "web_search", "file_reader",
                "file_writer", "code_validator", "self_inspect", "memory_search",
            ]
        system_prompt = PLAN_SYSTEM.format(available_tools=", ".join(tool_names))

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = await gateway.acompletion(
            messages=messages,
            complexity=TaskComplexity.SIMPLE,
        )

        extractor = StructuredOutputManager()
        plan = await extractor.extract(response.content, GeneratedPlan)
        if plan is None or not plan.steps:
            return None

        steps: list[PlanStep] = []
        for gen_step in plan.steps:
            steps.append(PlanStep(
                id=uuid4().hex[:8],
                description=gen_step.description,
                status=GoalStatus.PENDING,
            ))

        logger.info(f"LLM generated {len(steps)} plan steps")
        return steps
    except Exception as e:
        logger.debug(f"LLM planning failed, using heuristics: {e}")
        return None


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
