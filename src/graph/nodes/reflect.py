"""Reflect node — self-reflection on execution progress."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from src.graph.enums import Confidence, Phase, TaskComplexity
from src.graph.models import ReflectionResult
from src.graph.state import AgentState

if TYPE_CHECKING:
    from src.llm.gateway import LLMGateway
    from src.tools.registry import ToolRegistry


async def reflect_node(
    state: AgentState,
    *,
    gateway: LLMGateway | None = None,
    tools: ToolRegistry | None = None,
) -> dict[str, Any]:
    """Perform self-reflection on the execution so far.

    Evaluates completed steps, tool results, and progress toward the goal.
    When gateway is provided, uses LLM for deeper analysis.
    Otherwise falls back to heuristic reflection.

    Args:
        state: Current agent state with execution results.
        gateway: Optional LLM gateway for LLM-enhanced reflection.
        tools: Optional ToolRegistry for gap detection.

    Returns:
        Partial state update with reflection results.
    """
    goal = state.get("current_goal")
    completed_steps = state.get("completed_steps", [])
    errors = state.get("errors", [])

    goal_text = goal.text if goal else "Unknown goal"
    logger.info(f"Reflecting on execution of: {goal_text[:60]}...")

    # Try LLM reflection first, fall back to heuristics
    if gateway is not None:
        result = await _llm_reflect(gateway, state)
        if result is not None:
            return result

    return _heuristic_reflect(state, goal_text, completed_steps, errors, tools)


def _heuristic_reflect(
    state: AgentState,
    goal_text: str,
    completed_steps: list[Any],
    errors: list[str],
    tools: ToolRegistry | None = None,
) -> dict[str, Any]:
    """Heuristic-based reflection using completion ratios."""
    plan_steps = state.get("plan_steps", [])
    total_steps = len(plan_steps) if plan_steps else 1
    completed_count = len(completed_steps) if completed_steps else 0
    completion_ratio = completed_count / total_steps if total_steps > 0 else 0.0

    has_errors = bool(errors)
    if has_errors:
        confidence = Confidence.LOW
    elif completion_ratio >= 0.8:
        confidence = Confidence.HIGH
    elif completion_ratio >= 0.5:
        confidence = Confidence.MEDIUM
    else:
        confidence = Confidence.LOW

    lessons: list[str] = []
    memory_obs: list[str] = []

    if completed_steps:
        lessons.append(f"Completed {completed_count}/{total_steps} steps successfully")

    if has_errors:
        lessons.append(f"Encountered {len(errors)} errors during execution")
        memory_obs.append(f"Error pattern: {errors[-1][:100] if errors else 'none'}")

    # Detect tool gaps from "Unknown tool" errors in tool results
    tool_results = state.get("tool_results", [])
    missing_tools: list[str] = []
    for tr in tool_results:
        if hasattr(tr, "error") and tr.error and "Unknown tool" in str(tr.error):
            # Confirm the tool is truly absent from registry (not just a typo)
            if tools is not None and tools.get_handler(tr.tool_name) is None:
                missing_tools.append(f"tool matching '{tr.tool_name}' capability")

    if missing_tools:
        lessons.append(f"Missing tool capabilities detected: {', '.join(missing_tools)}")

    should_replan = completion_ratio < 0.3 and has_errors
    should_evolve = (
        completion_ratio >= 0.5
        and confidence in {Confidence.HIGH, Confidence.VERY_HIGH}
        and len(completed_steps) >= 3
    )

    reflection = ReflectionResult(
        summary=f"Executed {completed_count}/{total_steps} steps for: {goal_text[:80]}",
        lessons_learned=lessons,
        confidence=confidence,
        should_evolve=should_evolve,
        should_replan=should_replan,
        memory_observations=memory_obs,
        cost_efficiency=1.0,
    )

    logger.info(
        f"Reflection: confidence={confidence.value}, "
        f"complete={completion_ratio:.0%}, "
        f"should_evolve={should_evolve}, should_replan={should_replan}"
    )

    result: dict[str, Any] = {
        "phase": Phase.VERIFY,
        "reflection": reflection,
        "confidence": confidence,
        "memory_observations": memory_obs,
    }

    if missing_tools:
        result["pending_tool_gaps"] = missing_tools

    return result


async def _llm_reflect(
    gateway: LLMGateway,
    state: AgentState,
) -> dict[str, Any] | None:
    """Attempt LLM-based reflection. Returns None on failure."""
    try:
        from src.graph.prompts import REFLECT_SYSTEM, REFLECT_USER
        from src.graph.schemas import ReflectionAnalysis
        from src.llm.structured_output import StructuredOutputManager

        goal = state.get("current_goal")
        completed_steps = state.get("completed_steps", [])
        plan_steps = state.get("plan_steps", [])
        errors = state.get("errors", [])
        tool_results = state.get("tool_results", [])

        goal_text = goal.text if goal else "Unknown goal"
        total_steps = len(plan_steps) if plan_steps else 1
        completed_count = len(completed_steps) if completed_steps else 0

        completed_summary = "\n".join(
            f"- {s.description}" for s in completed_steps[-5:]
        ) if completed_steps else "None yet"

        errors_summary = "\n".join(f"- {e[:100]}" for e in errors[-3:]) if errors else "None"

        # Include tool failure details for gap detection
        tool_errors = []
        for tr in tool_results:
            if hasattr(tr, "error") and tr.error:
                tool_errors.append(f"- {tr.tool_name}: {tr.error[:100]}")
        tool_errors_str = "\n".join(tool_errors[-3:]) if tool_errors else "None"

        user_prompt = REFLECT_USER.format(
            goal_text=goal_text,
            completed_count=completed_count,
            total_steps=total_steps,
            completed_summary=completed_summary,
            error_count=len(errors),
            errors_summary=errors_summary,
            tool_errors=tool_errors_str,
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": REFLECT_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]

        response = await gateway.acompletion(
            messages=messages,
            complexity=TaskComplexity.COMPLEX,
        )

        extractor = StructuredOutputManager()
        analysis = await extractor.extract(response.content, ReflectionAnalysis)
        if analysis is None:
            return None

        # Map LLM confidence to enum
        conf = Confidence.HIGH if analysis.confidence >= 0.7 else (
            Confidence.MEDIUM if analysis.confidence >= 0.4 else Confidence.LOW
        )

        reflection = ReflectionResult(
            summary=analysis.progress_assessment[:200],
            lessons_learned=analysis.lessons_learned,
            confidence=conf,
            should_evolve=analysis.should_evolve,
            should_replan=analysis.should_replan,
            memory_observations=analysis.memory_observations,
            cost_efficiency=1.0,
        )

        logger.info(
            f"LLM Reflection: confidence={conf.value}, "
            f"should_evolve={analysis.should_evolve}, "
            f"should_replan={analysis.should_replan}"
        )

        result: dict[str, Any] = {
            "phase": Phase.VERIFY,
            "reflection": reflection,
            "confidence": conf,
            "memory_observations": analysis.memory_observations,
        }

        # Propagate missing tool gaps identified by LLM
        if analysis.missing_tools:
            result["pending_tool_gaps"] = analysis.missing_tools

        return result
    except Exception as e:
        logger.debug(f"LLM reflection failed, using heuristics: {e}")
        return None
