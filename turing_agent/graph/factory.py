"""Factory functions for creating initial graph states."""

from __future__ import annotations

from turing_agent.graph.enums import (
    Confidence,
    GoalStatus,
    MutationType,
    Phase,
    Strategy,
)
from turing_agent.graph.models import Goal
from turing_agent.graph.state import AgentState, EvolutionState


def initial_state(
    goal_text: str,
    thread_id: str,
    max_iterations: int = 25,
) -> AgentState:
    """Create a fresh AgentState with sensible defaults.

    Args:
        goal_text: The user-provided goal description.
        thread_id: LangGraph thread ID for checkpointing.
        max_iterations: Maximum graph iterations before forced stop.

    Returns:
        AgentState: Initialized state dictionary.
    """
    return AgentState(
        phase=Phase.CLASSIFY,
        iteration_count=0,
        max_iterations=max_iterations,
        current_goal=Goal(text=goal_text, status=GoalStatus.ACTIVE),
        strategy=Strategy.DIRECT,
        plan_steps=[],
        current_step_index=0,
        messages=[],
        tools_called=[],
        tool_results=[],
        completed_steps=[],
        retrieved_memories=[],
        memory_observations=[],
        reflection=None,
        confidence=Confidence.MEDIUM,
        evolution_history=[],
        skills_learned=[],
        sub_agents=[],
        total_tokens_used=0,
        cost_records=[],
        budget_remaining=0.0,
        final_output="",
        is_complete=False,
        errors=[],
        thread_id=thread_id,
        generation=0,
    )


def initial_evolution_state(
    trigger_reason: str,
    parent_thread_id: str,
    mutation_type: MutationType,
    target_path: str | None = None,
) -> EvolutionState:
    """Create initial state for the evolution subgraph.

    Args:
        trigger_reason: What triggered evolution.
        parent_thread_id: Main agent thread for context.
        mutation_type: Type of mutation to perform.
        target_path: Optional target file/component path.

    Returns:
        EvolutionState: Initialized evolution state.
    """
    return EvolutionState(
        trigger_reason=trigger_reason,
        parent_thread_id=parent_thread_id,
        performance_metrics={},
        failure_patterns=[],
        improvement_opportunities=[],
        mutation_type=mutation_type,
        target_path=target_path,
        original_content=None,
        mutated_content=None,
        diff_content=None,
        static_analysis_passed=False,
        security_scan_passed=False,
        semantic_check_passed=False,
        sandbox_result=None,
        test_results=None,
        deployed=False,
        deployed_version_id=None,
        ab_test_id=None,
        rollback_available=False,
        status="initialized",
        errors=[],
    )


def validate_state(state: AgentState) -> list[str]:
    """Validate state invariants. Returns list of violation messages.

    An empty list means the state is valid.
    """
    violations: list[str] = []

    if not state.get("thread_id"):
        violations.append("thread_id is required")

    goal = state.get("current_goal")
    if goal and not goal.text.strip():
        violations.append("current_goal.text must not be empty")

    max_iter = state.get("max_iterations", 0)
    if max_iter <= 0:
        violations.append("max_iterations must be positive")

    iter_count = state.get("iteration_count", 0)
    if iter_count > max_iter:
        violations.append(
            f"iteration_count ({iter_count}) exceeds max_iterations ({max_iter})"
        )

    step_index = state.get("current_step_index", 0)
    plan_steps = state.get("plan_steps", [])
    if plan_steps and step_index >= len(plan_steps):
        violations.append(
            f"current_step_index ({step_index}) >= plan_steps length ({len(plan_steps)})"
        )

    return violations
