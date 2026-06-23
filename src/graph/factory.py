"""Factory functions for creating initial graph states."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from src.graph.enums import (
    Confidence,
    GoalStatus,
    MutationType,
    Phase,
    Strategy,
)
from src.graph.models import Goal
from src.graph.state import AgentState, EvolutionState


def initial_state(
    goal_text: str,
    thread_id: str,
    max_iterations: int | None = None,
    no_evolution: bool = False,
) -> AgentState:
    """Create a fresh AgentState with sensible defaults.

    Args:
        goal_text: The user-provided goal description.
        thread_id: LangGraph thread ID for checkpointing.
        max_iterations: The explicit runtime iteration cap, or ``None`` to defer
            to the classified goal complexity (``effective_max_iterations`` derives
            the tier cap at routing time). ``None`` is the desired default for the
            CLI/worker paths so a TRIVIAL goal caps early and a COMPLEX goal keeps
            headroom; an eval spec pins an int. NOT collapsed to
            ``AgentSettings.max_iterations`` here — that value is the recursion-
            limit BASIS (runner.py), not the runtime cap.
        no_evolution: Skip the ``evolve`` node. Propagated via STATE (read by
            ``route_after_verify``) rather than RunnableConfig, because LangGraph
            passes ``config=None`` to conditional-edge routers in this graph.

    Returns:
        AgentState: Initialized state dictionary.
    """
    return AgentState(
        phase=Phase.CLASSIFY,
        iteration_count=0,
        max_iterations=max_iterations,
        no_evolution=no_evolution,
        current_goal=Goal(text=goal_text, status=GoalStatus.ACTIVE),
        # Immutable objective anchor: current_goal is a mutable Goal object that
        # classify re-emits (and future nodes could touch); submitted_goal freezes
        # the literal text so recall + the OBJECTIVE can never drift (#254 backstop).
        submitted_goal=goal_text,
        strategy=Strategy.DIRECT,
        plan_steps=[],
        current_step_index=0,
        # Seed the conversation thread with the goal so execute feeds real
        # history into each LLM call and memory folding has context to compress.
        messages=[HumanMessage(content=goal_text)],
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
        pending_tool_gaps=[],
        attempted_tool_gaps=[],
        pending_agent_gaps=[],
        attempted_agent_gaps=[],
        sub_agents_spawned=[],
        delegation_results=[],
        total_tokens_used=0,
        cost_records=[],
        budget_remaining=0.0,
        final_output="",
        is_complete=False,
        # Convergence early-exit (B3): seeded empty so the first verify pass
        # always resets consecutive_stable_verifies to 0 (no prior fingerprint).
        consecutive_stable_verifies=0,
        last_verify_fingerprint=None,
        errors=[],
        fold_history=[],
        last_fold_iteration=0,
        thread_id=thread_id,
        generation=0,
        eval_rescue_attempted=False,
        eval_attempt_id="",
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

    # max_iterations may be None — "derive from goal complexity at routing
    # time" (B1). Pre-classify we cannot validate against an unknown cap, so the
    # check only fires when an explicit pin is present.
    max_iter = state.get("max_iterations")
    if max_iter is not None:
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
