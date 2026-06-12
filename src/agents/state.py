"""Isolated state TypedDict for sub-agent execution.

Mirrors the relevant fields from AgentState but is self-contained.
Sub-agent nodes access fields by key (TypedDict), so the same node
functions from src/graph/nodes/ work with this state as long as the
field names match.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

from src.graph.enums import Confidence, Phase, Strategy
from src.graph.models import CostRecord, Goal, GoalStatus, PlanStep, ReflectionResult, ToolResult


class SubAgentState(TypedDict, total=False):
    """Isolated state for a sub-agent execution.

    Sub-agents run as independent LangGraph subgraphs with their own
    state, budget, and tool set. This state mirrors the relevant
    AgentState fields so that existing node functions can be reused.
    """

    # ── Graph Control ──────────────────────────────────────────────────
    phase: Phase
    iteration_count: int
    max_iterations: int

    # ── Goal & Planning ────────────────────────────────────────────────
    goal_text: str
    current_goal: Goal  # Matches AgentState field used by all shared node functions
    strategy: Strategy
    plan_steps: list[PlanStep]
    current_step_index: int

    # ── Execution ──────────────────────────────────────────────────────
    messages: Annotated[list[AnyMessage], add_messages]
    tools_called: Annotated[list[dict[str, Any]], operator.add]
    tool_results: Annotated[list[ToolResult], operator.add]
    completed_steps: Annotated[list[PlanStep], operator.add]

    # ── Memory (isolated) ──────────────────────────────────────────────
    retrieved_memories: list[dict[str, Any]]
    memory_observations: Annotated[list[str], operator.add]

    # ── Reflection ─────────────────────────────────────────────────────
    reflection: ReflectionResult | None
    confidence: Confidence

    # ── Dynamic Tool Creation ──────────────────────────────────────────
    pending_tool_gaps: Annotated[list[str], operator.add]
    tools_created: Annotated[list[dict[str, Any]], operator.add]

    # ── Cost & Budget ──────────────────────────────────────────────────
    total_tokens_used: int
    cost_records: Annotated[list[CostRecord], operator.add]
    budget_remaining: float

    # ── Output ─────────────────────────────────────────────────────────
    final_output: str
    is_complete: bool
    errors: Annotated[list[str], operator.add]

    # ── Metadata ───────────────────────────────────────────────────────
    parent_thread_id: str
    depth: int


def initial_sub_agent_state(
    goal_text: str,
    parent_thread_id: str,
    max_iterations: int = 10,
    depth: int = 0,
) -> SubAgentState:
    """Create a fresh SubAgentState with sensible defaults.

    Args:
        goal_text: The subtask goal for this sub-agent.
        parent_thread_id: Parent's thread ID for tracking.
        max_iterations: Maximum iterations for the sub-agent.
        depth: Current nesting depth.

    Returns:
        SubAgentState: Initialized state dictionary.
    """
    return SubAgentState(
        phase=Phase.CLASSIFY,
        iteration_count=0,
        max_iterations=max_iterations,
        goal_text=goal_text,
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
        pending_tool_gaps=[],
        tools_created=[],
        total_tokens_used=0,
        cost_records=[],
        budget_remaining=0.0,
        final_output="",
        is_complete=False,
        errors=[],
        parent_thread_id=parent_thread_id,
        depth=depth,
    )
