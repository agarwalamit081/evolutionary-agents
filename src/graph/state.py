"""AgentState and EvolutionState TypedDicts for LangGraph."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

from src.graph.enums import (
    Confidence,
    MutationType,
    Phase,
    Strategy,
)
from src.graph.models import (
    CostRecord,
    Goal,
    PlanStep,
    ReflectionResult,
    SkillDef,
    SubAgentSpec,
    ToolResult,
)


class AgentState(TypedDict, total=False):
    """Main graph state for the task execution pipeline.

    Fields use Annotated reducers for accumulation where needed.
    Nodes return partial state dicts — never mutate in-place.
    """

    # ─── Graph Control ──────────────────────────────────────────────────
    phase: Phase
    iteration_count: int
    max_iterations: int

    # ─── Goal & Planning ────────────────────────────────────────────────
    current_goal: Goal
    strategy: Strategy
    plan_steps: list[PlanStep]
    current_step_index: int

    # ─── Execution ──────────────────────────────────────────────────────
    messages: Annotated[list[AnyMessage], add_messages]
    tools_called: Annotated[list[dict[str, Any]], operator.add]
    tool_results: Annotated[list[ToolResult], operator.add]
    completed_steps: Annotated[list[PlanStep], operator.add]

    # ─── Memory ─────────────────────────────────────────────────────────
    retrieved_memories: list[dict[str, Any]]
    memory_observations: Annotated[list[str], operator.add]

    # ─── Reflection ─────────────────────────────────────────────────────
    reflection: ReflectionResult | None
    confidence: Confidence

    # ─── Evolution ──────────────────────────────────────────────────────
    evolution_history: Annotated[list[dict[str, Any]], operator.add]
    skills_learned: Annotated[list[SkillDef], operator.add]

    # ─── Dynamic Tool Creation ──────────────────────────────────────────
    pending_tool_gaps: list[str]
    attempted_tool_gaps: Annotated[list[str], operator.add]
    tools_created: Annotated[list[dict[str, Any]], operator.add]

    # ─── Sub-Agent Delegation ──────────────────────────────────────────
    sub_agents: list[SubAgentSpec]
    pending_agent_gaps: list[str]
    sub_agents_spawned: Annotated[list[dict[str, Any]], operator.add]
    delegation_results: Annotated[list[dict[str, Any]], operator.add]

    # ─── Cost & Budget ──────────────────────────────────────────────────
    total_tokens_used: int
    cost_records: Annotated[list[CostRecord], operator.add]
    budget_remaining: float

    # ─── Output ─────────────────────────────────────────────────────────
    final_output: str
    is_complete: bool
    errors: Annotated[list[str], operator.add]

    # ─── Metadata ───────────────────────────────────────────────────────
    thread_id: str
    generation: int


class EvolutionState(TypedDict, total=False):
    """State for the evolution subgraph."""

    # ─── Trigger ────────────────────────────────────────────────────────
    trigger_reason: str
    parent_thread_id: str

    # ─── Analysis ───────────────────────────────────────────────────────
    performance_metrics: dict[str, float]
    failure_patterns: list[str]
    improvement_opportunities: list[str]

    # ─── Mutation ───────────────────────────────────────────────────────
    mutation_type: MutationType
    target_path: str | None
    original_content: str | None
    mutated_content: str | None
    diff_content: str | None

    # ─── Validation ─────────────────────────────────────────────────────
    static_analysis_passed: bool
    security_scan_passed: bool
    semantic_check_passed: bool
    sandbox_result: dict[str, Any] | None
    test_results: dict[str, Any] | None

    # ─── Deployment ─────────────────────────────────────────────────────
    deployed: bool
    deployed_version_id: str | None
    ab_test_id: str | None
    rollback_available: bool

    # ─── Status ─────────────────────────────────────────────────────────
    status: str
    errors: Annotated[list[str], operator.add]
