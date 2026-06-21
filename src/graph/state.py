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
    # Single-shot guard: structure_analysis runs its proactive detection at most
    # once per run, preventing re-seed loops regardless of reducer semantics.
    structure_analysis_done: bool

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
    # Skill ids recalled this run (semantic tier, findings-05 C). Populated by
    # ``retrieve_memory_node``; consumed by ``store_memory_node`` to feed the EMA
    # in ``WarmMemoryStore.update_fitness`` — the signal governance fitness-
    # retirement + recall ranking improve on. Overwrite (no reducer): the skills
    # recalled in the final plan iteration are credited for the run's outcome —
    # one success signal per run, no skill↔tool mapping needed (findings-05 D).
    recalled_skill_ids: list[str]

    # ─── Reflection ─────────────────────────────────────────────────────
    reflection: ReflectionResult | None
    confidence: Confidence

    # ─── Evolution ──────────────────────────────────────────────────────
    # CLI/API ``--no-evolution`` flag. Read by ``route_after_verify`` to
    # short-circuit the ``evolve`` node. Threaded through STATE (not
    # RunnableConfig) because LangGraph passes ``config=None`` to
    # conditional-edge routers in this graph (AsyncPostgresSaver checkpointer
    # + interrupt_before + subgraphs), so a config-based flag is silently
    # dropped — see Phase 4 live review (F4).
    no_evolution: bool
    evolution_history: Annotated[list[dict[str, Any]], operator.add]
    skills_learned: Annotated[list[SkillDef], operator.add]
    # Phase 4 E — evolve→execute edge for deployed TOOL mutations.
    # ``evolve_reexecute_offered`` is a per-cycle routing signal set True when a
    # TOOL mutation was live-registered in the ToolRegistry this cycle
    # (route_after_evolve then returns "execute" for one re-execution pass).
    # ``evolve_reexecute_done`` is the once-per-run guard: set True on the first
    # successful registration, so a second evolve cycle never re-offers (the run
    # re-executes at most once). No reducer (overwrite): the last cycle's signal
    # wins, and ``evolve_reexecute_done`` is monotonic-True (never reset).
    evolve_reexecute_offered: bool
    evolve_reexecute_done: bool

    # ─── Dynamic Tool Creation ──────────────────────────────────────────
    pending_tool_gaps: list[str]
    attempted_tool_gaps: Annotated[list[str], operator.add]
    tools_created: Annotated[list[dict[str, Any]], operator.add]

    # ─── Sub-Agent Delegation ──────────────────────────────────────────
    sub_agents: list[SubAgentSpec]
    pending_agent_gaps: list[str]
    attempted_agent_gaps: Annotated[list[str], operator.add]
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

    # ─── Memory Folding ─────────────────────────────────────────────────
    fold_history: Annotated[list[dict[str, Any]], operator.add]
    last_fold_iteration: int

    # ─── Metadata ───────────────────────────────────────────────────────
    thread_id: str
    generation: int

    # ─── Evaluation (Phase 3) ────────────────────────────────────────────
    # ``eval_goal_spec_id`` associates this run with a GoalSpec (set by the
    # ``--eval`` / ``--run-id`` CLI path). The verify node resolves it via
    # ``lookup_goal_spec`` and, when ``EVAL_ENABLED``, runs its correctness
    # checks; the aggregate score + per-check breakdown are written back here
    # for the harness to extract. Stored as plain JSON-serializable values so
    # checkpoints remain serializable (never a live Pydantic/GoalSpec object).
    eval_goal_spec_id: str
    eval_correctness_score: float
    eval_checks: list[dict[str, Any]]
    eval_correctness_passed: bool
    # F-e: set True the first time correctness checks run on an INCOMPLETE
    # verify (the rescue path), so the LLM-judge is bounded to ~one rescue call
    # per run rather than firing on every verify that finds deliverables.
    eval_rescue_attempted: bool
    # Per-run-attempt discriminator for the eval store. ``thread_id`` is stable
    # across re-runs of the same ``--run-id`` (it IS the resume key), so
    # ``eval_results`` would otherwise blend every attempt of a run under one
    # run_id. ``eval_attempt_id`` is generated once per invocation so a score
    # means ONE attempt, not a blend (store.query_latest_attempt uses it).
    eval_attempt_id: str


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
