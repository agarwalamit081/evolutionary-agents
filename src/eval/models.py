"""Benchmark data models for performance evaluation."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class NodeMetrics(BaseModel):
    """Metrics for a single graph node execution."""

    node_name: str
    latency_ms: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0
    tool_count: int = 0
    sub_agent_count: int = 0


class BenchmarkGoal(BaseModel):
    """A benchmark scenario with expected outcomes."""

    name: str = Field(description="Unique identifier for this benchmark")
    description: str = Field(description="What this benchmark tests")
    goal_text: str = Field(description="Goal text to pass to the agent")
    category: str = Field(description="simple, complex, multi_agent, or tool_creation")
    max_iterations: int = 25
    expected_min_steps: int = 1
    expected_tools_used: int = 0
    timeout_seconds: int = 300


class BenchmarkResult(BaseModel):
    """Complete benchmark result for a single goal."""

    goal_name: str
    category: str
    success: bool
    total_latency_ms: int
    total_tokens: int
    total_cost_usd: float
    iterations: int
    node_metrics: list[NodeMetrics] = Field(default_factory=list)
    tool_results_count: int = 0
    sub_agents_spawned: int = 0
    tools_created: int = 0
    quality_score: float = 0.0  # 0.0–1.0 based on output relevance
    errors: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)
    final_output: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Correctness layer (Phase 3). ``None`` when no GoalSpec checks ran for this
    # goal; otherwise the mean of the non-skipped check scores (0.0–1.0). The
    # per-check breakdown lives in ``checks``.
    correctness_score: float | None = None
    checks: list[CheckResult] = Field(default_factory=list)


class CheckConfig(BaseModel):
    """Declaration of a single correctness check to run over a deliverable.

    ``check_type`` selects the evaluator from the registry in
    ``src/eval/checks.py`` (structural | golden | execution | oracle). ``params``
    carries type-specific arguments (e.g. required fields, row-count bounds,
    assertion lists, sandbox probe code, the LLM-judge reference).
    """

    check_type: str = Field(description="structural | golden | execution | oracle")
    name: str = Field(description="Human-readable label for this check")
    params: dict[str, Any] = Field(default_factory=dict)


class GoalSpec(BaseModel):
    """Machine-verifiable spec for a benchmark goal.

    Extends the free-form ``BenchmarkGoal`` with the deliverables the agent must
    produce and the correctness ``checks`` that gate completion. A goal is
    associated with a live run via ``eval_goal_spec_id`` in ``AgentState``; the
    verify node looks the spec up and runs its checks when ``EVAL_ENABLED``.
    """

    spec_id: str = Field(description="Stable id stored in state to look up the spec")
    name: str = Field(description="Unique identifier for this benchmark")
    description: str = ""
    goal_text: str = Field(description="Goal text to pass to the agent")
    category: str = Field(
        default="tool_creation",
        description="simple, complex, multi_agent, or tool_creation",
    )
    max_iterations: int = 25
    timeout_seconds: int = 600
    expected_deliverables: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    checks: list[CheckConfig] = Field(default_factory=list)

    def to_benchmark_goal(self) -> BenchmarkGoal:
        """Project this spec onto the legacy process-only benchmark goal."""
        return BenchmarkGoal(
            name=self.name,
            description=self.description,
            goal_text=self.goal_text,
            category=self.category,
            max_iterations=self.max_iterations,
            timeout_seconds=self.timeout_seconds,
        )


class CheckResult(BaseModel):
    """Outcome of a single correctness check."""

    check_name: str
    check_type: str
    passed: bool
    score: float = 0.0  # 0.0–1.0; fraction of the check's conditions satisfied
    evidence: dict[str, Any] = Field(default_factory=dict)
    error: str = ""
    # Skipped checks (e.g. oracle judge when deepeval/ragas are absent) carry no
    # signal: they are excluded from the mean score and never block completion.
    skipped: bool = False


class CorrectnessResult(BaseModel):
    """Aggregate correctness outcome for one goal run."""

    spec_id: str = ""
    overall_score: float = 0.0  # 0.0–1.0 mean of the non-skipped check scores
    passed: bool = False  # True only when every non-skipped check passed
    checks: list[CheckResult] = Field(default_factory=list)


# BenchmarkResult.checks/GoldSpec reference names defined later in this module
# (PEP 563 deferred annotations + grouping). Pydantic v2 leaves those models
# incomplete until an explicit rebuild once every referenced name exists at
# module scope. Harmless if already complete.
BenchmarkResult.model_rebuild()
GoalSpec.model_rebuild()
CorrectnessResult.model_rebuild()
