"""Pydantic models for graph state structured data."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from src.graph.enums import (
    Confidence,
    GoalStatus,
    TaskComplexity,
)


def _utcnow() -> datetime:
    """Return timezone-aware UTC now."""
    return datetime.now(timezone.utc)


class Goal(BaseModel):
    """A user-provided goal with metadata."""

    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    text: str
    priority: int = Field(default=5, ge=1, le=10)
    status: GoalStatus = GoalStatus.PENDING
    complexity: TaskComplexity = TaskComplexity.SIMPLE
    parent_goal_id: str | None = None
    sub_goals: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)


class PlanStep(BaseModel):
    """A single step in an execution plan."""

    id: str = Field(default_factory=lambda: uuid4().hex[:8])
    description: str
    tool_name: str | None = None
    tool_input: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    status: GoalStatus = GoalStatus.PENDING
    result: str | None = None
    tokens_used: int = 0
    duration_ms: int = 0


class SkillDef(BaseModel):
    """Definition of a learned skill."""

    id: str
    name: str
    description: str
    skill_type: str
    code_content: str
    test_content: str | None = None
    input_schema: dict[str, Any] = Field(default_factory=dict)
    version: int = 1
    fitness_score: float = Field(default=0.5, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=_utcnow)


class SubAgentSpec(BaseModel):
    """Specification for a sub-agent delegation."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    goal: str
    model_tier: TaskComplexity = TaskComplexity.SIMPLE
    parent_thread_id: str
    shared_memory: bool = True
    max_iterations: int = 10
    status: GoalStatus = GoalStatus.PENDING
    result: str | None = None

    # ── Persistent sub-agent fields ────────────────────────────────────
    name: str = ""
    description: str = ""
    template_type: str = "fixed"  # "fixed" or "custom"
    tool_scope: str = "inherit_all"  # "inherit_all", "inherit_subset", "self_create"
    tool_subset: list[str] = Field(default_factory=list)
    budget_mode: str = "shared"  # "shared" or "separate"
    budget_limit: float = 0.0
    depth_limit: int = 0  # 0 = no recursion
    node_config: dict[str, Any] = Field(default_factory=dict)
    system_prompt_override: str | None = None
    version: int = 1
    is_active: bool = True

    # Rolling performance metrics
    total_runs: int = 0
    success_rate: float = 0.0
    avg_cost: float = 0.0
    avg_latency_ms: int = 0
    quality_score: float = Field(default=0.5, ge=0.0, le=1.0)


class ReflectionResult(BaseModel):
    """Result of self-reflection on execution."""

    summary: str
    lessons_learned: list[str] = Field(default_factory=list)
    confidence: Confidence = Confidence.MEDIUM
    should_evolve: bool = False
    should_replan: bool = False
    memory_observations: list[str] = Field(default_factory=list)
    cost_efficiency: float = Field(default=1.0, ge=0.0, le=2.0)


class ToolResult(BaseModel):
    """Result from a tool execution."""

    tool_name: str
    success: bool
    output: str
    error: str | None = None
    tokens_used: int = 0
    duration_ms: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class CostRecord(BaseModel):
    """Record of a single LLM API call cost."""

    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0
    cost_usd: float
    latency_ms: int
    timestamp: datetime = Field(default_factory=_utcnow)
