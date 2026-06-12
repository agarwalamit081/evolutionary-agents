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
