"""Prometheus metrics for the Turing Agent.

Provides counters, histograms, and gauges for monitoring LLM calls,
graph execution, memory operations, and tool usage.
"""

from __future__ import annotations

from typing import Any


# Prometheus client is optional — metrics are no-ops when unavailable
try:
    from prometheus_client import Counter, Gauge, Histogram

    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False


def _counter(name: str, description: str, labels: list[str] | None = None) -> Any:
    """Create a Prometheus counter, or a no-op if prometheus_client is missing."""
    if not _PROMETHEUS_AVAILABLE:
        return None
    return Counter(name, description, labels or [])


def _histogram(name: str, description: str, labels: list[str] | None = None) -> Any:
    """Create a Prometheus histogram, or a no-op if prometheus_client is missing."""
    if not _PROMETHEUS_AVAILABLE:
        return None
    return Histogram(name, description, labels or [])


def _gauge(name: str, description: str, labels: list[str] | None = None) -> Any:
    """Create a Prometheus gauge, or a no-op if prometheus_client is missing."""
    if not _PROMETHEUS_AVAILABLE:
        return None
    return Gauge(name, description, labels or [])


# ─── LLM Metrics ────────────────────────────────────────────────────

LLM_REQUEST_DURATION = _histogram(
    "llm_request_duration_seconds",
    "Duration of LLM API requests",
    ["model", "provider"],
)

LLM_REQUEST_TOKENS = _counter(
    "llm_request_tokens_total",
    "Total tokens consumed by LLM requests",
    ["model", "provider", "token_type"],
)

LLM_REQUEST_COST = _counter(
    "llm_request_cost_usd_total",
    "Total cost of LLM requests in USD",
    ["model", "provider"],
)

LLM_REQUEST_ERRORS = _counter(
    "llm_request_errors_total",
    "Total number of LLM request errors",
    ["model", "provider", "error_type"],
)

# ─── Graph Metrics ──────────────────────────────────────────────────

GRAPH_NODE_DURATION = _histogram(
    "graph_node_duration_seconds",
    "Duration of graph node execution",
    ["node_name"],
)

GRAPH_ITERATIONS = _histogram(
    "graph_iteration_count",
    "Number of iterations per graph run",
)

GRAPH_TASKS_COMPLETED = _counter(
    "graph_tasks_completed_total",
    "Total number of completed tasks",
    ["strategy", "complexity"],
)

# ─── Memory Metrics ─────────────────────────────────────────────────

MEMORY_OPERATIONS = _counter(
    "memory_operations_total",
    "Total memory operations",
    ["operation", "tier"],
)

# ─── Tool Metrics ───────────────────────────────────────────────────

TOOL_CALLS = _counter(
    "tool_calls_total",
    "Total tool calls",
    ["tool_name"],
)

TOOL_CALL_ERRORS = _counter(
    "tool_call_errors_total",
    "Total tool call errors",
    ["tool_name"],
)

TOOL_CALL_DURATION = _histogram(
    "tool_call_duration_seconds",
    "Duration of tool calls",
    ["tool_name"],
)


def record_llm_request(
    model: str,
    provider: str,
    duration_seconds: float,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost_usd: float = 0.0,
) -> None:
    """Record metrics for an LLM request."""
    if LLM_REQUEST_DURATION:
        LLM_REQUEST_DURATION.labels(model=model, provider=provider).observe(duration_seconds)
    if LLM_REQUEST_TOKENS and input_tokens:
        LLM_REQUEST_TOKENS.labels(model=model, provider=provider, token_type="input").inc(input_tokens)
    if LLM_REQUEST_TOKENS and output_tokens:
        LLM_REQUEST_TOKENS.labels(model=model, provider=provider, token_type="output").inc(output_tokens)
    if LLM_REQUEST_COST and cost_usd > 0:
        LLM_REQUEST_COST.labels(model=model, provider=provider).inc(cost_usd)


def record_node_duration(node_name: str, duration_seconds: float) -> None:
    """Record metrics for a graph node execution."""
    if GRAPH_NODE_DURATION:
        GRAPH_NODE_DURATION.labels(node_name=node_name).observe(duration_seconds)


def record_tool_call(tool_name: str, duration_seconds: float, success: bool = True) -> None:
    """Record metrics for a tool call."""
    if TOOL_CALLS:
        TOOL_CALLS.labels(tool_name=tool_name).inc()
    if TOOL_CALL_DURATION:
        TOOL_CALL_DURATION.labels(tool_name=tool_name).observe(duration_seconds)
    if not success and TOOL_CALL_ERRORS:
        TOOL_CALL_ERRORS.labels(tool_name=tool_name).inc()


def record_memory_operation(operation: str, tier: str) -> None:
    """Record metrics for a memory operation."""
    if MEMORY_OPERATIONS:
        MEMORY_OPERATIONS.labels(operation=operation, tier=tier).inc()
