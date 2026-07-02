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

LLM_CACHE_HITS = _counter(
    "llm_cache_hits_total",
    "Prompt-cache lookups that returned a cached response",
    ["model"],
)

LLM_CACHE_MISSES = _counter(
    "llm_cache_misses_total",
    "Prompt-cache lookups that missed (no entry present)",
    ["model"],
)

# Provider-native prompt-cache accounting (e.g. Anthropic cache_control). litellm
# surfaces these on the Usage object as _cache_read_input_tokens /
# _cache_creation_input_tokens. Measuring them lets us compute cache hit-rate.
LLM_PROMPT_CACHE_READ_TOKENS = _counter(
    "llm_prompt_cache_read_tokens_total",
    "Provider-native prompt-cache tokens served from cache (read)",
    ["model", "provider"],
)

LLM_PROMPT_CACHE_CREATION_TOKENS = _counter(
    "llm_prompt_cache_creation_tokens_total",
    "Provider-native prompt-cache tokens written to cache (creation)",
    ["model", "provider"],
)

CIRCUIT_BREAKER_STATE_TRANSITIONS = _counter(
    "circuit_breaker_state_transitions_total",
    "Per-provider circuit breaker state transitions",
    ["provider", "state"],
)

LATENCY_GATE_DEMOTIONS = _counter(
    "latency_gate_demotions_total",
    "Per-provider latency-gate demotions (EWMA latency exceeded threshold)",
    ["provider"],
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

# ─── Capability Curve Metrics ───────────────────────────────────────
# The nightly battery's correctness score over time, plus the regression gate.
# CAPABILITY_CURVE_SCORE is the latest nightly battery mean (set every gate
# run); CAPABILITY_CURVE_REGRESSIONS counts detected regressions. Both are
# observability-only — the gate's detect step is read-only and never raises.

CAPABILITY_CURVE_SCORE = _gauge(
    "capability_curve_score",
    "Latest nightly battery mean correctness score",
)

CAPABILITY_CURVE_REGRESSIONS = _counter(
    "capability_curve_regressions_total",
    "Battery-curve regressions detected (floor + delta + min-points)",
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


def record_cache_lookup(model: str, hit: bool) -> None:
    """Record a prompt-cache lookup outcome (hit or miss)."""
    if hit and LLM_CACHE_HITS:
        LLM_CACHE_HITS.labels(model=model).inc()
    elif not hit and LLM_CACHE_MISSES:
        LLM_CACHE_MISSES.labels(model=model).inc()


def record_prompt_cache_tokens(
    model: str, provider: str, read_tokens: int, creation_tokens: int
) -> None:
    """Record provider-native prompt-cache token counts for one LLM response.

    ``read_tokens`` are tokens served from the provider's prompt cache; ``creation_tokens``
    are tokens written into it (a cache-write cost). Both come from litellm's Usage
    object and are absent/0 when caching is off or the provider does not report them,
    so this is a no-op for zero values.
    """
    if LLM_PROMPT_CACHE_READ_TOKENS and read_tokens:
        LLM_PROMPT_CACHE_READ_TOKENS.labels(model=model, provider=provider).inc(read_tokens)
    if LLM_PROMPT_CACHE_CREATION_TOKENS and creation_tokens:
        LLM_PROMPT_CACHE_CREATION_TOKENS.labels(
            model=model, provider=provider
        ).inc(creation_tokens)


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
