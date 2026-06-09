"""LLM response data models and enums."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any


class TaskComplexity(str, Enum):
    """Task complexity levels mapped to model tiers."""

    MICRO = "micro"          # Tier 0: classification, labeling
    STANDARD = "standard"    # Tier 1: routine generation, extraction
    REASONING = "reasoning"  # Tier 2: complex reasoning, analysis
    EMBEDDING = "embedding"  # Embedding generation


@dataclass
class LLMResponse:
    """Standard response from an LLM completion call."""

    content: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    cached: bool = False
    thinking_tokens: int = 0
    finish_reason: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


@dataclass
class ToolCallResponse:
    """Response from an LLM call with tool use."""

    content: str | None
    tool_calls: list[dict[str, Any]]
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float


@dataclass
class BatchRequest:
    """A single request within a batch."""

    messages: list[dict[str, Any]]
    model: str
    temperature: float = 0.5
    max_tokens: int | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class BatchResponse:
    """Response from a batch request."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    metadata: dict[str, Any] | None = None


@dataclass
class CostRecord:
    """Record of a single LLM API call cost."""

    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: Decimal
    timestamp: datetime
    metadata: dict[str, Any] | None = None


@dataclass
class ReasoningContent:
    """Extracted reasoning/thinking content from models that support it."""

    content: str
    token_count: int
    model: str
    provider: str
