"""Request/response models for the metric-driven prompt optimizer sidecar.

These Pydantic models are the wire contract for the optimizer's HTTP API
(``POST /optimize``) and the in-process return type of ``PromptOptimizer``.
They carry NO dspy dependency so the schema can be imported anywhere (the
scheduler posts an :class:`OptimizeRequest` without importing the engine).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class OptimizerError(Exception):
    """Base error for the optimizer sidecar (mirrors the project exception rule)."""


class ConfigurationError(OptimizerError):
    """Raised for an unsupported backend or an un-shipped node profile.

    ``textgrad`` (torch) is intentionally deferred behind the ``backend`` switch,
    and only the ``classify`` profile ships in v1 — both surface as this error
    rather than a silent stub.
    """


class UsageReport(BaseModel):
    """Optimizer-side LLM usage, attributed under the optimizer run_id."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0


class OptimizeRequest(BaseModel):
    """A request to search a better prompt for a node.

    Every field is optional: ``None`` falls back to the matching
    ``OptimizerSettings`` default, so the scheduler can POST an empty body and
    the configured nightly defaults apply.
    """

    node: Optional[str] = None  # None -> settings.optimizer.target_node
    backend: Optional[
        Literal["dspy-gepa", "dspy-mipro", "dspy-copro", "textgrad"]
    ] = None  # None -> settings default
    # Advisory only; OPTIMIZER_MAX_COST_USD is the hard cap actually enforced.
    budget_hint: Optional[float] = None


class OptimizeResponse(BaseModel):
    """The outcome of one optimization attempt (promoted OR skipped)."""

    node: str
    promoted: bool
    reason: str
    baseline: Optional[float] = None
    candidate_score: Optional[float] = None
    suffixes: list[str] = Field(default_factory=list)
    usage: UsageReport = Field(default_factory=UsageReport)
