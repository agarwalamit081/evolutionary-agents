"""Metric-driven prompt optimizer sidecar (Phase 2 C2: DSPy + GEPA).

A default-off, C1-gated, cost-bounded sidecar that turns the existing
:class:`~src.evolution.promote.GoldenCanary` correctness score into an
objective and searches a better prompt for a node via DSPy/GEPA. The optimized
instruction is validated against the real canary (full agent runs) and, on
beating the baseline, pushed through the EXISTING
:class:`~src.evolution.promote.PromotionGate` (canary-gated, auto-rollback).

This package is imported ONLY inside the optimizer container + tests; the
scheduler drives it over HTTP, so api/worker never import it. The dspy import is
kept inside :mod:`src.optimizer.engine` (imported lazily at compile time), so
importing this top-level package does not require dspy.
"""

from __future__ import annotations

from src.optimizer.models import (
    ConfigurationError,
    OptimizeRequest,
    OptimizeResponse,
    OptimizerError,
    UsageReport,
)

__all__ = [
    "ConfigurationError",
    "OptimizeRequest",
    "OptimizeResponse",
    "OptimizerError",
    "UsageReport",
]
