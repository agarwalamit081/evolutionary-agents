"""Performance + correctness evaluation for the Turing Agent.

- ``BenchmarkHarness`` runs predefined goals and collects process metrics
  (latency/tokens/cost/tool usage).
- ``GoalSpec`` / ``CheckConfig`` / ``CheckResult`` / ``CorrectnessResult`` model
  the Phase-3 correctness layer; ``run_checks`` executes a spec's checks.
- ``EvalStore`` persists per-check results; the golden Battery-04 specs live in
  ``src/eval/golden.py``.
"""

from src.eval.harness import BenchmarkHarness
from src.eval.models import (
    BenchmarkGoal,
    BenchmarkResult,
    CheckConfig,
    CheckResult,
    CorrectnessResult,
    GoalSpec,
    NodeMetrics,
)

__all__ = [
    "BenchmarkHarness",
    "BenchmarkGoal",
    "BenchmarkResult",
    "CheckConfig",
    "CheckResult",
    "CorrectnessResult",
    "GoalSpec",
    "NodeMetrics",
]
