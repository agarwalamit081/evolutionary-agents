"""battery-04 q09 regression: a benchmark / canary run is SCORE-ONLY.

When ``evolution_promote_to_live`` is on, the evolve node wires a
``PromotionGate`` whose canary (``GoldenCanary``) re-enters the task graph via
``BenchmarkHarness.run_benchmark``. Because ``run_benchmark`` compiled the FULL
graph (including ``evolve``), the canary's own runs reached ``evolve`` and fired
``run_cycle`` → ``promote()`` → another canary → … — a latent recursion. On the
q09 live run one canary spawned 3 mutation chains before terminating.

The fix: ``run_benchmark`` seeds ``no_evolution=True`` into the run state so
``route_after_verify`` skips ``evolve``. A benchmark/canary must never mutate the
system it is measuring — its own mutations would confound the candidate-prompt
score, and skipping evolve terminates the cascade at the root.

This test captures the EXACT state the harness invokes the compiled graph with
(a fake compiler) and asserts the flag is set, without a live graph run.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.eval.harness import BenchmarkHarness
from src.eval.models import BenchmarkGoal


@pytest.mark.asyncio
async def test_run_benchmark_seeds_no_evolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """The harness invokes the graph with no_evolution=True (score-only run)."""
    captured: dict[str, object] = {}

    class _FakeCompiled:
        async def ainvoke(self, state: dict[str, object]) -> dict[str, object]:
            captured["state"] = state
            return {"is_complete": True, "final_output": "ok"}

    # run_benchmark imports compile_task_graph at CALL time from this module,
    # so patching the source attribute is what the local import resolves to.
    monkeypatch.setattr("src.graph.task_graph.compile_task_graph", lambda **_kw: _FakeCompiled())

    goal = BenchmarkGoal(
        name="t", description="score-only check", goal_text="do the thing",
        category="test", max_iterations=3,
    )
    harness = BenchmarkHarness(
        gateway=MagicMock(), tools=MagicMock(), sub_agent_registry=MagicMock()
    )

    await harness.run_benchmark(goal)

    invoked = captured.get("state")
    assert isinstance(invoked, dict), "harness did not invoke the compiled graph"
    assert invoked.get("no_evolution") is True, (
        "run_benchmark must run score-only (no_evolution=True) so a canary run "
        "cannot itself evolve + promote (prevents promote→canary→evolve→promote "
        "recursion). State was: " + repr(invoked.get("no_evolution"))
    )
