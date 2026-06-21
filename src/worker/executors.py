"""Production run executor for the worker (Phase 2b).

The default executor reuses the CLI's proven ``main._run_agent`` so the worker
runs the EXACT same dependency-wired, checkpointed, cost-tracked agent path as a
host CLI run — no duplicated run logic. It passes ``origin="api"`` so the run's
checkpoint thread is ``api-{run_id}`` (distinct from a CLI run sharing the same
run_id).

This imports ``main`` (a top-level script) rather than a library ``execute_run``
because the run core still lives in ``main.py``. Phase 3 (standalone run-executor
entrypoint) extracts ``execute_run`` into ``src/runner.py`` and this import
becomes ``from src.runner import execute_run``. The coupling is isolated to this
one function until then.
"""

from __future__ import annotations

from typing import Any

from src.worker.schema import RunJob


async def default_agent_executor(job: RunJob) -> dict[str, Any]:
    """Run a queued job through the full agent graph; return its final state.

    ``resume=False``: each claim is a fresh execution that resumes from the last
    checkpoint via the stable ``api-{run_id}`` thread (handled inside
    ``_run_agent`` when a checkpointer is wired). At-least-once redelivery thus
    resumes mid-run rather than restarting.
    """
    # Local import: main.py pulls in the full CLI/Click surface + src modules;
    # importing it at module load would force that cost on every worker import
    # (including tests that use a fake executor and never run the agent).
    from main import _run_agent

    return await _run_agent(
        goal_text=job.goal,
        max_iterations=job.max_iterations,
        no_evolution=job.no_evolution,
        run_id=job.run_id,
        model=job.model,
        resume=False,
        origin="api",
    )
