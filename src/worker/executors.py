"""Production run executor for the worker (Phase 2b / Phase 3).

The default executor calls the canonical ``src.runner.execute_run`` so the worker
runs the EXACT same dependency-wired, checkpointed, cost-tracked agent path as a
host CLI run — no duplicated run logic. It passes ``origin="api"`` so the run's
checkpoint thread is ``api-{run_id}`` (distinct from a CLI run sharing the same
run_id).

``execute_run`` was extracted from ``main.py`` into ``src/runner.py`` (Phase 3,
P3a) precisely so this executor imports a plain library module instead of the
CLI entrypoint — the worker no longer pulls in the Click surface. ``src.runner``
defers its graph/DB imports, so importing it is cheap.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

from src.runner import RunProgressCallback, execute_run
from src.tools._paths import clear_flat_results_subdirs
from src.worker.schema import RunJob


async def default_agent_executor(
    job: RunJob,
    on_progress: Callable[[int], Awaitable[None]] | None = None,
) -> dict[str, Any]:
    """Run a queued job through the full agent graph; return its final state.

    ``resume=False``: each claim is a fresh execution that resumes from the last
    checkpoint via the stable ``api-{run_id}`` thread (handled inside
    ``execute_run`` when a checkpointer is wired). At-least-once redelivery thus
    resumes mid-run rather than restarting.

    ``on_progress``: forwarded to ``execute_run`` so the worker can stream the
    live iteration_count into the run-status hash mid-run (#255). ``None`` keeps
    the atomic ainvoke path.

    ``job.clear_flat_subdirs`` (battery flat-root mode): cleared HERE on the
    worker (which mounts the results volume — the scheduler does not) before the
    run starts, so a prior night's differing file set in a cross-dependent
    goal's ``results/<qNN>/`` dir can't linger into this run. Best-effort
    (never raises) and empty for every non-battery run.
    """
    _progress: RunProgressCallback | None = on_progress
    if job.clear_flat_subdirs:
        cleared = clear_flat_results_subdirs(job.clear_flat_subdirs)
        if cleared:
            logger.info(
                f"Pre-run flat clear removed {cleared} dir(s) for {job.run_id}: "
                f"{job.clear_flat_subdirs}"
            )
    return await execute_run(
        goal_text=job.goal,
        max_iterations=job.max_iterations,
        no_evolution=job.no_evolution,
        run_id=job.run_id,
        model=job.model,
        resume=False,
        origin="api",
        on_progress=_progress,
        results_per_run_subdir=job.results_per_run_subdir,
    )
