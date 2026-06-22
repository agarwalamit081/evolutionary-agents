"""Nightly capability-curve battery scheduler (#197 Phase 5).

The missing scheduling layer: a sidecar (``python -m src.scheduler``) that, on a
cron schedule, enqueues every ``BATTERY04_GOALS`` spec as a ``RunJob`` into the
``turing:runs`` stream — so the worker runs the battery through the real deployed
stack and the eval layer populates ``eval_results`` autonomously (the capability
curve), instead of only when a human runs ``--eval``.
"""

from src.scheduler.battery import (
    BatteryEnqueuer,
    build_battery_jobs,
    build_run_id,
    make_battery_scheduler,
)

__all__ = [
    "BatteryEnqueuer",
    "build_battery_jobs",
    "build_run_id",
    "make_battery_scheduler",
]
