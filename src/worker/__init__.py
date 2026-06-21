"""Redis-Streams run queue + worker consumer (Phase 2b overhaul).

The seam that decouples the stateless API (which enqueues run requests) from
heavy run execution (workers consume one at a time). Public surface:

- :class:`RunJob` / :class:`RunStatus` / :class:`JobStatus` — stream + status
  models.
- :class:`RunsQueue` — producer + consumer commands over the ``turing:runs``
  stream (XADD / XREADGROUP / XACK / XAUTOCLAIM).
- :class:`RunStatusStore` — per-run status hashes for API polling.
- :class:`RunConsumer` — at-least-once drain loop with an injected executor.
- :func:`default_agent_executor` — production executor reusing ``main._run_agent``.
"""

from __future__ import annotations

from src.worker.executors import default_agent_executor
from src.worker.queue import RunsQueue, StreamEntry
from src.worker.runner import RunConsumer, RunExecutor
from src.worker.schema import JobStatus, RunJob, RunStatus
from src.worker.status import RunStatusStore

__all__ = [
    "RunsQueue",
    "StreamEntry",
    "RunStatusStore",
    "RunConsumer",
    "RunExecutor",
    "default_agent_executor",
    "RunJob",
    "RunStatus",
    "JobStatus",
]
