"""Pydantic models for the Redis-Streams run queue (Phase 2b).

``RunJob`` is the stream payload: a serializable description of one agent run
that the API enqueues and a worker executes. ``RunStatus`` is the per-run
record the worker writes to a Redis hash so the API can report progress/result
without holding the HTTP request open (the inline ``ainvoke`` it replaced).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Lifecycle of a queued run, as recorded in the status hash."""

    QUEUED = "queued"  # API enqueued; no worker has claimed it yet.
    RUNNING = "running"  # A worker claimed the job and is executing the run.
    COMPLETED = "completed"  # The run finished (is_complete may still be False).
    FAILED = "failed"  # The executor raised; left for redelivery / terminal error.
    # ── Resumable-terminal statuses (run-control hardening) ──────────────
    # Each is terminal (the entry is acked, NOT redelivered) but the per-iteration
    # AsyncPostgresSaver checkpoint is intact, so ``--resume <run_id>`` continues
    # from the last write. JobStatus lives in the Redis status hash as a string —
    # NOT a DB column — so adding members needs no Alembic migration.
    TIMEOUT = "timeout"  # Wall-clock bound exceeded (WorkerSettings.run_timeout_s / RunJob).
    BUDGET_EXHAUSTED = "budget_exhausted"  # Opt-in budget hard-stop (budget_hard_stop). Caveat: resume re-trips.
    CANCELLED = "cancelled"  # Graceful cancel via Redis flag (POST /runs/{id}/cancel).


class RunJob(BaseModel):
    """A single agent run request on the ``turing:runs`` stream.

    Mirrors the CLI flags threaded through ``src.runner.execute_run``: ``run_id`` keys
    the checkpoint thread (``api-{run_id}``) and the per-run results subdir;
    ``model`` optionally pins a model for the whole run; ``no_evolution`` skips
    the evolution phase.
    """

    run_id: str = Field(..., description="Unique run id; thread_id = api-{run_id}.")
    goal: str = Field(..., min_length=1, description="The goal to accomplish.")
    max_iterations: int | None = Field(
        default=None, description="Iteration cap; None → settings.agent.max_iterations."
    )
    no_evolution: bool = Field(default=False, description="Skip the evolution phase.")
    model: str | None = Field(
        default=None,
        description="Optional pinned model (registry key or litellm id).",
    )
    run_timeout_s: float | None = Field(
        default=None,
        description=(
            "Per-run wall-clock timeout (s). Overrides WorkerSettings.run_timeout_s; "
            "None → use the worker default (which defaults to 0 = no timeout). "
            "0 disables even if the worker default is set."
        ),
    )


class RunStatus(BaseModel):
    """Per-run status record stored at ``turing:run:{run_id}``."""

    run_id: str
    thread_id: str
    status: JobStatus = JobStatus.QUEUED
    final_output: str = ""
    is_complete: bool = False
    iteration_count: int = 0
    error: str = ""
    started_at: str = ""
    finished_at: str = ""
    # The stream entry id (XACK/XDEL handle) returned by ``RunsQueue.enqueue``.
    # Cancel records it so ``POST /runs/{id}/cancel`` can delete the pending
    # entry the instant the flag is set (P1 — otherwise ``reclaim_stale``
    # redelivers it to a peer before the in-flight worker acks, respawning the
    # run from its checkpoint). Empty for status hashes written before this
    # field existed (a load-bearing default — ``from_hash`` round-trips cleanly)
    # and for any future enqueue path that does not yet capture the id.
    entry_id: str = ""
    # The resolved per-run output folder (``<results_root>/<run_id>/``) stamped
    # at run start. Surfaces the artifact location through the API/Redis so a
    # caller discovers the deliverables' folder without guessing — the run's
    # artifacts live in a per-run subdir (RESULTS_PER_RUN_SUBDIR), not the flat
    # results/ root. Empty for status hashes written before this field existed
    # (load-bearing default → ``from_hash`` round-trips cleanly) and for runs
    # whose run_id did not resolve to a subdir.
    results_dir: str = ""

    def to_hash(self) -> dict[str, str]:
        """Flatten to a Redis hash mapping (all values str).

        ``mode="json"`` so enum members serialize to their *value* (``running``)
        — NOT the ``ClassName.MEMBER`` repr that ``str()`` yields for a
        ``(str, Enum)`` under Python 3.11+, which ``from_hash`` could not coerce
        back (a silent roundtrip break). The outer ``str()`` then stringifies
        the JSON-native bool/int values for the Redis hash.
        """
        return {
            k: str(v) if v is not None else ""
            for k, v in self.model_dump(mode="json").items()
        }

    @classmethod
    def from_hash(cls, mapping: dict[str | bytes, str | bytes]) -> RunStatus:
        """Reconstruct from a Redis ``HGETALL`` mapping (bytes-safe)."""
        clean: dict[str, Any] = {
            (k.decode() if isinstance(k, bytes) else k): (
                v.decode() if isinstance(v, bytes) else v
            )
            for k, v in mapping.items()
        }
        if not clean:
            raise KeyError("empty status hash")
        # Coerce the typed fields; leave the rest as strings.
        clean["is_complete"] = clean.get("is_complete", "False") in {
            "True",
            "true",
            "1",
        }
        try:
            clean["iteration_count"] = int(clean.get("iteration_count", "0") or 0)
        except ValueError:
            clean["iteration_count"] = 0
        return cls.model_validate(clean)


def utc_now_iso() -> str:
    """UTC now as ISO-8601 (timezone-aware). For status timestamps."""
    return datetime.now(timezone.utc).isoformat()
