"""Agent run/status routes.

``POST /run`` does NOT execute the agent inline. It enqueues a ``RunJob`` to the
``turing:runs`` Redis Stream (Phase 2b) and returns ``202 Accepted`` with a job
handle; a worker process drains the stream and runs the agent at-least-once.
This removes the horizontal-scaling blocker where one in-flight ``ainvoke`` held
an HTTP connection and the event loop. Clients poll ``GET /runs/{run_id}`` for
status/result.
"""

from __future__ import annotations

import uuid

import redis.asyncio as aioredis
from fastapi import APIRouter, HTTPException, status
from loguru import logger
from pydantic import BaseModel, Field

from src.worker.queue import RunsQueue
from src.worker.schema import JobStatus, RunJob, RunStatus
from src.worker.status import RunStatusStore

router = APIRouter()

# Mount prefix for this router (set in app.py ``include_router``). Defined here
# so ``status_url`` can reference it without the two strings drifting apart.
API_PREFIX = "/api/v1/agent"


class RunRequest(BaseModel):
    """Request to enqueue an agent run."""

    goal: str = Field(..., min_length=1)
    max_iterations: int | None = None
    no_evolution: bool = False
    run_id: str | None = Field(
        default=None,
        description="Optional run id; auto-generated (uuid4 hex) when omitted.",
    )
    model: str | None = None
    run_timeout_s: float | None = Field(
        default=None,
        description=(
            "Per-run wall-clock timeout (s); overrides WorkerSettings.run_timeout_s. "
            "None → worker default (0 = no timeout). 0 disables explicitly."
        ),
    )


class EnqueueResponse(BaseModel):
    """Handle returned when a run is enqueued (202 Accepted)."""

    run_id: str
    thread_id: str
    status: str
    status_url: str


class StatusResponse(BaseModel):
    """Polled status/result of a run."""

    run_id: str
    thread_id: str
    status: str
    final_output: str
    is_complete: bool
    iteration_count: int
    error: str
    started_at: str
    finished_at: str


class CancelResponse(BaseModel):
    """Acknowledgement of a graceful cancel request (202 Accepted).

    The cancel is cooperative: this confirms the flag was set, not that the run
    has stopped. The worker polls the flag at its per-iteration progress callback
    (~1-iteration latency) and transitions the run to ``CANCELLED`` (the run-level
    timeout is the hard bound if the worker is mid-iteration).
    """

    run_id: str
    status: str


def _client_and_queue() -> tuple[RunsQueue, RunStatusStore]:
    """Build a per-request queue + status store over a fresh Redis client.

    Uses the configured ``redis_url``; raises ``HTTPException(503)`` if Redis is
    unreachable so the caller never sees a stack trace (security rule).
    """
    from src.config import get_settings

    settings = get_settings()
    try:
        redis_client = aioredis.from_url(settings.redis.redis_url)
    except Exception as exc:
        logger.warning(f"Redis client build failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Run queue unavailable",
        ) from exc
    return (
        RunsQueue(redis_client, settings.worker),
        RunStatusStore(redis_client, settings.worker),
    )


@router.post(
    "/run",
    response_model=EnqueueResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def enqueue_run(request: RunRequest) -> EnqueueResponse:
    """Enqueue an agent run; a worker executes it asynchronously."""
    queue, status_store = _client_and_queue()

    run_id = request.run_id or uuid.uuid4().hex
    thread_id = f"api-{run_id}"

    # P1 dedup — a run_id IS the run's identity (thread_id = api-{run_id}, the
    # checkpoint-resume key). A second entry for an already-QUEUED/RUNNING run
    # would resume the SAME checkpoint (double-spend until the lease lock
    # serializes them) — exactly the "respawn" shape. Refuse it (409) so a
    # repeated POST /run can't mint a duplicate; terminal statuses leave the
    # door open for a deliberate fresh re-enqueue (resume-by-run_id intact).
    existing = await status_store.get(run_id)
    if existing is not None and existing.status in (
        JobStatus.QUEUED,
        JobStatus.RUNNING,
    ):
        logger.info(
            f"API: reject duplicate enqueue for in-flight run {run_id} "
            f"({existing.status.value})"
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"run_id {run_id} already {existing.status.value}",
        )

    job = RunJob(
        run_id=run_id,
        goal=request.goal,
        max_iterations=request.max_iterations,
        no_evolution=request.no_evolution,
        model=request.model,
        run_timeout_s=request.run_timeout_s,
    )

    try:
        await queue.ensure_group()
        entry_id = await queue.enqueue(job)
        await status_store.mark(
            run_id, thread_id, JobStatus.QUEUED, entry_id=entry_id
        )
    except Exception as exc:
        logger.warning(f"Enqueue failed for run {run_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Run queue unavailable",
        ) from exc

    logger.info(f"API: enqueued run {run_id} for goal: {request.goal[:80]}")
    return EnqueueResponse(
        run_id=run_id,
        thread_id=thread_id,
        status=JobStatus.QUEUED.value,
        status_url=f"{API_PREFIX}/runs/{run_id}",
    )


@router.get("/runs/{run_id}", response_model=StatusResponse)
async def get_run_status(run_id: str) -> StatusResponse:
    """Poll a run's status/result. ``404`` when unknown or the status expired."""
    _, status_store = _client_and_queue()
    record: RunStatus | None = await status_store.get(run_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown or expired run_id: {run_id}",
        )
    return _to_response(record)


@router.post(
    "/runs/{run_id}/cancel",
    response_model=CancelResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def cancel_run(run_id: str) -> CancelResponse:
    """Request a graceful cancel of an in-flight run.

    Sets a Redis flag the worker polls at its per-iteration progress callback →
    ``RunCancelled`` → terminal ``CANCELLED`` + acked (NOT redelivered). Cancel
    is cooperative (~1-iteration latency); the run-level wall-clock timeout is
    the hard bound. ``404`` when no status record exists for ``run_id``
    (unknown/expired). Idempotent: a repeat POST is a no-op (the flag's mere
    presence is the signal), so cancelling an already-terminal run is harmless.

    P1 — terminal NOW, not just cooperative: in addition to the flag, the
    pending stream entry is ``XACK``+``XDEL``'d on the spot, so ``reclaim_stale``
    (XAUTOCLAIM) can never hand it to a peer worker that would resume the run
    from its last checkpoint and keep spending while the in-flight owner winds
    down. If no ``entry_id`` was recorded (status hash predates the field, or a
    future enqueue path didn't capture it) the flag + worker ack + run timeout
    remain the backstop. ``404`` (not a silent no-op) only when no status record
    exists at all.
    """
    queue, status_store = _client_and_queue()
    record: RunStatus | None = await status_store.get(run_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown or expired run_id: {run_id}",
        )
    await status_store.request_cancel(run_id)
    if record.entry_id:
        deleted = await queue.delete_entry(record.entry_id)
        logger.info(
            f"API: cancel requested for run {run_id} "
            f"(stream entry {record.entry_id} deleted={deleted})"
        )
    else:
        logger.info(
            f"API: cancel requested for run {run_id} "
            f"(no entry_id on record — flag-only)"
        )
    return CancelResponse(run_id=run_id, status="cancel_requested")


def _to_response(record: RunStatus) -> StatusResponse:
    return StatusResponse(
        run_id=record.run_id,
        thread_id=record.thread_id,
        status=record.status.value,
        final_output=record.final_output,
        is_complete=record.is_complete,
        iteration_count=record.iteration_count,
        error=record.error,
        started_at=record.started_at,
        finished_at=record.finished_at,
    )
