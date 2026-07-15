"""Periodic checkpoint garbage-collection (battery-04 q101 / Phase 6).

Drops langgraph checkpoint rows (``checkpoints`` / ``checkpoint_writes`` /
``checkpoint_blobs``) for runs whose NEWEST checkpoint is older than the
configured TTL. Checkpoint rows carry NO wall-clock timestamp column —
langgraph generates time-ordered UUIDv6 ``checkpoint_id``s, so the newest
``checkpoint_id`` per thread parses back to a Unix timestamp (validated
against the run-id date suffix to 0-day drift; see
``tests/test_scheduler/test_checkpoint_gc.py``). A thread whose newest
checkpoint is older than the TTL is a finished/abandoned run (runs are bounded
to minutes–hours by the cost/iteration caps), so the WHOLE thread is dropped.

Safety: opt-in (``CHECKPOINT_GC_ENABLED``, default off) AND dry-run default
(``CHECKPOINT_GC_DRY_RUN=true``): the job logs candidate threads + row counts
and deletes nothing until the owner flips ``dry_run`` false after reviewing a
log. A parse failure on any ``checkpoint_id`` is treated as "not stale" (err
toward keeping). No migration — deletes on langgraph's existing tables (there
are no FKs between them, so delete order is irrelevant).

Mirrors the governance-prune contract: ``run()`` is observability-only and
never raises — a DB hiccup is logged at WARNING and reported, so the scheduler
survives. ``add_checkpoint_gc_job`` mirrors ``add_governance_prune_job``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any
from uuid import UUID

from loguru import logger
from sqlalchemy import bindparam, text

# UUIDv6 epoch: 100-ns intervals between 1582-10-15 00:00:00 UTC and the Unix
# epoch (1970-01-01). Subtracting it from a UUIDv6 timestamp's 100-ns interval
# count yields Unix-epoch 100-ns intervals.
_UUID_EPOCH_100NS = 0x01B21DD213814000


def _uuid6_to_unix(checkpoint_id: str) -> float:
    """Parse a langgraph UUIDv6 ``checkpoint_id`` to Unix seconds.

    UUIDv6 (RFC 9562) stores the 60-bit timestamp big-endian across the first
    three fields: ``time_low`` (high 32 bits) | ``time_mid`` (mid 16 bits) |
    ``time_hi_and_version`` (low 12 bits under the version nibble). This is the
    SAME 60-bit count as UUIDv1, just reordered for lexicographic = chronological
    sort, so the standard ``UUID.fields`` breakdown extracts it directly.
    Raises ``ValueError`` for non-UUID strings; callers treat that as "not stale".
    """
    f = UUID(checkpoint_id).fields
    intervals = (f[0] << 28) | (f[1] << 12) | (f[2] & 0x0FFF)
    return (intervals - _UUID_EPOCH_100NS) / 1e7  # 100-ns → seconds


def _stale_threads(
    newest_by_thread: dict[str, str],
    *,
    ttl_days: int,
    now: Callable[[], float] = time.time,
) -> list[str]:
    """Return thread_ids whose newest ``checkpoint_id`` is older than ``ttl_days``.

    Pure + deterministic (testable without a DB): a thread is stale iff its
    newest checkpoint parses to a Unix time before ``now() - ttl_days``. Any
    thread whose ``checkpoint_id`` fails to parse is skipped (kept — safe).
    """
    cutoff = now() - ttl_days * 86_400
    stale: list[str] = []
    for thread_id, newest_id in newest_by_thread.items():
        try:
            if _uuid6_to_unix(newest_id) < cutoff:
                stale.append(thread_id)
        except (ValueError, AttributeError, OverflowError):
            # Not a parseable UUIDv6 — can't age it, so keep it.
            continue
    return stale


class CheckpointGc:
    """Drop langgraph checkpoint rows for runs whose newest checkpoint > TTL old.

    ``session_factory`` + ``now`` are injectable so the pure decision logic is
    unit-tested without a DB (``_stale_threads``); the I/O path is thin.
    """

    def __init__(
        self,
        settings: Any = None,
        *,
        session_factory: Callable[[], Any] | None = None,
        now: Callable[[], float] = time.time,
    ) -> None:
        self._settings = settings
        self._session_factory = session_factory
        self._now = now

    async def run(self) -> dict[str, Any]:
        """Scan checkpoint threads; log (dry-run) or drop (live) the stale ones.

        Observability-only: never raises. Returns a report dict.
        """
        s = self._settings
        ttl_days: int = getattr(s, "ttl_days", 7)
        dry_run: bool = bool(getattr(s, "dry_run", True))
        try:
            session_factory = self._session_factory
            if session_factory is None:
                from src.db.session import get_session  # noqa: PLC0415

                session_factory = get_session

            newest_rows = await self._newest_by_thread(session_factory)
            stale = _stale_threads(newest_rows, ttl_days=ttl_days, now=self._now)
            logger.info(
                "Checkpoint GC: scanned {} thread(s); {} stale (ttl_days={}, dry_run={})",
                len(newest_rows),
                len(stale),
                ttl_days,
                dry_run,
            )
            if not stale:
                return {"gc": True, "scanned": len(newest_rows), "stale": 0, "dry_run": dry_run}

            sample = sorted(stale)[:5]
            if dry_run:
                logger.warning(
                    "Checkpoint GC DRY RUN: would drop {} thread(s) (sample: {}). "
                    "Set CHECKPOINT_GC_DRY_RUN=false to apply.",
                    len(stale),
                    sample,
                )
                return {
                    "gc": True,
                    "scanned": len(newest_rows),
                    "stale": len(stale),
                    "dry_run": True,
                    "sample": sample,
                }

            deleted = await self._drop_threads(session_factory, stale)
            logger.info(
                "Checkpoint GC: deleted {} checkpoint(s) / {} write(s) / {} blob(s) "
                "across {} thread(s).",
                deleted["checkpoints"],
                deleted["writes"],
                deleted["blobs"],
                len(stale),
            )
            return {
                "gc": True,
                "scanned": len(newest_rows),
                "stale": len(stale),
                "dry_run": False,
                "sample": sample,
                **deleted,
            }
        except Exception as exc:  # noqa: BLE001 — never abort the scheduler
            logger.warning("Checkpoint GC failed (observability-only): {}", exc)
            return {"gc": False, "error": str(exc)}

    @staticmethod
    async def _newest_by_thread(session_factory: Callable[[], Any]) -> dict[str, str]:
        """Map each thread_id to its newest (max) checkpoint_id."""
        stmt = text("SELECT thread_id, max(checkpoint_id) FROM checkpoints GROUP BY thread_id")
        async with session_factory() as session:
            result = await session.execute(stmt)
            return {row[0]: row[1] for row in result.fetchall()}

    @staticmethod
    async def _drop_threads(
        session_factory: Callable[[], Any], stale: list[str]
    ) -> dict[str, int]:
        """Delete every stale thread's rows from all three checkpoint tables.

        The thread_id IN-list is parameterized via SQLAlchemy's expanding
        bindparam (no value interpolation). The table NAME is a fixed
        allowlist constant — SQL cannot parameterize identifiers, and these are
        not user input, so a per-table ``text()`` is the safe standard idiom.
        Langgraph's tables have no FKs between them, so delete order is
        irrelevant; all three are thread-scoped.
        """
        counts: dict[str, int] = {}
        async with session_factory() as session:
            for tbl in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
                stmt = text(f"DELETE FROM {tbl} WHERE thread_id IN :threads").bindparams(
                    bindparam("threads", expanding=True)
                )
                result = await session.execute(stmt, {"threads": stale})
                counts[tbl] = int(result.rowcount or 0)
            await session.commit()
        return {
            "writes": counts["checkpoint_writes"],
            "blobs": counts["checkpoint_blobs"],
            "checkpoints": counts["checkpoints"],
        }


def add_checkpoint_gc_job(scheduler: Any, gc: CheckpointGc, settings_s: Any) -> None:
    """Register the periodic ``turing-checkpoint-gc`` job on ``scheduler``.

    apscheduler is imported lazily so importing this module never requires the
    dep — mirroring ``add_governance_prune_job``. Fires on ``settings_s.cron``
    (default 06:00 UTC — clear of the 02:00 battery / 03:30 optimizer / 04:00
    prune / 05:00 curve-gate). Same discipline: ``max_instances=1,
    coalesce=True, misfire_grace_time=3600`` so a missed fire is coalesced, not
    piled up.
    """
    from apscheduler.triggers.cron import CronTrigger  # noqa: PLC0415

    async def _fire() -> None:
        await gc.run()

    scheduler.add_job(
        _fire,
        CronTrigger.from_crontab(settings_s.cron, timezone=settings_s.timezone),
        id="turing-checkpoint-gc",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3600,
    )
