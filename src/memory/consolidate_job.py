"""Scheduled memory consolidation (Q84) — periodic decay/prune of cold episodes.

``ColdMemory.consolidate`` decays old cold-episode importance (×0.5) and
deletes those that fall below ``min_importance`` — but it was never wired to a
scheduler (``MemoryManager.consolidate`` existed but had no caller), so a
long-lived worker accumulated stale low-value episodes forever. This module
registers that pass on a cron, mirroring the checkpoint-GC / governance-prune
contract: opt-in (``MEMORY_CONSOLIDATE_ENABLED``, default off) and
observability-only — ``run()`` catches every exception (logs WARNING) and
never raises, so a DB hiccup can never abort the scheduler.

The job opens its OWN short-lived session (``get_session``) and builds a bare
``ColdMemory`` store — consolidate() touches only ``created_at`` /
``importance`` (no embedding, no generator), so no Redis/Neo4j/LLM dependency
is needed. The same pass is reachable directly via
``MemoryManager.consolidate`` (which now reads the same knobs).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from loguru import logger


class MemoryConsolidator:
    """Run the cold-memory consolidation pass (decay + prune) on a schedule.

    ``settings`` is the full :class:`~src.config.settings.Settings`; knobs are
    read from ``settings.memory`` (``consolidate_*``) and ``settings.llm``
    (``embedding_dim``). ``session_factory`` is injectable for testing.
    """

    def __init__(
        self,
        settings: Any = None,
        *,
        session_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._settings = settings
        self._session_factory = session_factory

    async def run(self) -> dict[str, Any]:
        """Decay + prune old low-importance cold episodes.

        Observability-only: never raises. Returns a report dict.
        """
        s = self._settings
        ms = getattr(s, "memory", None)
        max_age_days: int = getattr(ms, "consolidate_max_age_days", 90)
        min_importance: float = getattr(ms, "consolidate_min_importance", 0.1)
        try:
            session_factory = self._session_factory
            if session_factory is None:
                from src.db.session import get_session  # noqa: PLC0415

                session_factory = get_session

            embedding_dim = getattr(getattr(s, "llm", None), "embedding_dim", 768)
            from src.memory.cold import ColdMemory  # noqa: PLC0415

            async with session_factory() as session:
                cold = ColdMemory(
                    session=session, embedding_dim=embedding_dim, generator=None
                )
                deleted = await cold.consolidate(
                    max_age_days=max_age_days,
                    min_importance=min_importance,
                )
            logger.info(
                "Memory consolidation: cold_deleted={} (max_age_days={}, "
                "min_importance={})",
                deleted,
                max_age_days,
                min_importance,
            )
            return {
                "consolidated": True,
                "cold_deleted": deleted,
                "max_age_days": max_age_days,
                "min_importance": min_importance,
            }
        except Exception as exc:  # noqa: BLE001 — never abort the scheduler
            logger.warning("Memory consolidation failed (observability-only): {}", exc)
            return {"consolidated": False, "error": str(exc)}


def add_memory_consolidation_job(
    scheduler: Any, consolidator: MemoryConsolidator, settings_s: Any
) -> None:
    """Register the periodic ``turing-memory-consolidation`` job on ``scheduler``.

    apscheduler is imported lazily so importing this module never requires the
    dep — mirroring ``add_checkpoint_gc_job`` / ``add_governance_prune_job``.
    Fires on ``settings_s.consolidate_cron`` (default 03:00 UTC — a fresh night
    between the 02:00 battery and the 03:30 optimizer / 04:00 prune). Same
    discipline: ``max_instances=1, coalesce=True, misfire_grace_time=3600`` so a
    missed fire is coalesced, not piled up.
    """
    from apscheduler.triggers.cron import CronTrigger  # noqa: PLC0415

    async def _fire() -> None:
        await consolidator.run()

    scheduler.add_job(
        _fire,
        CronTrigger.from_crontab(
            settings_s.consolidate_cron, timezone=settings_s.consolidate_timezone
        ),
        id="turing-memory-consolidation",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3600,
    )
