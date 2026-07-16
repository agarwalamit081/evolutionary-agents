"""src.memory.consolidate_job — scheduled cold-memory consolidation (Q84).

``MemoryConsolidator.run`` opens its own short-lived session, builds a bare
``ColdMemory`` store, and runs the decay/prune pass (observability-only — never
raises). ``add_memory_consolidation_job`` registers the pass on a cron read
from settings, mirroring ``add_checkpoint_gc_job`` / ``add_governance_prune_job``.

``ColdMemory`` is monkeypatched to a fake so the job's wiring is exercised
without a live DB; the fake session factory + APScheduler are likewise faked.
``MemorySettings`` knobs are pinned via init kwargs so the live .env cannot
influence the verdict.
"""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace
from typing import Any

import pytest
from apscheduler.triggers.cron import CronTrigger

from src.config.settings import MemorySettings
from src.memory.consolidate_job import (
    MemoryConsolidator,
    add_memory_consolidation_job,
)


class _FakeScheduler:
    def __init__(self) -> None:
        self.jobs: list[dict[str, Any]] = []

    def add_job(self, func: Any, trigger: Any = None, **kwargs: Any) -> None:
        self.jobs.append({"func": func, "trigger": trigger, **kwargs})


class _FakeSettings:
    """Bare settings stub: run() reads .memory (consolidate_* knobs)."""

    def __init__(self, ms: MemorySettings | None = None) -> None:
        self.memory = ms or MemorySettings(
            consolidate_max_age_days=90, consolidate_min_importance=0.1
        )
        self.llm = SimpleNamespace(embedding_dim=768)


class _FakeSessionCtx:
    def __init__(self) -> None:
        self.entered = 0

    async def __aenter__(self) -> str:
        self.entered += 1
        return "session"

    async def __aexit__(self, *_args: object) -> bool:
        return False


def _factory() -> _FakeSessionCtx:
    return _FakeSessionCtx()


class _FakeCold:
    consolidate_calls: list[tuple[int, float]] = []
    raise_on_consolidate = False

    def __init__(self, *, session: Any, generator: Any) -> None:
        _FakeCold.last_init = (session, generator)

    async def consolidate(self, *, max_age_days: int, min_importance: float) -> int:
        _FakeCold.consolidate_calls.append((max_age_days, min_importance))
        if _FakeCold.raise_on_consolidate:
            raise RuntimeError("db down")
        return 7


@pytest.fixture(autouse=True)
def _reset_fake_cold() -> Any:
    _FakeCold.consolidate_calls = []
    _FakeCold.raise_on_consolidate = False
    _FakeCold.last_init = None  # type: ignore[assignment]
    yield


class TestAddConsolidationJob:
    def test_registers_with_correct_metadata_and_cron(self) -> None:
        scheduler = _FakeScheduler()
        settings_s = MemorySettings(
            consolidate_cron="5 4 * * *", consolidate_timezone="UTC"
        )
        consolidator = MemoryConsolidator(None)  # not run, only registered

        add_memory_consolidation_job(scheduler, consolidator, settings_s)

        assert len(scheduler.jobs) == 1
        job = scheduler.jobs[0]
        assert job["id"] == "turing-memory-consolidation"
        assert job["max_instances"] == 1
        assert job["coalesce"] is True
        assert job["misfire_grace_time"] == 3600
        assert isinstance(job["trigger"], CronTrigger)

    def test_cron_read_from_settings(self) -> None:
        # The trigger must be built from settings_s.consolidate_cron, not hardcoded
        # — verify by computing the next fire time for the sentinel "5 4 * * *".
        scheduler = _FakeScheduler()
        settings_s = MemorySettings(
            consolidate_cron="5 4 * * *", consolidate_timezone="UTC"
        )
        add_memory_consolidation_job(scheduler, MemoryConsolidator(None), settings_s)
        trigger = scheduler.jobs[0]["trigger"]

        now = dt.datetime(2026, 1, 1, 0, 0, tzinfo=dt.timezone.utc)
        nxt = trigger.get_next_fire_time(None, now)
        assert nxt is not None
        assert nxt.hour == 4 and nxt.minute == 5

    @pytest.mark.asyncio
    async def test_registered_job_invokes_consolidator_run(self) -> None:
        scheduler = _FakeScheduler()
        settings_s = MemorySettings(consolidate_cron="0 3 * * *")
        consolidator = MemoryConsolidator(None)
        run_count = {"n": 0}

        async def _fake_run() -> dict[str, Any]:
            run_count["n"] += 1
            return {"consolidated": True}

        consolidator.run = _fake_run  # type: ignore[method-assign]
        add_memory_consolidation_job(scheduler, consolidator, settings_s)

        # The registered _fire wrapper awaits consolidator.run().
        await scheduler.jobs[0]["func"]()
        assert run_count["n"] == 1


class TestMemoryConsolidatorRun:
    @pytest.mark.asyncio
    async def test_run_invokes_consolidate_and_reports(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("src.memory.cold.ColdMemory", _FakeCold)
        ctx = _FakeSessionCtx()
        consolidator = MemoryConsolidator(
            _FakeSettings(), session_factory=lambda: ctx  # type: ignore[arg-type]
        )

        result = await consolidator.run()

        assert result == {
            "consolidated": True,
            "cold_deleted": 7,
            "max_age_days": 90,
            "min_importance": 0.1,
        }
        # The session was opened and consolidate ran with the configured knobs.
        assert ctx.entered == 1
        assert _FakeCold.consolidate_calls == [(90, 0.1)]
        # generator=None (consolidate needs no embeddings).
        assert _FakeCold.last_init == ("session", None)

    @pytest.mark.asyncio
    async def test_run_is_observability_only_never_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _FakeCold.raise_on_consolidate = True
        monkeypatch.setattr("src.memory.cold.ColdMemory", _FakeCold)
        consolidator = MemoryConsolidator(
            _FakeSettings(), session_factory=_factory  # type: ignore[arg-type]
        )

        # A DB hiccup is swallowed — the scheduler must never be aborted.
        result = await consolidator.run()

        assert result["consolidated"] is False
        assert "error" in result
