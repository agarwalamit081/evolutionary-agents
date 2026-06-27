"""Shared fixtures for the Redis-Streams worker seam tests (Phase 2b).

Hermetic: each test gets a fresh in-memory fakeredis server (Streams-faithful as
of fakeredis 2.36 — consumer groups, ``XREADGROUP``, ``XAUTOCLAIM``, ``XACK``
all work), so the at-least-once / crash-recovery contracts run without a live
Redis. ``tests/test_worker/test_live.py`` separately exercises a real Redis and
skips when one is unreachable.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
import fakeredis

from src.config.settings import WorkerSettings


@pytest.fixture
def worker_settings() -> WorkerSettings:
    """Fast, hermetic worker settings — no blocking, immediate reclaim.

    ``block_ms`` low so an empty ``XREADGROUP`` returns quickly (no test hang);
    ``reclaim_min_idle_ms=0`` so ``XAUTOCLAIM`` reclaims a just-delivered entry
    (the crash-recovery test needs no wall-clock wait); ``status_ttl_s=0``
    disables the status-hash TTL so records live for assertions.
    """
    return WorkerSettings(
        runs_stream="turing:runs:test",
        group="turing-workers-test",
        consumer_name="worker-test",
        read_batch_size=5,
        block_ms=10,
        reclaim_min_idle_ms=0,
        status_ttl_s=0,
        # "Unarmed" by default (0s ⇒ no hard wall-clock cap) so the
        # timeout-precedence / stray-dead-letter tests stay hermetic to the
        # ambient .env value of WORKER_RUN_TIMEOUT_S. Tests that exercise the
        # armed path opt in via ``model_copy(update={"run_timeout_s": ...})``.
        run_timeout_s=0,
    )


@pytest_asyncio.fixture
async def fake_redis():
    """A fresh in-memory async Redis per test (fakeredis Streams support)."""
    client = fakeredis.FakeAsyncRedis()
    yield client
    await client.aclose()
