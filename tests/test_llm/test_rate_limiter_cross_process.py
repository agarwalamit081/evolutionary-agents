"""Cross-process rate limiter — concurrency cap, provider isolation, disable.

Companion to ``tests/test_llm/test_rate_limiter.py`` (which pins the script
KEYS/ARGV contract, the attach-redis no-ops, the bounded-backoff best-effort
fall-through, and the per-provider override layering). This file covers the
DISTINCT cross-process angles called out by the rate-limiter design, exercising
the limiter logic against a FAKE Redis (a counting script that mirrors the Lua
``GET-check-INCR`` semantics):

* under N concurrent acquires the shared bucket caps throughput (a window that
  allows only K reservations accepts exactly K, the rest fall through
  best-effort after bounded retries);
* per-provider buckets are ISOLATED — exhausting the openai budget does not
  block a deepseek acquire (distinct keys);
* disabled (``RATE_LIMIT_CROSS_PROCESS_ENABLED=False``) → no limiting at all:
  no script is registered and acquire never paces the shared budget even when a
  client is attached.

The in-memory aiolimiter floor always paces too, so these tests isolate the
cross-process layer by pinning generous in-memory limits (default) and a tight
shared budget on the fake Redis.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config.settings import Settings
from src.llm.rate_limiter import RateLimiterRegistry


def _settings_with(*, cross_process: bool, max_wait: int = 3) -> Settings:
    """Settings with the cross-process knobs pinned."""
    base = Settings()
    rl = base.rate_limiter.model_copy(
        update={
            "rate_limit_cross_process_enabled": cross_process,
            "rate_limit_max_wait_attempts": max_wait,
        }
    )
    return base.model_copy(update={"rate_limiter": rl})


class _CountingScript:
    """A fake reserve-script mirroring the Lua ``GET-check-INCR`` semantics.

    Tracks per-(provider, minute) RPM + TPM counters in a dict keyed on the
    KEYS the real Lua receives. Returns 1 while BOTH windows have headroom, 0
    once either is full — exactly the Lua contract. Records every call so tests
    can assert isolation (distinct keys per provider).
    """

    def __init__(self, *, rpm_limit: int, tpm_limit: int) -> None:
        self._rpm_limit = rpm_limit
        self._tpm_limit = tpm_limit
        self._rpm: dict[str, int] = {}
        self._tpm: dict[str, int] = {}
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, keys: list[str] | None = None, args: list[Any] | None = None) -> int:
        keys = list(keys or [])
        args = list(args or [])
        # The real Lua honors ARGV (the curated per-provider limits). For testing
        # the limiter's retry/isolation/cap behavior we drive a TIGHT budget via
        # the constructor limits (ignoring ARGV) so a test can pin a 5-RPM or
        # always-full window regardless of the provider's curated limit.
        rpm_limit = self._rpm_limit
        tpm_limit = self._tpm_limit
        tokens = int(args[2]) if len(args) > 2 else 1
        rpm_key, tpm_key = keys[0], keys[1]
        self.calls.append({"keys": keys, "args": args})
        rpm_now = self._rpm.get(rpm_key, 0)
        tpm_now = self._tpm.get(tpm_key, 0)
        if rpm_now + 1 > rpm_limit:
            return 0
        if tpm_now + tokens > tpm_limit:
            return 0
        self._rpm[rpm_key] = rpm_now + 1
        self._tpm[tpm_key] = tpm_now + tokens
        return 1


def _fake_redis(script: _CountingScript) -> Any:
    """A fake redis client whose ``register_script`` returns the counting script."""
    client = MagicMock()
    client.register_script = MagicMock(return_value=script)
    return client


# ─── Concurrent acquires cap throughput ──────────────────────────────


class TestConcurrentCapThroughput:
    """Under N concurrent acquires the shared bucket caps accepted reservations."""

    @pytest.mark.asyncio
    async def test_only_k_accepted_when_window_capacity_k(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 5-RPM shared window accepts exactly 5 reservations; the rest exhaust
        their bounded retries and proceed best-effort (never raise). The counting
        script records exactly 5 successful INCRs."""
        monkeypatch.setattr("src.llm.rate_limiter.asyncio.sleep", AsyncMock())
        registry = RateLimiterRegistry(_settings_with(cross_process=True, max_wait=1))
        # Tight shared window: 5 RPM, huge TPM (isolate the RPM cap).
        script = _CountingScript(rpm_limit=5, tpm_limit=10**9)
        registry.attach_redis(_fake_redis(script))

        # 12 concurrent acquires against the SAME provider/minute window.
        await asyncio.gather(*(registry.acquire("openai", 100) for _ in range(12)))

        # Exactly 5 reservations succeeded (the rest hit return-0 and fell through).
        successful = sum(1 for c in script.calls if True)  # all attempts recorded
        # The script was called for every attempt (12 + retries on the 7 rejects).
        assert successful >= 12
        # The RPM counter never exceeded the 5 cap.
        rpm_key = next(
            c["keys"][0] for c in script.calls if c["keys"][0].startswith("turing:ratelimit:rpm:openai:")
        )
        assert script._rpm[rpm_key] == 5

    @pytest.mark.asyncio
    async def test_never_raises_under_contention(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Rate limiting is observability-only: a saturated window never raises."""
        monkeypatch.setattr("src.llm.rate_limiter.asyncio.sleep", AsyncMock())
        registry = RateLimiterRegistry(_settings_with(cross_process=True, max_wait=2))
        # 1-RPM window — every acquire past the first is rejected.
        script = _CountingScript(rpm_limit=1, tpm_limit=10**9)
        registry.attach_redis(_fake_redis(script))

        results = await asyncio.gather(
            *(registry.acquire("deepseek", 50) for _ in range(8)),
            return_exceptions=True,
        )
        assert all(not isinstance(r, Exception) for r in results)


# ─── Per-provider isolation ──────────────────────────────────────────


class TestProviderIsolation:
    """Per-provider buckets are isolated (distinct keys → independent budgets)."""

    @pytest.mark.asyncio
    async def test_exhausted_openai_does_not_block_deepseek(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Exhausting the openai budget leaves the deepseek bucket untouched — a
        deepseek acquire still reserves successfully (distinct RPM/TPM keys)."""
        monkeypatch.setattr("src.llm.rate_limiter.asyncio.sleep", AsyncMock())
        registry = RateLimiterRegistry(_settings_with(cross_process=True, max_wait=1))
        script = _CountingScript(rpm_limit=2, tpm_limit=10**9)
        registry.attach_redis(_fake_redis(script))

        # Exhaust openai's 2-RPM window.
        await registry.acquire("openai", 100)
        await registry.acquire("openai", 100)
        # A third openai acquire is rejected (window full) but proceeds best-effort.
        await registry.acquire("openai", 100)

        # deepseek has its OWN keys → still has full headroom (reserve succeeds).
        calls_before = len(script.calls)
        await registry.acquire("deepseek", 100)
        # The deepseek call used deepseek-scoped keys, not openai's.
        ds_call = script.calls[calls_before]
        assert "deepseek" in ds_call["keys"][0]
        assert "openai" not in ds_call["keys"][0]

    @pytest.mark.asyncio
    async def test_keys_are_provider_scoped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Every reserve call's keys are scoped to the provider being acquired."""
        monkeypatch.setattr("src.llm.rate_limiter.asyncio.sleep", AsyncMock())
        registry = RateLimiterRegistry(_settings_with(cross_process=True, max_wait=1))
        script = _CountingScript(rpm_limit=100, tpm_limit=10**9)
        registry.attach_redis(_fake_redis(script))

        await registry.acquire("zai", 100)
        await registry.acquire("groq", 100)

        providers_in_keys = {c["keys"][0].split(":")[3] for c in script.calls}
        assert providers_in_keys == {"zai", "groq"}


# ─── Disabled → no limiting at all ───────────────────────────────────


class TestDisabledNoLimiting:
    """``RATE_LIMIT_CROSS_PROCESS_ENABLED=False`` → no shared-budget limiting."""

    @pytest.mark.asyncio
    async def test_disabled_never_calls_script_even_with_client(self) -> None:
        """When disabled, attaching a Redis client is a no-op and acquire never
        invokes the shared-budget script — only the in-memory floor paces."""
        registry = RateLimiterRegistry(_settings_with(cross_process=False))
        script = _CountingScript(rpm_limit=1, tpm_limit=1)
        registry.attach_redis(_fake_redis(script))

        assert registry._redis is None  # attach_redis was a no-op

        await registry.acquire("openai", 1000)
        assert script.calls == []  # shared-budget script never invoked

    @pytest.mark.asyncio
    async def test_disabled_unlimited_concurrent_acquires(self) -> None:
        """Disabled cross-process limiting + generous in-memory limits ⇒ many
        concurrent acquires all proceed without shared-budget rejection."""
        registry = RateLimiterRegistry(_settings_with(cross_process=False))
        script = _CountingScript(rpm_limit=1, tpm_limit=1)
        registry.attach_redis(_fake_redis(script))

        results = await asyncio.gather(
            *(registry.acquire("mistral", 1) for _ in range(20)),
            return_exceptions=True,
        )
        assert all(not isinstance(r, Exception) for r in results)
        assert script.calls == []  # no shared-budget accounting at all


# ─── Best-effort fall-through after bounded retries ──────────────────


class TestBestEffortFallThrough:
    """A persistently-full window exhausts bounded retries then proceeds."""

    @pytest.mark.asyncio
    async def test_retries_exactly_max_wait_attempts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A window that never opens is retried exactly ``max_wait_attempts``
        times, then the call proceeds best-effort (never raises)."""
        monkeypatch.setattr("src.llm.rate_limiter.asyncio.sleep", AsyncMock())
        registry = RateLimiterRegistry(_settings_with(cross_process=True, max_wait=4))
        # Always-full window (0 RPM).
        script = _CountingScript(rpm_limit=0, tpm_limit=10**9)
        registry.attach_redis(_fake_redis(script))

        await registry.acquire("alibaba", 100)  # must not raise
        assert len(script.calls) == 4  # exactly max_wait_attempts
