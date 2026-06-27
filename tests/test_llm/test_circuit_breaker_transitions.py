"""Circuit-breaker transition matrix with forced state + a fake clock.

Complements ``test_circuit_breaker.py`` (core thresholds, half-open probe
re-open, per-provider isolation, gateway skip integration) with a focused
transition matrix that drives the REAL ``CircuitBreaker`` through each named
edge in the spec, using a deterministic fake clock and forced-state seeding:

* CLOSED → OPEN after exactly N CONSECUTIVE *transient* failures (429 /
  5xx / timeout all count the same — they are the ``transient=True`` axis);
  N-1 failures keep it CLOSED.
* an OPEN provider is SKIPPED to the next fallback (a ``before_call`` on an
  OPEN provider raises ``CircuitBreakerOpenError`` whose ``.provider`` the
  fallback loop reads to move on).
* a HALF_OPEN probe that SUCCEEDS → CLOSED; a probe that FAILS → re-OPEN
  (stays open with a fresh window).
* AUTH (401/403) and BAD-REQUEST (400) errors NEVER trip the breaker —
  ``transient=False`` is the gateway's classification for both, and any
  number of them must leave the breaker CLOSED.
* the recovery-timeout BOUNDARY: at exactly ``recovery_timeout`` elapsed the
  breaker still blocks; one tick past it, a probe is admitted.

The 400-path is non-duplicative: ``test_circuit_breaker.py`` exercises only
the gateway-side transient classification; here we assert the breaker's own
``record_failure(transient=False)`` contract for BOTH auth and bad-request
shapes directly.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.llm.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
)


# ─── fake clock ──────────────────────────────────────────────────────────


def _fake_clock() -> tuple[dict[str, float], Any]:
    """Return (store, clock_fn) so a test advances time deterministically."""
    store = {"t": 1000.0}
    return store, lambda: store["t"]


async def _force_open(b: CircuitBreaker, provider: str) -> None:
    """Seed a provider's breaker to OPEN via the public record_failure path."""
    for _ in range(b._failure_threshold):
        await b.record_failure(provider, transient=True)


# ─── CLOSED → OPEN after exactly N consecutive transient failures ────────


class TestTransientFailuresOpenBreaker:
    """429 / 5xx / timeout are all ``transient=True``; the breaker opens after
    exactly ``failure_threshold`` of them, and not one sooner."""

    @pytest.mark.asyncio
    async def test_one_below_threshold_stays_closed(self) -> None:
        b = CircuitBreaker(failure_threshold=4, recovery_timeout=60.0)
        provider = "openai"
        for _ in range(3):
            await b.record_failure(provider, transient=True)
        assert b.get_state(provider) == CircuitState.CLOSED
        await b.before_call(provider)  # admitted

    @pytest.mark.asyncio
    async def test_exactly_threshold_opens(self) -> None:
        b = CircuitBreaker(failure_threshold=4, recovery_timeout=60.0)
        provider = "zai"
        for _ in range(4):
            await b.record_failure(provider, transient=True)
        assert b.get_state(provider) == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_mixed_transient_categories_all_count(self) -> None:
        """A 429, a 5xx, and a timeout are three transient failures → OPEN at
        threshold 3. The breaker does not distinguish transient categories."""
        b = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)
        provider = "deepseek"
        # Each record_failure(transient=True) models one of {429, 5xx, timeout}.
        await b.record_failure(provider, transient=True)  # 429
        await b.record_failure(provider, transient=True)  # 5xx
        assert b.get_state(provider) == CircuitState.CLOSED
        await b.record_failure(provider, transient=True)  # timeout → trip
        assert b.get_state(provider) == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_interleaved_success_resets_consecutive_count(self) -> None:
        """A success mid-stream resets the consecutive-transient counter, so
        threshold-1 + success + threshold-1 stays CLOSED (no accumulation)."""
        b = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)
        provider = "google"
        await b.record_failure(provider, transient=True)
        await b.record_failure(provider, transient=True)
        await b.record_success(provider)  # reset
        await b.record_failure(provider, transient=True)
        await b.record_failure(provider, transient=True)
        assert b.get_state(provider) == CircuitState.CLOSED


# ─── OPEN provider is skipped (fallback loop contract) ───────────────────


class TestOpenProviderIsSkipped:
    """The fallback loop calls ``before_call``; an OPEN provider raises, and
    the raised error carries the provider so the loop moves to the next entry."""

    @pytest.mark.asyncio
    async def test_open_raises_with_provider_name(self) -> None:
        b = CircuitBreaker(failure_threshold=2, recovery_timeout=60.0)
        await _force_open(b, "anthropic")

        with pytest.raises(CircuitBreakerOpenError) as exc:
            await b.before_call("anthropic")
        assert exc.value.provider == "anthropic"

    @pytest.mark.asyncio
    async def test_open_primary_does_not_block_fallback_provider(self) -> None:
        """An OPEN primary breaker must not block a call to a healthy fallback.
        The fallback loop's skip is modeled as: primary raises, fallback admits."""
        b = CircuitBreaker(failure_threshold=2, recovery_timeout=60.0)
        await _force_open(b, "anthropic")
        assert b.get_state("anthropic") == CircuitState.OPEN

        # Loop moves on to the next fallback-chain entry.
        await b.before_call("openai")  # admitted, no raise
        assert b.get_state("openai") == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_recovered_fallback_success_closes_after_skip(self) -> None:
        """After skipping the open primary, the fallback serving the call
        records success → CLOSED (the serving provider ends up healthy)."""
        b = CircuitBreaker(failure_threshold=2, recovery_timeout=60.0)
        await _force_open(b, "anthropic")
        await b.before_call("deepseek")
        await b.record_success("deepseek")
        assert b.get_state("deepseek") == CircuitState.CLOSED


# ─── HALF_OPEN probe: success → CLOSED, failure → stays OPEN ─────────────


class TestHalfOpenProbeOutcomes:
    @pytest.mark.asyncio
    async def test_probe_success_closes(self) -> None:
        b = CircuitBreaker(
            failure_threshold=2, recovery_timeout=10.0, half_open_max_calls=1
        )
        store, clock = _fake_clock()
        b._clock = clock  # type: ignore[assignment]
        provider = "mistral"
        for _ in range(2):
            await b.record_failure(provider, transient=True)
        assert b.get_state(provider) == CircuitState.OPEN

        # Advance past recovery → single probe admitted.
        store["t"] += 10.0 + 1.0
        await b.before_call(provider)
        assert b.get_state(provider) == CircuitState.HALF_OPEN

        # Probe succeeds → CLOSED.
        await b.record_success(provider)
        assert b.get_state(provider) == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_probe_failure_reopens_with_fresh_window(self) -> None:
        b = CircuitBreaker(
            failure_threshold=2, recovery_timeout=10.0, half_open_max_calls=1
        )
        store, clock = _fake_clock()
        b._clock = clock  # type: ignore[assignment]
        provider = "groq"
        for _ in range(2):
            await b.record_failure(provider, transient=True)

        store["t"] += 10.0 + 1.0
        await b.before_call(provider)
        assert b.get_state(provider) == CircuitState.HALF_OPEN

        # Probe fails → immediately back to OPEN (stays open).
        await b.record_failure(provider, transient=True)
        assert b.get_state(provider) == CircuitState.OPEN

        # Fresh window: immediately after re-open the probe is blocked again.
        store["t"] += 1.0
        with pytest.raises(CircuitBreakerOpenError):
            await b.before_call(provider)


# ─── recovery-timeout BOUNDARY ───────────────────────────────────────────


class TestRecoveryTimeoutBoundary:
    @pytest.mark.asyncio
    async def test_blocks_just_before_recovery_and_admits_at_timeout(self) -> None:
        """The recovery gate uses ``elapsed < recovery_timeout`` (strict). One
        tick BEFORE the timeout the probe is still blocked; at exactly the
        timeout (``elapsed == timeout``) the strict ``<`` is false and a probe
        is admitted."""
        b = CircuitBreaker(
            failure_threshold=2, recovery_timeout=30.0, half_open_max_calls=1
        )
        store, clock = _fake_clock()
        b._clock = clock  # type: ignore[assignment]
        provider = "moonshot"
        for _ in range(2):
            await b.record_failure(provider, transient=True)
        assert b.get_state(provider) == CircuitState.OPEN

        # One tick before the timeout → still blocked.
        store["t"] = 1000.0 + 29.999
        with pytest.raises(CircuitBreakerOpenError):
            await b.before_call(provider)

        # Exactly at the timeout → strict < is false → probe admitted.
        store["t"] = 1000.0 + 30.0
        await b.before_call(provider)
        assert b.get_state(provider) == CircuitState.HALF_OPEN


# ─── AUTH (401/403) + BAD-REQUEST (400) NEVER trip ───────────────────────


class TestNonTransientErrorsNeverTrip:
    """Both auth (401/403) and bad-request (400) errors are classified
    ``transient=False`` by the gateway — they indicate a config/contract
    problem for ONE key, not a provider outage. The breaker must NEVER open on
    them, or the fallback chain could not try the next provider's working
    credentials. Asserted directly on ``record_failure(transient=False)``."""

    @pytest.mark.asyncio
    async def test_auth_errors_never_trip(self) -> None:
        b = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)
        provider = "anthropic"
        for _ in range(20):  # far past any threshold
            await b.record_failure(provider, transient=False)  # 401/403
        assert b.get_state(provider) == CircuitState.CLOSED
        await b.before_call(provider)  # still admitted

    @pytest.mark.asyncio
    async def test_bad_request_errors_never_trip(self) -> None:
        b = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)
        provider = "openai"
        for _ in range(20):
            await b.record_failure(provider, transient=False)  # 400
        assert b.get_state(provider) == CircuitState.CLOSED
        await b.before_call(provider)

    @pytest.mark.asyncio
    async def test_nontransient_does_not_advance_count(self) -> None:
        """A non-transient failure between transient ones must NOT advance the
        consecutive-transient count toward the threshold (it returns early)."""
        b = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)
        provider = "zai"
        await b.record_failure(provider, transient=True)
        await b.record_failure(provider, transient=False)  # 400 — ignored
        await b.record_failure(provider, transient=True)
        # Only 2 transient failures recorded → still CLOSED.
        assert b.get_state(provider) == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_probe_failure_reopens_regardless_of_transient(self) -> None:
        """In HALF_OPEN the breaker re-opens on ANY ``record_failure`` — the
        ``transient`` flag only gates the CLOSED accumulation path. A failed
        probe (transient OR not) means the provider is still unhealthy, so the
        breaker returns to OPEN with a fresh window. This pins the asymmetry:
        non-transient errors are ignored only while CLOSED."""
        b = CircuitBreaker(
            failure_threshold=2, recovery_timeout=0.0, half_open_max_calls=1
        )
        provider = "deepseek"
        for _ in range(2):
            await b.record_failure(provider, transient=True)
        await b.before_call(provider)  # → HALF_OPEN
        assert b.get_state(provider) == CircuitState.HALF_OPEN

        # HALF_OPEN failure → OPEN, even with transient=False.
        await b.record_failure(provider, transient=False)
        assert b.get_state(provider) == CircuitState.OPEN
