"""Dependency-free per-provider circuit breaker for LLM outage protection.

Implements a CLOSED → OPEN → HALF_OPEN state machine keyed by provider string.
Only *transient* failures (rate limits, timeouts, 5xx, connection errors) trip
the breaker — auth/validation errors (401/403/400) never do, because those
indicate a configuration problem for one key, not a provider outage, and
tripping would prevent the fallback chain from trying the next provider's
working credentials.

No external dependencies: uses asyncio + time.monotonic only. State
transitions increment a Prometheus counter when the client library is
available; a missing registry never breaks the breaker.
"""

from __future__ import annotations

import asyncio
import time
from enum import Enum

from loguru import logger

from src.observability.metrics import CIRCUIT_BREAKER_STATE_TRANSITIONS


class CircuitState(Enum):
    """Breaker states for a single provider."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpenError(Exception):
    """Raised by :meth:`CircuitBreaker.before_call` when a provider's circuit is open.

    Carries the provider name so callers can log/skip to the next fallback.
    """

    def __init__(self, provider: str) -> None:
        self.provider = provider
        super().__init__(f"Circuit breaker open for provider '{provider}'")


class _ProviderState:
    """Mutable per-provider breaker bookkeeping (not thread-safe on its own;
    guarded by the breaker's asyncio.Lock)."""

    __slots__ = ("state", "failure_count", "opened_at", "half_open_in_flight")

    def __init__(self) -> None:
        self.state: CircuitState = CircuitState.CLOSED
        self.failure_count: int = 0
        self.opened_at: float = 0.0
        self.half_open_in_flight: int = 0


class CircuitBreaker:
    """Per-provider circuit breaker with CLOSED/OPEN/HALF_OPEN semantics.

    The gateway sits ABOVE the per-call tenacity retry: tenacity retries
    transient blips within a single provider attempt, while the breaker
    decides whether a provider should be attempted at all (and, when open,
    short-circuits to the next provider in the fallback chain).

    Args:
        failure_threshold: Consecutive transient failures (CLOSED) required
            to trip a provider's breaker to OPEN. Default 3.
        recovery_timeout: Seconds the breaker stays OPEN before allowing a
            single HALF_OPEN probe. Default 60.
        half_open_max_calls: Max concurrent probes permitted while HALF_OPEN.
            Default 1.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 1,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_max_calls = half_open_max_calls
        self._states: dict[str, _ProviderState] = {}
        self._lock = asyncio.Lock()
        # Indirection so tests can advance the clock deterministically.
        self._clock = time.monotonic

    def _get_state(self, provider: str) -> _ProviderState:
        state = self._states.get(provider)
        if state is None:
            state = _ProviderState()
            self._states[provider] = state
        return state

    def get_state(self, provider: str) -> CircuitState:
        """Return the current breaker state for a provider (test/observability)."""
        return self._get_state(provider).state

    def _transition(
        self, provider: str, state: _ProviderState, new_state: CircuitState
    ) -> None:
        """Apply a state change and increment the transition counter.

        Guards the Prometheus increment in try/except so a missing/failed
        registry can never break the breaker's bookkeeping.
        """
        old_state = state.state
        state.state = new_state
        if new_state == CircuitState.CLOSED:
            state.failure_count = 0
            state.opened_at = 0.0
            state.half_open_in_flight = 0
        elif new_state == CircuitState.OPEN:
            state.opened_at = self._clock()
            state.half_open_in_flight = 0
        # HALF_OPEN: failure_count preserved for re-open threshold; in_flight
        # managed by before_call/record paths.
        logger.debug(
            f"Circuit breaker '{provider}': {old_state.value} → {new_state.value}"
        )
        try:
            if CIRCUIT_BREAKER_STATE_TRANSITIONS is not None:
                CIRCUIT_BREAKER_STATE_TRANSITIONS.labels(
                    provider=provider, state=new_state.value
                ).inc()
        except Exception as exc:  # noqa: BLE001 — never let metrics break the breaker
            logger.debug(f"circuit_breaker metric increment failed: {exc}")

    async def before_call(self, provider: str) -> None:
        """Gate a provider attempt. Raises ``CircuitBreakerOpenError`` when the
        provider's circuit is OPEN and the recovery timeout has not elapsed.

        When OPEN and the recovery timeout HAS elapsed, transitions to
        HALF_OPEN and allows a single probe call through.
        """
        async with self._lock:
            state = self._get_state(provider)
            if state.state == CircuitState.CLOSED:
                return
            if state.state == CircuitState.OPEN:
                elapsed = self._clock() - state.opened_at
                if elapsed < self._recovery_timeout:
                    raise CircuitBreakerOpenError(provider)
                # Recovery elapsed: allow a single probe.
                self._transition(provider, state, CircuitState.HALF_OPEN)
            # HALF_OPEN: admit up to half_open_max_calls concurrent probes.
            if state.half_open_in_flight >= self._half_open_max_calls:
                raise CircuitBreakerOpenError(provider)
            state.half_open_in_flight += 1

    async def record_success(self, provider: str) -> None:
        """Record a successful call: resets failure count and, if HALF_OPEN,
        closes the circuit. No-op (beyond count reset) when already CLOSED.
        """
        async with self._lock:
            state = self._get_state(provider)
            if state.state == CircuitState.HALF_OPEN:
                self._transition(provider, state, CircuitState.CLOSED)
            else:
                state.failure_count = 0
                state.half_open_in_flight = 0

    async def record_failure(self, provider: str, *, transient: bool) -> None:
        """Record a failed call. Only ``transient=True`` failures count toward
        opening the circuit; auth/validation failures (transient=False) never
        trip it. On reaching the threshold (CLOSED) the breaker opens; a
        HALF_OPEN probe failure immediately re-opens it.
        """
        async with self._lock:
            state = self._get_state(provider)
            if state.state == CircuitState.HALF_OPEN:
                # Probe failed: re-open immediately regardless of count.
                self._transition(provider, state, CircuitState.OPEN)
                return
            if not transient:
                # Auth/validation error — must NOT trip the breaker.
                return
            state.failure_count += 1
            if (
                state.state == CircuitState.CLOSED
                and state.failure_count >= self._failure_threshold
            ):
                self._transition(provider, state, CircuitState.OPEN)
