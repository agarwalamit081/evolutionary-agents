"""Dependency-free per-provider latency demotion gate.

The complement to :mod:`src.llm.circuit_breaker`: the breaker asks "is the
provider UP?" (it opens on *failures*), while this gate asks "is it FAST
ENOOUGH?" and skips a provider whose recent *successful* calls are persistently
slow — even when they return 200. This catches the failure mode the breaker
structurally cannot: a provider (often a free/tier fallback reached only after
a primary fails) that succeeds but at 200–280 s/call, silently burning
wall-clock without ever tripping an outage signal or a timeout that retries.

A provider is demoted once its exponentially-weighted moving average (EWMA) of
per-call ``latency_ms`` exceeds ``threshold_ms`` over at least ``min_samples``
calls. A demotion is temporary (``cooldown_s``): after it elapses the provider
is re-admitted for a probe, so a provider that recovers is not permanently
exiled (unlike a misconfigured static chain edit).

No external dependencies: asyncio + time.monotonic only. The Prometheus
counter increment is guarded so a missing registry can never break bookkeeping.
Default-off (``enabled=False``) — opt-in via ``LATENCY_GATE_ENABLED``, matching
the project's provider-native-capability convention.
"""

from __future__ import annotations

import asyncio
import time

from loguru import logger

from src.observability.metrics import LATENCY_GATE_DEMOTIONS


class LatencyGateOpenError(Exception):
    """Raised by :meth:`LatencyGate.before_call` when a provider is currently
    demoted for excessive latency.

    Carries the provider name so callers can log/skip to the next fallback
    (mirrors :class:`src.llm.circuit_breaker.CircuitBreakerOpenError`).
    """

    def __init__(self, provider: str) -> None:
        self.provider = provider
        super().__init__(f"Latency gate demoting provider '{provider}'")


class _ProviderLatency:
    """Mutable per-provider EWMA bookkeeping (not thread-safe on its own;
    guarded by the gate's asyncio.Lock)."""

    __slots__ = ("ewma_ms", "samples", "demoted_until")

    def __init__(self) -> None:
        self.ewma_ms: float = 0.0
        self.samples: int = 0
        self.demoted_until: float = 0.0


class LatencyGate:
    """Per-provider latency demotion gate.

    The gateway records each successful call's ``latency_ms`` (attributed to the
    provider that ultimately served the request, including any fallback/retry
    overhead) via :meth:`record_call`, and consults :meth:`before_call` in the
    fallback loop to skip a provider whose EWMA latency has exceeded the
    threshold. Demotion is cooldown-bounded and self-healing.

    Args:
        threshold_ms: EWMA latency ceiling in milliseconds. A provider whose
            rolling mean exceeds this is demoted. Default 150000 (150 s) — sits
            above the moderate-tier primaries observed in production
            (glm-4.7 ~36 s, deepseek-v4-pro ~65 s mean) but catches egregious
            slow fallbacks (e.g. glm-5-turbo ~240 s).
        min_samples: Minimum successful calls before a provider may be demoted
            (buffers against a single slow outlier). Default 3.
        cooldown_s: Seconds a demotion lasts before the provider is re-admitted
            for a probe. Default 120.
        alpha: EWMA weight on the newest sample (0 < alpha <= 1). Higher weighs
            recent calls more. Default 0.5.
        enabled: Master switch. When False, both ``before_call`` and
            ``record_call`` are no-ops (the gate self-disables rather than
            forcing every call site to branch). Default False.
    """

    def __init__(
        self,
        *,
        threshold_ms: float = 150_000.0,
        min_samples: int = 3,
        cooldown_s: float = 120.0,
        alpha: float = 0.5,
        enabled: bool = False,
    ) -> None:
        self._threshold_ms = threshold_ms
        self._min_samples = min_samples
        self._cooldown_s = cooldown_s
        self._alpha = alpha
        self._enabled = enabled
        self._states: dict[str, _ProviderLatency] = {}
        self._lock = asyncio.Lock()
        # Indirection so tests can advance the clock deterministically.
        self._clock = time.monotonic

    def _get_state(self, provider: str) -> _ProviderLatency:
        state = self._states.get(provider)
        if state is None:
            state = _ProviderLatency()
            self._states[provider] = state
        return state

    def get_demoted_until(self, provider: str) -> float:
        """Monotonic timestamp until which ``provider`` is demoted (0 if not)."""
        return self._get_state(provider).demoted_until

    async def before_call(self, provider: str) -> None:
        """Gate a provider attempt. Raises :class:`LatencyGateOpenError` when
        the provider is currently demoted and the cooldown has not elapsed.

        When demoted and the cooldown HAS elapsed, clears the demotion so the
        provider is re-admitted for a probe (the EWMA history is retained, so a
        still-slow provider re-demotes quickly). No-op when the gate is
        disabled.
        """
        if not self._enabled:
            return
        async with self._lock:
            state = self._get_state(provider)
            if state.demoted_until <= 0.0:
                return
            now = self._clock()
            if now < state.demoted_until:
                raise LatencyGateOpenError(provider)
            # Cooldown elapsed: re-admit for a probe (history retained).
            state.demoted_until = 0.0
            logger.debug(
                f"Latency gate '{provider}': cooldown elapsed, re-admitting probe"
            )

    async def record_call(self, provider: str, latency_ms: float) -> None:
        """Record a successful call's latency and demote if the rolling EWMA
        exceeds the threshold. No-op when the gate is disabled.

        Non-fatal: a non-numeric or negative ``latency_ms`` is ignored rather
        than corrupting the EWMA.
        """
        if not self._enabled:
            return
        try:
            latency = float(latency_ms)
        except (TypeError, ValueError):
            return
        if latency < 0 or latency != latency:  # NaN guard
            return
        async with self._lock:
            state = self._get_state(provider)
            if state.samples == 0:
                state.ewma_ms = latency
            else:
                state.ewma_ms = self._alpha * latency + (1.0 - self._alpha) * state.ewma_ms
            state.samples += 1
            if (
                state.demoted_until <= 0.0
                and state.samples >= self._min_samples
                and state.ewma_ms > self._threshold_ms
            ):
                state.demoted_until = self._clock() + self._cooldown_s
                logger.info(
                    f"Latency gate demoting '{provider}': EWMA "
                    f"{state.ewma_ms:.0f}ms > {self._threshold_ms:.0f}ms over "
                    f"{state.samples} samples (cooldown {self._cooldown_s:.0f}s)"
                )
                try:
                    if LATENCY_GATE_DEMOTIONS is not None:
                        LATENCY_GATE_DEMOTIONS.labels(provider=provider).inc()
                except Exception as exc:  # noqa: BLE001 — never let metrics break the gate
                    logger.debug(f"latency_gate metric increment failed: {exc}")
