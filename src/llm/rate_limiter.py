"""Per-provider rate limiting with two layers.

1. **In-memory aiolimiter** (per-gateway-instance) — single-process fairness
   among the coroutines in one process, and the fallback gate when no shared
   Redis budget is available.
2. **Optional Redis cross-process token budget** — coordinates the WHOLE fleet
   (e.g. the 2 worker processes) against ONE shared provider quota, so the
   fleet cannot collectively exceed a provider's RPM/TPM. Fixed-window counters
   keyed by provider+minute, reserved atomically via a Lua script, with bounded
   backoff. Best-effort: a Redis hiccup or a persistently-full window falls
   through (rate limiting is observability-only — the provider's own 429 + the
   retry/circuit-breaker stack is the hard backstop). Gated behind
   ``RATE_LIMIT_CROSS_PROCESS_ENABLED``; degrades to in-memory-only when off or
   when no Redis client is attached.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import aiolimiter
from loguru import logger

from src.config.settings import Settings


# Default rate limits per provider: (requests_per_minute, tokens_per_minute)
PROVIDER_LIMITS: dict[str, tuple[int, int]] = {
    "anthropic": (50, 40_000),
    "openai": (60, 150_000),
    "deepseek": (60, 200_000),
    "alibaba": (60, 100_000),
    "google": (60, 120_000),
    "mistral": (60, 80_000),
    "groq": (60, 30_000),
    "zai": (60, 100_000),
    "moonshot": (60, 80_000),
    "minimax": (60, 80_000),
    "openrouter": (30, 50_000),
    "nvidia": (40, 60_000),
}


# Lua: atomically reserve 1 RPM + ``tokens`` TPM against two minute-scoped keys.
# Reserve ONLY if BOTH windows have headroom (the GET-check-INCR is atomic because
# a Lua script runs without interleaving on Redis's single thread, so two workers
# can't both read 59 and both write 61). KEYS[1]=rpm key, KEYS[2]=tpm key;
# ARGV[1]=rpm limit, ARGV[2]=tpm limit, ARGV[3]=tokens, ARGV[4]=window ttl seconds.
# Returns 1 if reserved, 0 if a window is full.
_RATE_RESERVE_LUA = """
local rpm_count = tonumber(redis.call('GET', KEYS[1]) or '0')
local tpm_count = tonumber(redis.call('GET', KEYS[2]) or '0')
if rpm_count + 1 > tonumber(ARGV[1]) then
    return 0
end
if tpm_count + tonumber(ARGV[3]) > tonumber(ARGV[2]) then
    return 0
end
redis.call('INCR', KEYS[1])
redis.call('INCRBY', KEYS[2], ARGV[3])
redis.call('EXPIRE', KEYS[1], ARGV[4], 'NX')
redis.call('EXPIRE', KEYS[2], ARGV[4], 'NX')
return 1
"""

# Window TTL: a little past 60s so a minute-scoped key is reaped even with clock
# skew and never lingers more than ~2 minutes. ``EXPIRE ... NX`` sets it once.
_WINDOW_TTL_SECONDS = 120


class RateLimiterRegistry:
    """Manages per-provider rate limiters for LLM API calls.

    Each provider gets an independent in-memory aiolimiter for RPM and TPM, and
    — when a shared Redis client is attached and cross-process limiting is
    enabled — a shared Redis token budget coordinating the whole fleet against
    one provider quota.
    """

    def __init__(self, settings: Settings) -> None:
        self._rpm_limiters: dict[str, aiolimiter.AsyncLimiter] = {}
        self._tpm_limiters: dict[str, aiolimiter.AsyncLimiter] = {}
        self._settings = settings
        # Effective per-provider table: the curated PROVIDER_LIMITS overlaid
        # with any env-tunable overrides (Phase 4 G). Computed once; settings are
        # fixed for the registry's lifetime.
        self._limits: dict[str, tuple[int, int]] = self._build_limits()
        # Cross-process coordination (optional). The runtime attaches a shared
        # Redis client via attach_redis() once one is available; None ⇒ the
        # in-memory aiolimiter is the only gate (single-process fallback).
        rate_cfg = settings.rate_limiter
        self._cross_process_enabled: bool = bool(
            rate_cfg.rate_limit_cross_process_enabled
        )
        self._max_wait_attempts: int = max(
            1, int(rate_cfg.rate_limit_max_wait_attempts)
        )
        self._redis: Any = None
        self._redis_script: Any = None

    def attach_redis(self, client: Any) -> None:
        """Attach a shared Redis client for cross-process rate coordination.

        Registers the reserve Lua script once (EVALSHA + EVAL fallback cached on
        the client). Idempotent + best-effort: if registration fails the limiter
        stays in in-memory-only mode (logged at debug). No-op when cross-process
        limiting is disabled or ``client`` is None.
        """
        if not self._cross_process_enabled or client is None:
            return
        try:
            self._redis_script = client.register_script(_RATE_RESERVE_LUA)
            self._redis = client
            logger.debug("Redis cross-process rate limiter attached")
        except Exception as e:
            logger.debug(f"Redis rate limiter attach failed: {e}")
            self._redis = None
            self._redis_script = None

    def _build_limits(self) -> dict[str, tuple[int, int]]:
        """Merge the curated PROVIDER_LIMITS with env overrides."""
        limits: dict[str, tuple[int, int]] = dict(PROVIDER_LIMITS)
        overrides = self._settings.rate_limiter.rate_limit_provider_overrides or {}
        for provider, pair in overrides.items():
            try:
                rpm, tpm = pair
                limits[provider] = (int(rpm), int(tpm))
            except (TypeError, ValueError) as e:
                logger.debug(
                    f"Ignoring malformed rate-limit override for {provider}: {e}"
                )
        return limits

    def _get_or_create(self, provider: str) -> tuple[aiolimiter.AsyncLimiter, aiolimiter.AsyncLimiter]:
        """Get or create RPM and TPM limiters for a provider."""
        if provider not in self._rpm_limiters:
            rpm, tpm = self._limits.get(
                provider,
                (
                    self._settings.rate_limiter.rate_limit_default_rpm,
                    self._settings.rate_limiter.rate_limit_default_tpm,
                ),
            )
            # aiolimiter uses max_rate/time_period; we want per-minute
            self._rpm_limiters[provider] = aiolimiter.AsyncLimiter(max_rate=rpm, time_period=60)
            self._tpm_limiters[provider] = aiolimiter.AsyncLimiter(max_rate=tpm, time_period=60)
            logger.debug(f"Rate limiter created for {provider}: {rpm} RPM, {tpm} TPM")
        return self._rpm_limiters[provider], self._tpm_limiters[provider]

    async def _acquire_in_memory(self, provider: str, estimated_tokens: int) -> None:
        """Reserve a slot from the per-process aiolimiter (single-process gate)."""
        rpm_limiter, tpm_limiter = self._get_or_create(provider)
        await rpm_limiter.acquire()
        await tpm_limiter.acquire(estimated_tokens)

    async def _acquire_cross_process(self, provider: str, estimated_tokens: int) -> None:
        """Reserve against the shared Redis budget with bounded backoff.

        Loops up to ``_max_wait_attempts``: if a minute window is full, sleeps a
        small exponential backoff (yielding the event loop so in-flight calls in
        this and sibling processes can drain) and retries. After exhaustion the
        call proceeds best-effort — rate limiting never blocks a call
        indefinitely; the provider's own 429 + the retry/circuit-breaker stack
        is the hard backstop. Any Redis error on the hot path also falls through
        (logged at debug, never raised).
        """
        rpm, tpm = self.get_limits(provider)
        tokens = max(1, int(estimated_tokens))
        backoff = 0.1
        for _ in range(self._max_wait_attempts):
            minute = int(time.time()) // 60
            rpm_key = f"turing:ratelimit:rpm:{provider}:{minute}"
            tpm_key = f"turing:ratelimit:tpm:{provider}:{minute}"
            try:
                allowed = await self._redis_script(
                    keys=[rpm_key, tpm_key],
                    args=[rpm, tpm, tokens, _WINDOW_TTL_SECONDS],
                )
            except Exception as e:
                logger.debug(f"Redis rate reserve failed for {provider}: {e}")
                return
            if int(allowed) == 1:
                return
            # Window full: bounded backoff then retry. Yields the event loop so
            # in-flight calls (here + sibling processes) can drain.
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 2.0)
        logger.debug(
            f"Rate budget for {provider} still full after "
            f"{self._max_wait_attempts} waits; proceeding best-effort"
        )

    async def acquire(self, provider: str, estimated_tokens: int = 1) -> None:
        """Acquire a rate-limit slot for a provider call.

        Always reserves the in-memory aiolimiter (per-process fairness + the
        Redis-absent fallback). When a shared Redis budget is attached and
        cross-process limiting is enabled, ALSO reserves against the fleet-wide
        quota so the worker fleet can't collectively exceed a provider's RPM/TPM
        (e.g. 2 workers each doing 60 RPM against a 60-RPM provider).

        Args:
            provider: The LLM provider identifier.
            estimated_tokens: Estimated token count for the request (prompt +
                reserved output). Reserved against both the in-memory TPM window
                and the shared Redis TPM window.
        """
        await self._acquire_in_memory(provider, estimated_tokens)
        if self._cross_process_enabled and self._redis is not None:
            await self._acquire_cross_process(provider, estimated_tokens)

    def get_limits(self, provider: str) -> tuple[int, int]:
        """Return (RPM, TPM) for a provider."""
        rpm, tpm = self._limits.get(
            provider,
            (
                self._settings.rate_limiter.rate_limit_default_rpm,
                self._settings.rate_limiter.rate_limit_default_tpm,
            ),
        )
        return rpm, tpm
