"""Per-provider rate limiting with aiolimiter token buckets."""

from __future__ import annotations


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


class RateLimiterRegistry:
    """Manages per-provider rate limiters for LLM API calls.

    Each provider gets an independent aiolimiter for RPM and TPM.
    """

    def __init__(self, settings: Settings) -> None:
        self._rpm_limiters: dict[str, aiolimiter.AsyncLimiter] = {}
        self._tpm_limiters: dict[str, aiolimiter.AsyncLimiter] = {}
        self._settings = settings
        # Effective per-provider table: the curated PROVIDER_LIMITS overlaid
        # with any env-tunable overrides (Phase 4 G). Computed once; settings are
        # fixed for the registry's lifetime.
        self._limits: dict[str, tuple[int, int]] = self._build_limits()

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

    async def acquire(self, provider: str, estimated_tokens: int = 1) -> None:
        """Acquire rate limit slot for a provider call.

        Args:
            provider: The LLM provider identifier.
            estimated_tokens: Estimated token count for the request.
        """
        rpm_limiter, tpm_limiter = self._get_or_create(provider)
        await rpm_limiter.acquire()
        await tpm_limiter.acquire(estimated_tokens)

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
