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
    "qwen": (60, 100_000),
    "google": (60, 120_000),
    "mistral": (60, 80_000),
    "groq": (60, 30_000),
    "zai": (60, 100_000),
    "moonshot": (60, 80_000),
    "minimax": (60, 80_000),
    "openrouter": (30, 50_000),
}

DEFAULT_RPM: int = 60
DEFAULT_TPM: int = 100_000


class RateLimiterRegistry:
    """Manages per-provider rate limiters for LLM API calls.

    Each provider gets an independent aiolimiter for RPM and TPM.
    """

    def __init__(self, settings: Settings) -> None:
        self._rpm_limiters: dict[str, aiolimiter.AsyncLimiter] = {}
        self._tpm_limiters: dict[str, aiolimiter.AsyncLimiter] = {}
        self._settings = settings

    def _get_or_create(self, provider: str) -> tuple[aiolimiter.AsyncLimiter, aiolimiter.AsyncLimiter]:
        """Get or create RPM and TPM limiters for a provider."""
        if provider not in self._rpm_limiters:
            rpm, tpm = PROVIDER_LIMITS.get(provider, (DEFAULT_RPM, DEFAULT_TPM))
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
        rpm, tpm = PROVIDER_LIMITS.get(provider, (DEFAULT_RPM, DEFAULT_TPM))
        return rpm, tpm
