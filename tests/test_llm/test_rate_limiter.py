"""Tests for src.llm.rate_limiter — per-provider rate limiting."""

from __future__ import annotations

import pytest

from src.config.settings import Settings
from src.llm.rate_limiter import RateLimiterRegistry


@pytest.fixture
def limiter() -> RateLimiterRegistry:
    """Create a RateLimiterRegistry with default settings."""
    settings = Settings()
    return RateLimiterRegistry(settings)


class TestRateLimiterRegistry:
    """Tests for RateLimiterRegistry."""

    def test_get_limits_known_provider(self, limiter: RateLimiterRegistry) -> None:
        """Known provider returns configured limits."""
        rpm, tpm = limiter.get_limits("openai")
        assert rpm == 60
        assert tpm == 150_000

    def test_get_limits_unknown_provider(self, limiter: RateLimiterRegistry) -> None:
        """Unknown provider returns default limits."""
        rpm, tpm = limiter.get_limits("nonexistent_provider")
        assert rpm == 60
        assert tpm == 100_000

    @pytest.mark.asyncio
    async def test_acquire_does_not_block_small_request(self, limiter: RateLimiterRegistry) -> None:
        """Acquiring a single token completes immediately."""
        await limiter.acquire("test-provider", 1)
        # No assertion needed — if it returns, it succeeded

    @pytest.mark.asyncio
    async def test_acquire_creates_limiters_on_demand(self, limiter: RateLimiterRegistry) -> None:
        """First acquire for a new provider creates limiters without error."""
        await limiter.acquire("brand-new-provider", 10)
        # Verify limiters were created
        assert "brand-new-provider" in limiter._rpm_limiters
        assert "brand-new-provider" in limiter._tpm_limiters

    @pytest.mark.asyncio
    async def test_multiple_providers_independent(self, limiter: RateLimiterRegistry) -> None:
        """Two providers can acquire independently without interference."""
        await limiter.acquire("provider-a", 1)
        await limiter.acquire("provider-b", 1)
        # Both complete without error
        assert "provider-a" in limiter._rpm_limiters
        assert "provider-b" in limiter._rpm_limiters
