"""Tests for src.llm.rate_limiter — per-provider rate limiting."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

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


class TestProviderLimitOverrides:
    """Phase 4 G — PROVIDER_LIMITS is env-tunable via settings overrides."""

    def test_override_layers_over_curated_table(self) -> None:
        """A per-provider override replaces the curated RPM/TPM for that provider."""
        base = Settings()
        rl = base.rate_limiter.model_copy(
            update={"rate_limit_provider_overrides": {"openai": [10, 5_000]}}
        )
        settings = base.model_copy(update={"rate_limiter": rl})
        registry = RateLimiterRegistry(settings)

        rpm, tpm = registry.get_limits("openai")
        assert (rpm, tpm) == (10, 5_000)

    def test_override_leaves_other_providers_untouched(self) -> None:
        """Overriding one provider does not perturb the curated limits of others."""
        base = Settings()
        rl = base.rate_limiter.model_copy(
            update={"rate_limit_provider_overrides": {"openai": [10, 5_000]}}
        )
        settings = base.model_copy(update={"rate_limiter": rl})
        registry = RateLimiterRegistry(settings)

        rpm, tpm = registry.get_limits("anthropic")
        assert (rpm, tpm) == (50, 40_000)

    def test_no_override_uses_curated_table(self) -> None:
        """Default (empty overrides) reproduces the curated table exactly."""
        registry = RateLimiterRegistry(Settings())
        rpm, tpm = registry.get_limits("openai")
        assert (rpm, tpm) == (60, 150_000)


def _settings_with(*, cross_process: bool, max_wait: int = 5) -> Settings:
    """Settings with the cross-process knobs pinned (default keeps them on)."""
    base = Settings()
    rl = base.rate_limiter.model_copy(
        update={
            "rate_limit_cross_process_enabled": cross_process,
            "rate_limit_max_wait_attempts": max_wait,
        }
    )
    return base.model_copy(update={"rate_limiter": rl})


def _fake_redis_client(
    *, script_return: int = 1, script_raises: BaseException | None = None
) -> tuple[Any, list[dict[str, Any]]]:
    """Build a fake redis.asyncio client + record the reserve-script calls.

    ``register_script`` returns an async script callable matching the real
    ``redis.commands.core.AsyncScript.__call__(keys=, args=)`` contract. Each
    invocation is recorded so tests can assert the KEYS/ARGV contract the Lua
    script depends on.
    """
    calls: list[dict[str, Any]] = []

    async def _script(
        keys: list[str] | None = None,
        args: list[Any] | None = None,
    ) -> int:
        calls.append({"keys": list(keys or []), "args": list(args or [])})
        if script_raises is not None:
            raise script_raises
        return script_return

    client = MagicMock()
    client.register_script = MagicMock(return_value=_script)
    return client, calls


class TestAttachRedis:
    """attach_redis — opt-in registration of the shared cross-process budget."""

    def test_noop_when_disabled(self) -> None:
        """A disabled limiter never touches Redis even when given a client."""
        registry = RateLimiterRegistry(_settings_with(cross_process=False))
        client = MagicMock()

        registry.attach_redis(client)

        client.register_script.assert_not_called()
        assert registry._redis is None
        assert registry._redis_script is None

    def test_noop_when_client_none(self) -> None:
        """A None client is a no-op even when cross-process is enabled."""
        registry = RateLimiterRegistry(_settings_with(cross_process=True))

        registry.attach_redis(None)

        assert registry._redis is None

    def test_registers_script(self) -> None:
        """An enabled limiter registers the reserve Lua script once."""
        registry = RateLimiterRegistry(_settings_with(cross_process=True))
        client, _ = _fake_redis_client()

        registry.attach_redis(client)

        client.register_script.assert_called_once()
        assert registry._redis is client


class TestCrossProcessBudget:
    """acquire — two-layer (in-memory floor + shared Redis budget)."""

    @pytest.mark.asyncio
    async def test_in_memory_only_when_no_redis(self) -> None:
        """No Redis attached ⇒ acquire paces in-memory only (no script call)."""
        registry = RateLimiterRegistry(_settings_with(cross_process=True))

        await registry.acquire("zai", 1000)  # must not raise

        assert registry._redis is None  # never attached

    @pytest.mark.asyncio
    async def test_reserves_on_allow(self) -> None:
        """Script returns 1 ⇒ acquire reserves once and returns."""
        registry = RateLimiterRegistry(_settings_with(cross_process=True))
        client, calls = _fake_redis_client(script_return=1)
        registry.attach_redis(client)

        await registry.acquire("zai", 1000)

        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_keys_and_args_contract(self) -> None:
        """The script call carries the exact KEYS/ARGV the Lua depends on.

        KEYS = [turing:ratelimit:rpm:{provider}:{minute},
                turing:ratelimit:tpm:{provider}:{minute}]
        ARGV = [rpm_limit, tpm_limit, max(1, tokens), window_ttl]
        """
        registry = RateLimiterRegistry(_settings_with(cross_process=True))
        client, calls = _fake_redis_client(script_return=1)
        registry.attach_redis(client)

        await registry.acquire("zai", 7500)

        call = calls[0]
        rpm_key, tpm_key = call["keys"]
        assert rpm_key.startswith("turing:ratelimit:rpm:zai:")
        assert tpm_key.startswith("turing:ratelimit:tpm:zai:")
        # Same minute window suffix on both keys
        assert rpm_key.rsplit(":", 1)[1] == tpm_key.rsplit(":", 1)[1]
        # ARGV contract: [rpm_limit, tpm_limit, tokens(>=1), ttl]
        rpm_limit, tpm_limit, tokens, ttl = call["args"]
        assert (rpm_limit, tpm_limit) == (60, 100_000)  # curated zai limits
        assert tokens == 7500
        assert ttl == 120

    @pytest.mark.asyncio
    async def test_blocks_then_proceeds_best_effort(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Persistently-full window: bounded retries, then best-effort proceed.

        Never raises — rate limiting is observability-only; the provider's own
        429 + the retry/circuit-breaker stack is the hard backstop.
        """
        # Patch sleep so the bounded backoff doesn't burn real wall-clock.
        monkeypatch.setattr("src.llm.rate_limiter.asyncio.sleep", AsyncMock())
        registry = RateLimiterRegistry(
            _settings_with(cross_process=True, max_wait=3)
        )
        client, calls = _fake_redis_client(script_return=0)
        registry.attach_redis(client)

        await registry.acquire("zai", 1000)  # must not raise

        assert len(calls) == 3  # exactly max_wait_attempts

    @pytest.mark.asyncio
    async def test_falls_through_on_redis_error(self) -> None:
        """A Redis error on the hot path falls through immediately (no retry)."""
        registry = RateLimiterRegistry(
            _settings_with(cross_process=True, max_wait=3)
        )
        client, calls = _fake_redis_client(
            script_raises=ConnectionError("redis down")
        )
        registry.attach_redis(client)

        await registry.acquire("zai", 1000)  # must not raise

        assert len(calls) == 1  # error ⇒ immediate best-effort proceed
