"""Tests for src.config.settings module."""

from __future__ import annotations

import pytest

from src.config.settings import (
    BudgetSettings,
    RedisSettings,
    Settings,
    get_settings,
)


# ─── get_settings Tests ──────────────────────────────────────────────


class TestGetSettings:
    """Tests for the get_settings singleton."""

    def test_get_settings_returns_settings(self) -> None:
        """get_settings returns a Settings instance."""
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_get_settings_is_singleton(self) -> None:
        """Repeated calls return the same cached instance."""
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2


# ─── Database Settings Tests ─────────────────────────────────────────


class TestDatabaseSettings:
    """Tests for DatabaseSettings defaults and validation."""

    def test_database_url_uses_asyncpg(self) -> None:
        """Default database_url contains the asyncpg driver."""
        settings = Settings()
        assert "asyncpg" in settings.database.database_url

    def test_database_url_rejects_non_asyncpg(self) -> None:
        """A URL without asyncpg driver is rejected by the validator."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="asyncpg"):
            BudgetSettings.model_construct()  # unaffected; test DatabaseSettings directly
            from src.config.settings import DatabaseSettings

            DatabaseSettings(database_url="postgresql://user@localhost/db")


# ─── Redis Settings Tests ────────────────────────────────────────────


class TestRedisSettings:
    """Tests for RedisSettings defaults."""

    def test_redis_url_default(self) -> None:
        """Default redis_url is redis://localhost:6379/0."""
        redis_settings = RedisSettings()
        assert redis_settings.redis_url == "redis://localhost:6379/0"

    def test_cache_ttl_seconds_exists(self) -> None:
        """Verify cache_ttl_seconds field exists on RedisSettings."""
        redis_settings = RedisSettings()
        assert hasattr(redis_settings, "cache_ttl_seconds")
        assert isinstance(redis_settings.cache_ttl_seconds, int)
        assert redis_settings.cache_ttl_seconds > 0


# ─── Budget Settings Tests ───────────────────────────────────────────


class TestBudgetSettings:
    """Tests for BudgetSettings validation."""

    def test_budget_thresholds_valid(self) -> None:
        """Default critical threshold is greater than warn threshold."""
        budget = BudgetSettings()
        assert budget.budget_critical_threshold > budget.budget_warn_threshold

    def test_budget_critical_less_than_warn_rejected(self) -> None:
        """Critical <= warn threshold is rejected by the model validator."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="greater than warn"):
            BudgetSettings(
                budget_warn_threshold=0.90,
                budget_critical_threshold=0.70,
            )
