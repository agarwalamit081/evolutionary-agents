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


# ─── Agent Settings Tests (§7: single source of truth for run caps) ──


class TestAgentSettings:
    """AgentSettings run-cap defaults and env overrides (§7)."""

    def test_max_iterations_default_is_25(self) -> None:
        """The canonical iteration cap default is 25 (CLI/factory/routers align)."""
        from src.config.settings import AgentSettings

        assert AgentSettings(_env_file=None).max_iterations == 25

    def test_run_caps_have_defaults(self) -> None:
        """Tool and sub-agent run caps default to 3 and are overridable fields."""
        from src.config.settings import AgentSettings

        agent = AgentSettings(_env_file=None)
        assert agent.max_tools_per_run == 3
        assert agent.max_sub_agents_per_run == 3

    def test_max_iterations_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The iteration cap is env-overridable. AgentSettings uses
        case_sensitive=True, so the env var matches the field name (lowercase)."""
        from src.config.settings import AgentSettings

        monkeypatch.setenv("max_iterations", "15")
        assert AgentSettings(_env_file=None).max_iterations == 15

    def test_run_caps_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Tool/sub-agent run caps are env-overridable (case-sensitive, lowercase)."""
        from src.config.settings import AgentSettings

        monkeypatch.setenv("max_tools_per_run", "5")
        monkeypatch.setenv("max_sub_agents_per_run", "7")
        agent = AgentSettings(_env_file=None)
        assert agent.max_tools_per_run == 5
        assert agent.max_sub_agents_per_run == 7

    def test_run_caps_reject_non_positive(self) -> None:
        """Zero/negative run caps are rejected by the positive-int validator."""
        from pydantic import ValidationError

        from src.config.settings import AgentSettings

        with pytest.raises(ValidationError, match="positive integer"):
            AgentSettings(_env_file=None, max_tools_per_run=0)
        with pytest.raises(ValidationError, match="positive integer"):
            AgentSettings(_env_file=None, max_sub_agents_per_run=-1)

    def test_dead_max_sub_agents_field_removed(self) -> None:
        """The dead max_sub_agents field is gone (folded into max_sub_agents_per_run)."""
        from src.config.settings import AgentSettings

        assert not hasattr(AgentSettings(_env_file=None), "max_sub_agents")
