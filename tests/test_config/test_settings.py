"""Tests for src.config.settings module."""

from __future__ import annotations

import pytest

from src.config.settings import (
    BudgetSettings,
    RedisSettings,
    Settings,
    get_settings,
)

# NOTE on testing defaults: importing the test suite pulls in `litellm` (via the
# conftest → graph → gateway chain), and `import litellm` side-effect-loads the
# project `.env` into os.environ. So a settings group's CODE DEFAULT can only be
# asserted when BOTH the os.environ value is removed (monkeypatch.delenv) AND the
# .env file is skipped (_env_file=None). Tests that assert a default therefore
# take a `monkeypatch` fixture and delenv the relevant var; env-override tests
# set the var explicitly. (See also: settings.py reads case-insensitively.)


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

    def test_redis_url_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default redis_url is redis://localhost:6380/0 (compose host port, aligns with .env.example)."""
        monkeypatch.delenv("REDIS_URL", raising=False)
        redis_settings = RedisSettings(_env_file=None)
        assert redis_settings.redis_url == "redis://localhost:6380/0"

    def test_cache_ttl_seconds_exists(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify cache_ttl_seconds field exists on RedisSettings."""
        monkeypatch.delenv("CACHE_TTL_SECONDS", raising=False)
        redis_settings = RedisSettings(_env_file=None)
        assert hasattr(redis_settings, "cache_ttl_seconds")
        assert isinstance(redis_settings.cache_ttl_seconds, int)
        assert redis_settings.cache_ttl_seconds > 0


# ─── Budget Settings Tests ───────────────────────────────────────────


class TestBudgetSettings:
    """Tests for BudgetSettings validation."""

    def test_budget_thresholds_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default critical threshold is greater than warn threshold."""
        monkeypatch.delenv("BUDGET_CRITICAL_THRESHOLD", raising=False)
        monkeypatch.delenv("BUDGET_WARN_THRESHOLD", raising=False)
        budget = BudgetSettings(_env_file=None)
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

    def test_max_iterations_default_is_60(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The canonical iteration cap default is 60 (aligns with .env.example)."""
        from src.config.settings import AgentSettings

        monkeypatch.delenv("MAX_ITERATIONS", raising=False)
        assert AgentSettings(_env_file=None).max_iterations == 60

    def test_run_caps_have_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Tool and sub-agent run caps default to 12/5 and are overridable fields (align with .env.example)."""
        from src.config.settings import AgentSettings

        monkeypatch.delenv("MAX_TOOLS_PER_RUN", raising=False)
        monkeypatch.delenv("MAX_SUB_AGENTS_PER_RUN", raising=False)
        agent = AgentSettings(_env_file=None)
        assert agent.max_tools_per_run == 12
        assert agent.max_sub_agents_per_run == 5

    def test_max_iterations_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The iteration cap is env-overridable. Settings read case-insensitively, so a
        lowercase env var matches the lowercase field name."""
        from src.config.settings import AgentSettings

        monkeypatch.setenv("max_iterations", "15")
        assert AgentSettings(_env_file=None).max_iterations == 15

    def test_run_caps_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Tool/sub-agent run caps are env-overridable (case-insensitive matching)."""
        from src.config.settings import AgentSettings

        monkeypatch.setenv("max_tools_per_run", "5")
        monkeypatch.setenv("max_sub_agents_per_run", "7")
        agent = AgentSettings(_env_file=None)
        assert agent.max_tools_per_run == 5
        assert agent.max_sub_agents_per_run == 7

    def test_uppercase_env_override_is_honored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Regression: UPPERCASE .env keys MUST populate the lowercase fields.

        Previously every settings group used case_sensitive=True, which derives the
        env lookup key from the lowercase field name — so MAX_ITERATIONS,
        MAX_TOOLS_PER_RUN, MAX_SUB_AGENTS_PER_RUN (and DATABASE_URL, REDIS_URL, ...)
        in .env were silently ignored and the code default was used. With
        case_sensitive=False they now match case-insensitively (the real-world .env
        convention). This test fails under the old case_sensitive=True setting.
        """
        from src.config.settings import AgentSettings

        monkeypatch.setenv("MAX_ITERATIONS", "19")
        monkeypatch.setenv("MAX_TOOLS_PER_RUN", "2")
        monkeypatch.setenv("MAX_SUB_AGENTS_PER_RUN", "4")
        agent = AgentSettings(_env_file=None)
        assert agent.max_iterations == 19
        assert agent.max_tools_per_run == 2
        assert agent.max_sub_agents_per_run == 4

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

    def test_cumulative_caps_and_retire_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """B3 de-bloat: cumulative caps + retirement fields default as specified."""
        from src.config.settings import AgentSettings

        for var in (
            "MAX_ACTIVE_TOOLS",
            "MAX_ACTIVE_SUB_AGENTS",
            "CAPABILITY_REDUNDANCY_THRESHOLD",
            "RETIRE_MIN_RUNS",
            "RETIRE_SUCCESS_FLOOR",
            "RETIRE_EMPTY_OUTPUT_FLOOR",
            "RETIRE_RECENCY_DAYS",
        ):
            monkeypatch.delenv(var, raising=False)
        agent = AgentSettings(_env_file=None)
        assert agent.max_active_tools == 25
        assert agent.max_active_sub_agents == 60
        assert agent.capability_redundancy_threshold == 0.92
        assert agent.retire_min_runs == 20
        assert agent.retire_success_floor == 0.5
        assert agent.retire_empty_output_floor == 0.8
        assert agent.retire_recency_days == 30

    def test_cumulative_caps_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Cumulative caps + retirement are env-overridable (case-insensitive)."""
        from src.config.settings import AgentSettings

        monkeypatch.setenv("max_active_tools", "40")
        monkeypatch.setenv("max_active_sub_agents", "20")
        monkeypatch.setenv("capability_redundancy_threshold", "0.80")
        monkeypatch.setenv("retire_min_runs", "50")
        monkeypatch.setenv("retire_success_floor", "0.10")
        monkeypatch.setenv("retire_recency_days", "7")
        agent = AgentSettings(_env_file=None)
        assert agent.max_active_tools == 40
        assert agent.max_active_sub_agents == 20
        assert agent.capability_redundancy_threshold == 0.80
        assert agent.retire_min_runs == 50
        assert agent.retire_success_floor == 0.10
        assert agent.retire_recency_days == 7


class TestWorkerSettings:
    """The documented ``WORKER_*`` env vars (see ``.env.example``) MUST map.

    Regression for a Phase-2b bug surfaced in Phase 3: ``WorkerSettings`` had no
    ``env_prefix``, so only the accidental bare forms (``CONSUMER_NAME`` …) were
    honored and every documented ``WORKER_*`` var was silently ignored — e.g.
    ``WORKER_CONSUMER_NAME=foo`` left ``consumer_name='worker-1'``, which broke the
    worker entrypoint's explicit-name opt-in. The fix is ``env_prefix='worker_'``
    (+ ``worker_group``/``status_ttl_seconds`` renamed to ``group``/``status_ttl_s``
    so the generated names match ``.env.example`` exactly).
    """

    def test_documented_worker_env_vars_all_map(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """All 7 WORKER_* vars documented in .env.example map to their fields."""
        from src.config.settings import WorkerSettings

        monkeypatch.setenv("WORKER_RUNS_STREAM", "s:stream")
        monkeypatch.setenv("WORKER_GROUP", "s-group")
        monkeypatch.setenv("WORKER_CONSUMER_NAME", "s-consumer")
        monkeypatch.setenv("WORKER_READ_BATCH_SIZE", "42")
        monkeypatch.setenv("WORKER_BLOCK_MS", "1111")
        monkeypatch.setenv("WORKER_RECLAIM_MIN_IDLE_MS", "2222")
        monkeypatch.setenv("WORKER_STATUS_TTL_S", "333")
        monkeypatch.setenv("WORKER_DEAD_LETTER_MAX_ATTEMPTS", "7")
        w = WorkerSettings(_env_file=None)
        assert w.runs_stream == "s:stream"
        assert w.group == "s-group"
        assert w.consumer_name == "s-consumer"
        assert w.read_batch_size == 42
        assert w.block_ms == 1111
        assert w.reclaim_min_idle_ms == 2222
        assert w.status_ttl_s == 333
        assert w.dead_letter_max_attempts == 7

    def test_consumer_name_round_trip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """WORKER_CONSUMER_NAME populates consumer_name — the __main__ opt-in path."""
        from src.config.settings import WorkerSettings

        monkeypatch.setenv("WORKER_CONSUMER_NAME", "pinned-worker")
        w = WorkerSettings(_env_file=None)
        assert w.consumer_name == "pinned-worker"

    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Code defaults match .env.example when no WORKER_* vars are set."""
        from src.config.settings import WorkerSettings

        for var in (
            "WORKER_RUNS_STREAM",
            "WORKER_GROUP",
            "WORKER_CONSUMER_NAME",
            "WORKER_READ_BATCH_SIZE",
            "WORKER_BLOCK_MS",
            "WORKER_RECLAIM_MIN_IDLE_MS",
            "WORKER_STATUS_TTL_S",
            "WORKER_DEAD_LETTER_MAX_ATTEMPTS",
        ):
            monkeypatch.delenv(var, raising=False)
        w = WorkerSettings(_env_file=None)
        assert w.runs_stream == "turing:runs"
        assert w.group == "turing-workers"
        # EMPTY by default: a replicated worker pool must auto-derive a unique
        # consumer name per replica (src/worker/__main__._resolve_consumer_name).
        # A fixed default here would make every replica collide on it.
        assert w.consumer_name == ""
        assert w.read_batch_size == 5
        assert w.block_ms == 5000
        assert w.reclaim_min_idle_ms == 30000
        assert w.status_ttl_s == 86400
        # Default dead-letter cap (Bug B): 3 failed attempts before a poison run is
        # acked + marked FAILED permanently (stops infinite redelivery).
        assert w.dead_letter_max_attempts == 3

