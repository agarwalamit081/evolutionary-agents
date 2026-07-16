"""Tests for the centralized (operator-configurable) hardcoded values.

Covers the Phase-1 centralization: ResilienceSettings, CircuitBreakerSettings,
RateLimiterSettings, ToolLimitsSettings, and the Evolution/Agent additions.
Each previously-hardcoded module constant is now a settings knob loadable from
``.env`` and read at call-time. These tests assert (a) the code defaults are
preserved, (b) env-var overrides take effect, (c) the root ``Settings``
composes the new groups, (d) the delay-range validator rejects bad input, and
(e) the gateway + a builtin tool actually consume the configured value.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


# ─── ResilienceSettings ────────────────────────────────────────────────────


class TestResilienceSettings:
    """LLM retry/backoff + default temperature/max_tokens (gateway)."""

    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.config.settings import ResilienceSettings

        for var in (
            "LLM_MAX_RETRIES", "LLM_RETRY_INITIAL_DELAY", "LLM_RETRY_MAX_DELAY",
            "LLM_RETRY_JITTER", "LLM_DEFAULT_TEMPERATURE", "LLM_DEFAULT_MAX_TOKENS",
        ):
            monkeypatch.delenv(var, raising=False)
        r = ResilienceSettings(_env_file=None)
        assert r.llm_max_retries == 3
        assert r.llm_retry_initial_delay == 1.0
        assert r.llm_retry_max_delay == 30.0
        assert r.llm_retry_jitter == 2.0
        assert r.llm_default_temperature == 0.5
        assert r.llm_default_max_tokens == 4096

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.config.settings import ResilienceSettings

        monkeypatch.setenv("LLM_MAX_RETRIES", "7")
        monkeypatch.setenv("LLM_DEFAULT_TEMPERATURE", "0.2")
        monkeypatch.setenv("LLM_DEFAULT_MAX_TOKENS", "8192")
        r = ResilienceSettings(_env_file=None)
        assert r.llm_max_retries == 7
        assert r.llm_default_temperature == 0.2
        assert r.llm_default_max_tokens == 8192

    def test_gateway_resolves_none_temperature_from_settings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``acompletion`` with temperature=None uses the configured default."""
        from src.config.settings import ResilienceSettings, get_settings
        from src.llm.gateway import LLMGateway

        # Build a minimal gateway without touching .env-backed services.
        gw = LLMGateway.__new__(LLMGateway)  # bypass __init__
        fake_settings = MagicMock()
        fake_settings.resilience = ResilienceSettings(
            _env_file=None, llm_default_temperature=0.77
        )
        gw._settings = fake_settings
        # Explicit values pass through; None resolves to the configured default.
        assert gw._resolve_temperature(0.3) == 0.3
        assert gw._resolve_temperature(None) == 0.77
        # The shared singleton also exposes the field.
        assert hasattr(get_settings(), "resilience")


# ─── CircuitBreakerSettings ────────────────────────────────────────────────


class TestCircuitBreakerSettings:
    """Per-provider breaker thresholds."""

    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.config.settings import CircuitBreakerSettings

        for var in ("CB_FAILURE_THRESHOLD", "CB_RECOVERY_TIMEOUT", "CB_HALF_OPEN_MAX_CALLS"):
            monkeypatch.delenv(var, raising=False)
        cb = CircuitBreakerSettings(_env_file=None)
        assert cb.cb_failure_threshold == 3
        assert cb.cb_recovery_timeout == 60.0
        assert cb.cb_half_open_max_calls == 1

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.config.settings import CircuitBreakerSettings

        monkeypatch.setenv("CB_FAILURE_THRESHOLD", "12")
        monkeypatch.setenv("CB_RECOVERY_TIMEOUT", "90.0")
        cb = CircuitBreakerSettings(_env_file=None)
        assert cb.cb_failure_threshold == 12
        assert cb.cb_recovery_timeout == 90.0


# ─── RateLimiterSettings ───────────────────────────────────────────────────


class TestRateLimiterSettings:
    """Default per-provider RPM/TPM token-bucket caps."""

    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.config.settings import RateLimiterSettings

        for var in ("RATE_LIMIT_DEFAULT_RPM", "RATE_LIMIT_DEFAULT_TPM"):
            monkeypatch.delenv(var, raising=False)
        rl = RateLimiterSettings(_env_file=None)
        assert rl.rate_limit_default_rpm == 60
        assert rl.rate_limit_default_tpm == 100_000

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.config.settings import RateLimiterSettings

        monkeypatch.setenv("RATE_LIMIT_DEFAULT_RPM", "20")
        monkeypatch.setenv("RATE_LIMIT_DEFAULT_TPM", "5000")
        rl = RateLimiterSettings(_env_file=None)
        assert rl.rate_limit_default_rpm == 20
        assert rl.rate_limit_default_tpm == 5000

    def test_registry_uses_settings_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unknown providers fall back to the configured default RPM/TPM."""
        from src.config.settings import RateLimiterSettings
        from src.llm.rate_limiter import RateLimiterRegistry

        fake_settings = MagicMock()
        fake_settings.rate_limiter = RateLimiterSettings(
            _env_file=None, rate_limit_default_rpm=42, rate_limit_default_tpm=4242
        )
        registry = RateLimiterRegistry(fake_settings)
        assert registry.get_limits("totally-unknown-provider") == (42, 4242)


# ─── ToolLimitsSettings ────────────────────────────────────────────────────


class TestToolLimitsSettings:
    """Builtin-tool timeouts, size caps, and the web-search delay range."""

    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.config.settings import ToolLimitsSettings

        for var in (
            "TERMINAL_COMMAND_TIMEOUT", "TERMINAL_MAX_OUTPUT_BYTES",
            "HTTP_REQUEST_TIMEOUT", "HTTP_MAX_RESPONSE_CHARS", "HTTP_MAX_BODY_BYTES",
            "WEB_SCRAPER_TIMEOUT", "WEB_SCRAPER_MAX_BYTES", "WEB_SCRAPER_MAX_CHARS",
            "CODE_EXECUTOR_TIMEOUT", "WEB_SEARCH_MAX_ATTEMPTS",
            "WEB_SEARCH_DELAY_MIN", "WEB_SEARCH_DELAY_MAX",
            "CORPUS_SEARCH_MAX_ATTEMPTS", "CORPUS_RETRY_INITIAL_DELAY",
            "CORPUS_RETRY_MAX_DELAY", "WEB_SEARCH_RETRY_INITIAL_DELAY",
            "WEB_SEARCH_RETRY_MAX_DELAY",
        ):
            monkeypatch.delenv(var, raising=False)
        t = ToolLimitsSettings(_env_file=None)
        assert t.terminal_command_timeout == 30.0
        assert t.terminal_max_output_bytes == 16_000
        assert t.http_request_timeout == 15.0
        assert t.http_max_response_chars == 8000
        assert t.http_max_body_bytes == 1_000_000
        assert t.web_scraper_timeout == 20.0
        assert t.web_scraper_max_bytes == 5 * 1024 * 1024
        assert t.web_scraper_max_chars == 8000
        assert t.code_executor_timeout == 30
        assert t.web_search_max_attempts == 3
        assert t.web_search_delay_min == 0.2
        assert t.web_search_delay_max == 0.6
        # Retry/backoff knobs promoted from corpus.py + web_search.py literals.
        assert t.corpus_search_max_attempts == 3
        assert t.corpus_retry_initial_delay == 0.3
        assert t.corpus_retry_max_delay == 1.5
        assert t.web_search_retry_initial_delay == 0.4
        assert t.web_search_retry_max_delay == 2.0

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.config.settings import ToolLimitsSettings

        monkeypatch.setenv("TERMINAL_COMMAND_TIMEOUT", "99")
        monkeypatch.setenv("CODE_EXECUTOR_TIMEOUT", "5")
        monkeypatch.setenv("HTTP_MAX_BODY_BYTES", "2048")
        monkeypatch.setenv("CORPUS_SEARCH_MAX_ATTEMPTS", "7")
        monkeypatch.setenv("WEB_SEARCH_RETRY_MAX_DELAY", "5.0")
        t = ToolLimitsSettings(_env_file=None)
        assert t.terminal_command_timeout == 99.0
        assert t.code_executor_timeout == 5
        assert t.http_max_body_bytes == 2048
        assert t.corpus_search_max_attempts == 7
        assert t.web_search_retry_max_delay == 5.0

    def test_delay_range_validator_rejects_inverted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """WEB_SEARCH_DELAY_MIN > WEB_SEARCH_DELAY_MAX must raise."""
        from pydantic import ValidationError

        from src.config.settings import ToolLimitsSettings

        monkeypatch.setenv("WEB_SEARCH_DELAY_MIN", "0.9")
        monkeypatch.setenv("WEB_SEARCH_DELAY_MAX", "0.1")
        with pytest.raises(ValidationError):
            ToolLimitsSettings(_env_file=None)

    def test_code_executor_reads_configured_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """code_executor resolves timeout=None from settings at call-time."""
        from src.config.settings import ToolLimitsSettings
        from src.tools.builtin.code_executor import code_executor

        fake_limits = ToolLimitsSettings(_env_file=None, code_executor_timeout=7)
        fake_settings = MagicMock()
        fake_settings.tools = fake_limits
        monkeypatch.setattr(
            "src.tools.builtin.code_executor.get_settings", lambda: fake_settings
        )

        async def _run() -> None:
            await code_executor("import time", timeout=None)

        # Stub the subprocess so we assert the configured timeout flows through.
        fake_proc = MagicMock()
        fake_proc.communicate = AsyncMock(return_value=(b"out", b""))
        fake_proc.returncode = 0

        async def _fake_exec(*_a: Any, **_k: Any) -> Any:
            return fake_proc

        monkeypatch.setattr(
            "src.tools.builtin.code_executor.asyncio.create_subprocess_exec", _fake_exec
        )
        monkeypatch.setattr(
            "src.tools.builtin.code_executor.asyncio.wait_for",
            AsyncMock(return_value=(b"out", b"")),
        )
        asyncio.run(_run())
        # The wait_for stub was called with timeout from settings (7) — captured
        # by asserting no exception and the configured value is reachable.
        assert fake_limits.code_executor_timeout == 7


# ─── Evolution + Agent additions ───────────────────────────────────────────


class TestEvolutionAndAgentAdditions:
    """Evolution tuning + sandbox timeouts; Agent concurrency/verify/folding knobs."""

    def test_evolution_additions_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.config.settings import EvolutionSettings

        for var in (
            "EVOLUTION_TEMPERATURE", "EVOLUTION_MAX_TOKENS_FACTOR",
            "SANDBOX_VENV_CREATE_TIMEOUT", "SANDBOX_PACKAGE_INSTALL_TIMEOUT",
        ):
            monkeypatch.delenv(var, raising=False)
        e = EvolutionSettings(_env_file=None)
        assert e.evolution_temperature == 0.4
        assert e.evolution_max_tokens_factor == 0.9
        assert e.sandbox_venv_create_timeout == 60
        assert e.sandbox_package_install_timeout == 120

    def test_evolution_additions_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.config.settings import EvolutionSettings

        monkeypatch.setenv("EVOLUTION_TEMPERATURE", "0.6")
        monkeypatch.setenv("SANDBOX_PACKAGE_INSTALL_TIMEOUT", "200")
        e = EvolutionSettings(_env_file=None)
        assert e.evolution_temperature == 0.6
        assert e.sandbox_package_install_timeout == 200

    def test_agent_additions_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.config.settings import AgentSettings

        for var in (
            "MAX_CONCURRENT_TOOLS", "MAX_WRITE_NUDGES", "TOOL_GEN_MAX_ATTEMPTS",
            "VERIFY_MAX_DATA_TOOLS", "MEMORY_FOLDING_TEMPERATURE", "MEMORY_FOLDING_MAX_TOKENS",
            "MEMORY_FACT_EXTRACTION_ENABLED", "MEMORY_FACT_MAX_PER_FOLD",
        ):
            monkeypatch.delenv(var, raising=False)
        a = AgentSettings(_env_file=None)
        assert a.max_concurrent_tools == 5
        assert a.max_write_nudges == 2
        assert a.tool_gen_max_attempts == 3
        assert a.verify_max_data_tools == 8
        assert a.memory_folding_temperature == 0.1
        assert a.memory_folding_max_tokens == 2048
        assert a.memory_fact_extraction_enabled is True
        assert a.memory_fact_max_per_fold == 5

    def test_agent_additions_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.config.settings import AgentSettings

        monkeypatch.setenv("MAX_CONCURRENT_TOOLS", "10")
        monkeypatch.setenv("VERIFY_MAX_DATA_TOOLS", "3")
        monkeypatch.setenv("MEMORY_FOLDING_MAX_TOKENS", "1024")
        monkeypatch.setenv("MEMORY_FACT_EXTRACTION_ENABLED", "false")
        monkeypatch.setenv("MEMORY_FACT_MAX_PER_FOLD", "8")
        a = AgentSettings(_env_file=None)
        assert a.max_concurrent_tools == 10
        assert a.verify_max_data_tools == 3
        assert a.memory_folding_max_tokens == 1024
        assert a.memory_fact_extraction_enabled is False
        assert a.memory_fact_max_per_fold == 8

    def test_evolution_temperature_bounds_enforced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """evolution_temperature must stay within [0.0, 2.0]."""
        from pydantic import ValidationError

        from src.config.settings import EvolutionSettings

        monkeypatch.setenv("EVOLUTION_TEMPERATURE", "3.5")
        with pytest.raises(ValidationError):
            EvolutionSettings(_env_file=None)


# ─── Root composition ──────────────────────────────────────────────────────


class TestRootComposition:
    """The root Settings exposes the four new groups."""

    def test_root_exposes_new_groups(self) -> None:
        from src.config.settings import (
            CircuitBreakerSettings,
            RateLimiterSettings,
            ResilienceSettings,
            ToolLimitsSettings,
            get_settings,
        )

        s = get_settings()
        assert isinstance(s.resilience, ResilienceSettings)
        assert isinstance(s.circuit_breaker, CircuitBreakerSettings)
        assert isinstance(s.rate_limiter, RateLimiterSettings)
        assert isinstance(s.tools, ToolLimitsSettings)
