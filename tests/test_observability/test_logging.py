"""Tests for src.observability.logging — PII redaction, logging setup, and category sinks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from src.observability.logging import (
    LOG_CATEGORIES,
    PIIRedactor,
    _make_module_filter,
    get_logger,
    reset_logging,
    setup_logging,
)
from src.config.settings import LoggingSettings


class TestPIIRedactor:
    """Tests for the PIIRedactor class."""

    def test_redacts_bearer_tokens(self) -> None:
        """Bearer tokens are replaced with [REDACTED]."""
        redactor = PIIRedactor()
        result = redactor.redact("Authorization: Bearer abc123xyz456token")
        assert "[REDACTED]" in result
        assert "abc123xyz456token" not in result

    def test_redacts_sk_keys(self) -> None:
        """OpenAI-style sk- keys are replaced with [REDACTED]."""
        redactor = PIIRedactor()
        result = redactor.redact("Key: sk-abc12345678901234567890abcdefghij")
        assert "[REDACTED]" in result

    def test_redacts_api_keys(self) -> None:
        """Generic api_key assignments are redacted."""
        redactor = PIIRedactor()
        result = redactor.redact('api_key = "supersecretvalue1234567890"')
        assert "[REDACTED]" in result

    def test_redacts_postgresql_urls(self) -> None:
        """PostgreSQL connection strings are redacted."""
        redactor = PIIRedactor()
        result = redactor.redact("postgresql://user:pass@db.example.com:5432/mydb")
        assert "[REDACTED]" in result
        assert "user:pass" not in result

    def test_normal_text_unchanged(self) -> None:
        """Normal text passes through unchanged."""
        redactor = PIIRedactor()
        text = "The quick brown fox jumps over the lazy dog"
        assert redactor.redact(text) == text


class TestLoggingHelpers:
    """Tests for logging helper functions."""

    def test_reset_logging_clears_flag(self) -> None:
        """reset_logging sets _logging_configured to False."""
        import src.observability.logging as mod

        mod._logging_configured = True
        reset_logging()
        assert mod._logging_configured is False

    def test_get_logger_returns_logger(self) -> None:
        """get_logger returns the loguru logger instance."""
        result = get_logger()
        assert result is logger


class TestModuleFilter:
    """Tests for _make_module_filter."""

    def test_matching_prefix_passes(self) -> None:
        """Records from matching module prefix pass the filter."""
        filt = _make_module_filter(["src.llm."])
        record: dict[str, Any] = {"name": "src.llm.gateway"}
        assert filt(record) is True

    def test_non_matching_prefix_fails(self) -> None:
        """Records from non-matching modules are rejected."""
        filt = _make_module_filter(["src.llm."])
        record: dict[str, Any] = {"name": "src.tools.registry"}
        assert filt(record) is False

    def test_multiple_prefixes(self) -> None:
        """Multiple prefixes are OR-matched."""
        filt = _make_module_filter(["src.tools.", "src.sandbox."])
        assert filt({"name": "src.tools.registry"}) is True
        assert filt({"name": "src.sandbox.executor"}) is True
        assert filt({"name": "src.llm.gateway"}) is False

    def test_empty_name(self) -> None:
        """Empty name matches no prefix."""
        filt = _make_module_filter(["src.llm."])
        assert filt({"name": ""}) is False

    def test_none_name_does_not_raise(self) -> None:
        """A None name does not raise AttributeError.

        Regression: ``record["name"]`` is None for dynamically generated tool
        modules (e.g. ``<generated_tool:...>``). ``dict.get("name", "")``
        returns None (not the default) when the key is present-but-None, so the
        filter must coerce None → "" before ``.startswith`` rather than raise.
        """
        filt = _make_module_filter(["src.tools."])
        # Key present but value None — the generated-tool case.
        assert filt({"name": None}) is False
        # Key absent entirely must also be safe.
        assert filt({}) is False


class TestCategorySinks:
    """Tests for category-based log file creation."""

    def test_log_categories_define_expected_sinks(self) -> None:
        """LOG_CATEGORIES has llm, tools, subagents keys."""
        assert "llm" in LOG_CATEGORIES
        assert "tools" in LOG_CATEGORIES
        assert "subagents" in LOG_CATEGORIES

    def test_llm_category_captures_gateway(self) -> None:
        """LLM category filter matches src.llm.gateway."""
        filt = _make_module_filter(LOG_CATEGORIES["llm"])
        assert filt({"name": "src.llm.gateway"}) is True
        assert filt({"name": "src.llm.model_router"}) is True

    def test_tools_category_captures_tools_and_sandbox(self) -> None:
        """Tools category filter matches src.tools.* and src.sandbox.*."""
        filt = _make_module_filter(LOG_CATEGORIES["tools"])
        assert filt({"name": "src.tools.registry"}) is True
        assert filt({"name": "src.sandbox.executor"}) is True
        assert filt({"name": "src.llm.gateway"}) is False

    def test_subagents_category_captures_agents(self) -> None:
        """Subagents category filter matches src.agents.*."""
        filt = _make_module_filter(LOG_CATEGORIES["subagents"])
        assert filt({"name": "src.agents.runner"}) is True
        assert filt({"name": "src.agents.subgraph"}) is True
        assert filt({"name": "src.graph.nodes.classify"}) is False

    def test_setup_creates_category_log_files(self, tmp_path: Path) -> None:
        """setup_logging creates llm.log, tools.log, subagents.log."""
        reset_logging()
        settings = LoggingSettings(log_dir=str(tmp_path))
        setup_logging(settings)

        assert (tmp_path / "llm.log").exists()
        assert (tmp_path / "tools.log").exists()
        assert (tmp_path / "subagents.log").exists()
        assert (tmp_path / "turing_agent.log").exists()
        assert (tmp_path / "errors.log").exists()

        reset_logging()
