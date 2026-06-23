"""Tests for src.observability.logging — PII redaction, logging setup, and category sinks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from src.observability.logging import (
    LOG_CATEGORIES,
    PIIRedactor,
    _make_module_filter,
    add_query_log_sink,
    get_logger,
    remove_query_log_sink,
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


class TestQueryLogSink:
    """Tests for the per-query log sink add/remove lifecycle (#313)."""

    def test_add_returns_handler_id(self, tmp_path: Path) -> None:
        """add_query_log_sink returns a truthy handler id for teardown."""
        reset_logging()
        settings = LoggingSettings(log_dir=str(tmp_path))
        setup_logging(settings)
        try:
            sink_id = add_query_log_sink("query_alpha", settings)
            assert sink_id is not None
            assert isinstance(sink_id, int)
        finally:
            remove_query_log_sink(sink_id)  # type: ignore[arg-type]
            reset_logging()

    async def test_removed_sink_does_not_capture_subsequent_logs(
        self, tmp_path: Path
    ) -> None:
        """After remove_query_log_sink the file captures no further messages.

        Regression for #313: a per-query sink that was never torn down caused
        the *next* run's logs to bleed into the prior run's log file — which
        merged a separate q01 benchmark run into ``showcase-vector-db-2.log``
        and created the illusion of goal drift + non-termination. With teardown,
        run B's messages must NOT appear in run A's file.
        """
        reset_logging()
        settings = LoggingSettings(log_dir=str(tmp_path))
        setup_logging(settings)
        log_file = tmp_path / "run_a.log"
        try:
            sink_id = add_query_log_sink("run_a", settings)
            assert sink_id is not None
            # A marker that SHOULD be captured (logged while the sink is live).
            logger.info("RUN_A_MESSAGE_BEFORE_TEARDOWN")
            # enqueue=True writes via a background thread queue, so explicitly
            # await logger.complete() to guarantee RUN_A is on disk before we
            # tear the sink down (logger.complete returns a coroutine — only
            # awaited is it effective; a bare call in a sync test never drains).
            await logger.complete()
            # The fix: tear the sink down — the file stops capturing here.
            remove_query_log_sink(sink_id)
            # A marker for a *different* run, logged AFTER teardown — it must
            # NOT bleed into run_a.log (the bug it would have under #313).
            logger.info("RUN_B_MESSAGE_AFTER_TEARDOWN")
            await logger.complete()
        finally:
            reset_logging()

        content = log_file.read_text(encoding="utf-8")
        assert "RUN_A_MESSAGE_BEFORE_TEARDOWN" in content
        assert "RUN_B_MESSAGE_AFTER_TEARDOWN" not in content

    async def test_without_teardown_subsequent_logs_leak_into_file(
        self, tmp_path: Path
    ) -> None:
        """Without teardown a second run's logs DO leak into the first file.

        This is the inverse guard of
        :meth:`test_removed_sink_does_not_capture_subsequent_logs`: it proves the
        regression test is actually exercising the leak path (run B's marker
        appears in run_a.log when the sink is left open). It pins the #313
        failure mode so a regression to the old never-teardown code fails loudly.
        """
        reset_logging()
        settings = LoggingSettings(log_dir=str(tmp_path))
        setup_logging(settings)
        log_file = tmp_path / "run_a.log"
        try:
            sink_id = add_query_log_sink("run_a", settings)
            assert sink_id is not None
            logger.info("RUN_A_MESSAGE_BEFORE_TEARDOWN")
            # NO teardown — the sink stays live, so run B leaks in.
            logger.info("RUN_B_MESSAGE_AFTER_TEARDOWN")
            await logger.complete()
            remove_query_log_sink(sink_id)
        finally:
            reset_logging()

        content = log_file.read_text(encoding="utf-8")
        assert "RUN_A_MESSAGE_BEFORE_TEARDOWN" in content
        # This assertion documents the leak that the fix exists to prevent.
        assert "RUN_B_MESSAGE_AFTER_TEARDOWN" in content

    def test_remove_is_none_safe(self, tmp_path: Path) -> None:
        """remove_query_log_sink(None) and a bogus id never raise.

        Callers may not have captured an id (e.g. add_query_log_sink was mocked
        or raised), and remove must tolerate a stale/unknown id without raising.
        """
        reset_logging()
        settings = LoggingSettings(log_dir=str(tmp_path))
        setup_logging(settings)
        try:
            remove_query_log_sink(None)  # caller captured no id
            remove_query_log_sink(999999)  # already-removed / unknown id
        finally:
            reset_logging()
