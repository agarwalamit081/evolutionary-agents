"""Tests for src.observability.logging — PII redaction and logging setup."""

from __future__ import annotations

from src.observability.logging import PIIRedactor, get_logger, reset_logging


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
        from loguru import logger

        result = get_logger()
        assert result is logger
