"""
Loguru logging configuration for Turing Agent.

Configures structured logging with console and file handlers, PII redaction,
and stdlib logging interception for third-party library integration.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from types import FrameType
from typing import Any, Literal, cast

from loguru import logger

from src.config.settings import LoggingSettings, get_settings


# ─── PII Redaction Filter ───────────────────────────────────────────────


class PIIRedactor:
    """Redacts sensitive information from log messages."""

    # Patterns to match and redact
    PATTERNS = [
        # Bearer tokens
        (r'Bearer\s+[A-Za-z0-9\-._~+/]+=*', '[REDACTED]'),
        # OpenAI-style keys (sk-...)
        (r'sk-[A-Za-z0-9]{20,}', '[REDACTED]'),
        # Generic sensitive patterns with minimum length (reduced from 20+ to 4+)
        (r'[A-Za-z0-9_\-]*api_key["\']?\s*[:=]\s*["\']?[A-Za-z0-9\-]{4,}', 'api_key: [REDACTED]'),
        (r'[A-Za-z0-9_\-]*secret["\']?\s*[:=]\s*["\']?[A-Za-z0-9\-]{4,}', 'secret: [REDACTED]'),
        (r'[A-Za-z0-9_\-]*token["\']?\s*[:=]\s*["\']?[A-Za-z0-9\-]{4,}', 'token: [REDACTED]'),
        (r'[A-Za-z0-9_\-]*password["\']?\s*[:=]\s*["\']?[^\s"]{4,}', 'password: [REDACTED]'),
        # JWT tokens
        (r'eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]*', '[REDACTED]'),
        # Database connection strings
        (r'postgresql://[^@]+@[^/]+/[^\s]+', 'postgresql://[REDACTED]@[REDACTED]/[REDACTED]'),
        (r'redis://[^@]+@[^:]+:[0-9]+', 'redis://[REDACTED]@[REDACTED]:[PORT]'),
        # Additional API key patterns
        (r'["\']?sk-[A-Za-z0-9]{20,}["\']?', '[REDACTED]'),
        (r'["\']?pk-[A-Za-z0-9]{20,}["\']?', '[REDACTED]'),
    ]

    def __init__(self) -> None:
        """Compile regex patterns for performance."""
        self.compiled_patterns = [(re.compile(pattern, re.IGNORECASE), replacement)
                                   for pattern, replacement in self.PATTERNS]

    def redact(self, message: str) -> str:
        """Redact sensitive information from a log message.

        Args:
            message: The original log message.

        Returns:
            The message with sensitive patterns redacted.
        """
        redacted = message
        for pattern, replacement in self.compiled_patterns:
            redacted = pattern.sub(replacement, redacted)
        return redacted


_pii_redactor = PIIRedactor()


def pii_redaction_filter(record: Any) -> Literal[True]:
    """Loguru filter function that redacts PII from log messages.

    Args:
        record: The loguru record to process.

    Returns:
        True to indicate the record should be logged (after modification).
    """
    if "message" in record:
        record["message"] = _pii_redactor.redact(str(record["message"]))
    return True


# ─── Stdlib Logging Interception ────────────────────────────────────────


class InterceptHandler(logging.Handler):
    """Intercepts stdlib logging messages and routes them to loguru.

    This ensures that third-party libraries (LangChain, SQLAlchemy, etc.)
    use loguru's formatting and handlers.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record to loguru.

        Args:
            record: The stdlib logging record to intercept.
        """
        # Get corresponding loguru level
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)

        # Find the caller from the stack
        current_frame: FrameType | None = cast(FrameType | None, logging.currentframe())
        depth = 2
        while current_frame and current_frame.f_code.co_filename == logging.__file__:
            current_frame = current_frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def _configure_stdlib_interception() -> None:
    """Configure stdlib logging to route through loguru."""
    # Remove all existing stdlib handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Add our intercept handler
    intercept_handler = InterceptHandler()
    root_logger.addHandler(intercept_handler)

    # Set loguru as the only handler for all loggers
    # This prevents duplicate logs
    for name in ["langchain", "sqlalchemy", "httpx", "httpcore", "uvicorn"]:
        logging.getLogger(name).handlers = [intercept_handler]
        logging.getLogger(name).propagate = False

    # Set stdlib root level to NOTSET so loguru handles all levels
    root_logger.setLevel(logging.NOTSET)


# ─── Logging Configuration ──────────────────────────────────────────────


def _get_console_format() -> str:
    """Get the rich console format string."""
    return (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )


def _get_file_format() -> str:
    """Get the plain text file format string."""
    return (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{name}:{function}:{line} | "
        "{message}"
    )


def _get_json_format() -> str:
    """Get the JSON format string (for structured logging)."""
    # loguru's serialize=True produces JSON with full record
    # We just need to ensure PII redaction happens first
    return ""


# Track if setup has been called to ensure idempotency
_logging_configured: bool = False


def setup_logging(settings: LoggingSettings | None = None) -> None:
    """Configure loguru logging for the application.

    This function is idempotent — safe to call multiple times.
    Subsequent calls will not reconfigure logging.

    Args:
        settings: Optional LoggingSettings instance. If not provided,
                 settings will be loaded via get_settings().
    """
    global _logging_configured

    if _logging_configured:
        logger.debug("Logging already configured. Skipping setup_logging().")
        return

    if settings is None:
        try:
            settings = get_settings().logging
        except Exception:
            # Fallback to defaults if settings are unavailable
            settings = LoggingSettings()

    # Remove default loguru handler
    logger.remove()

    # Ensure log directory exists
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ─── Console Handler ────────────────────────────────────────────────
    logger.add(
        sys.stderr,
        format=_get_console_format(),
        level=settings.log_level,
        colorize=True,
        backtrace=True,
        diagnose=True,
        filter=pii_redaction_filter,
    )

    # ─── File Handler ───────────────────────────────────────────────────
    logger.add(
        log_dir / "turing_agent.log",
        format=_get_file_format(),
        level=settings.log_level,
        rotation=settings.log_rotation,
        retention=settings.log_retention,
        compression="zip",
        backtrace=True,
        diagnose=True,
        filter=pii_redaction_filter,
        enqueue=True,  # Thread-safe logging
    )

    # ─── Error File Handler ────────────────────────────────────────────
    logger.add(
        log_dir / "errors.log",
        format=_get_file_format(),
        level="WARNING",
        rotation=settings.log_rotation,
        retention=settings.log_retention,
        compression="zip",
        backtrace=True,
        diagnose=True,
        filter=pii_redaction_filter,
        enqueue=True,
    )

    # ─── JSON Handler (for structured logging) ─────────────────────────
    if settings.log_format == "structured":
        logger.add(
            log_dir / "turing_agent.jsonl",
            format=_get_json_format(),
            level=settings.log_level,
            rotation=settings.log_rotation,
            retention=settings.log_retention,
            compression="zip",
            serialize=True,  # JSON format
            filter=pii_redaction_filter,
            enqueue=True,
        )

    # ─── Configure stdlib interception ─────────────────────────────────
    _configure_stdlib_interception()

    # Mark as configured
    _logging_configured = True

    logger.info(
        f"Logging configured: level={settings.log_level}, "
        f"format={settings.log_format}, dir={settings.log_dir}"
    )


# ─── Convenience Functions ───────────────────────────────────────────────


def get_logger() -> Any:
    """Get the configured loguru logger instance.

    Returns:
        The loguru logger.

    Example:
        from src.observability.logging import get_logger
        logger = get_logger()
        logger.info("Application started")
    """
    return logger


def reset_logging() -> None:
    """Reset logging configuration (primarily for testing).

    This clears all handlers and resets the configured flag.
    After calling this, setup_logging() will run again.
    """
    global _logging_configured
    logger.remove()
    _logging_configured = False


# ─── Module Exports ─────────────────────────────────────────────────────

__all__ = [
    "logger",
    "setup_logging",
    "get_logger",
    "reset_logging",
    "PIIRedactor",
    "pii_redaction_filter",
    "InterceptHandler",
]
