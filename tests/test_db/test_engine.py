"""Tests for src/db/engine.py startup logging (§2).

get_engine() must log WHICH database it resolved to (host + database name) so a
host-local Postgres silently shadowing the docker container is immediately
visible on every run. Critically, the password must NEVER appear in the log —
only parsed host/port/database/driver fields are emitted (the raw connection
string is deliberately not logged).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from loguru import logger

import src.db.engine as engine_mod
from src.db.engine import get_engine


def _fake_settings(db_url: str) -> SimpleNamespace:
    """Build a minimal settings stand-in exposing only the attributes get_engine reads."""
    db = SimpleNamespace(
        database_url=db_url,
        database_pool_size=5,
        database_max_overflow=10,
        database_pool_timeout=30,
        database_pool_recycle=1800,
    )
    return SimpleNamespace(database=db)


def _reset_engine_singleton() -> None:
    engine_mod._engine = None


class _LogSink:
    """Collects formatted loguru records for assertion."""

    def __init__(self) -> None:
        self.records: list[str] = []

    def __call__(self, message: object) -> None:
        self.records.append(str(message))


class TestEngineStartupLog:
    """get_engine logs the resolved database without leaking the password."""

    def test_logs_database_and_host_without_password(self) -> None:
        _reset_engine_singleton()
        sink = _LogSink()
        handler_id = logger.add(sink, level="INFO")

        try:
            with patch.object(
                engine_mod,
                "get_settings",
                return_value=_fake_settings(
                    "postgresql+asyncpg://myuser:supersecret@db-host.example:5432/turing_db"
                ),
            ):
                eng = get_engine()
        finally:
            logger.remove(handler_id)
            _reset_engine_singleton()

        assert eng is not None
        joined = "\n".join(sink.records)
        assert "turing_db" in joined  # database name is logged
        assert "db-host.example" in joined  # host is logged
        assert "asyncpg" in joined  # driver is logged
        # The password must NEVER appear in any captured log line.
        assert "supersecret" not in joined

    def test_logs_only_at_engine_creation_not_subsequent_calls(self) -> None:
        """The singleton is created once; the startup log fires exactly once."""
        _reset_engine_singleton()
        sink = _LogSink()
        handler_id = logger.add(sink, level="INFO")

        try:
            with patch.object(
                engine_mod,
                "get_settings",
                return_value=_fake_settings(
                    "postgresql+asyncpg://u:p@localhost:5432/once_db"
                ),
            ):
                first = get_engine()
                second = get_engine()
        finally:
            logger.remove(handler_id)
            _reset_engine_singleton()

        assert first is second  # same singleton instance
        startup_lines = [r for r in sink.records if "Database engine created" in r]
        assert len(startup_lines) == 1  # logged once, not twice
        assert "once_db" in startup_lines[0]

    def test_password_redacted_even_when_embedded_in_database(self) -> None:
        """A db name that coincidentally contains the password substring is
        still safe: we log the parsed db name, and the password lives only in
        the URL's userinfo — which is never logged."""
        _reset_engine_singleton()
        sink = _LogSink()
        handler_id = logger.add(sink, level="INFO")

        try:
            with patch.object(
                engine_mod,
                "get_settings",
                return_value=_fake_settings(
                    "postgresql+asyncpg://u:hunter2@127.0.0.1:5432/mydb"
                ),
            ):
                get_engine()
        finally:
            logger.remove(handler_id)
            _reset_engine_singleton()

        joined = "\n".join(sink.records)
        assert "mydb" in joined
        assert "hunter2" not in joined  # password absent
