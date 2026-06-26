"""create_scheduled_task builtin (Phase 5 I1): validation + upsert + cap.

The DB path is verified with a capturing fake session (never a real DB),
mirroring ``test_metrics.py``. ``_FakeResult`` serves its single stored value via
whichever scalar accessor the tool calls (``scalar_one`` for the count query,
``scalar_one_or_none`` for the select-by-name) so one fake result type covers
both query shapes; the fake session returns ordered responses (one per execute
call). The four validation paths return BEFORE the DB block, so they patch
``get_session`` with an asserting mock to prove it's never reached.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from src.db.models import ScheduledTask
from src.tools.builtin.schedule_task import create_scheduled_task


def _settings(
    *, enabled: bool = True, max_tasks: int = 25, max_goal_chars: int = 2000
) -> object:
    return SimpleNamespace(
        agent_cron=SimpleNamespace(
            enabled=enabled,
            max_tasks=max_tasks,
            max_goal_chars=max_goal_chars,
        )
    )


class _FakeResult:
    """Serves one value through either scalar accessor the caller uses."""

    def __init__(self, value: Any) -> None:
        self._value = value

    def scalar_one(self) -> Any:
        return self._value

    def scalar_one_or_none(self) -> Any:
        return self._value


class _FakeSession:
    """Async CM yielding itself; returns ordered execute() responses."""

    def __init__(self, responses: list[Any]) -> None:
        # One response per expected execute() call, in call order.
        self._responses = list(responses)
        self.stmts: list[Any] = []
        self.added: list[Any] = []

    async def execute(self, stmt: Any) -> _FakeResult:
        self.stmts.append(stmt)
        value = self._responses.pop(0) if self._responses else None
        return _FakeResult(value)

    def add(self, obj: Any) -> None:
        self.added.append(obj)

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, *_args: Any) -> bool:
        return False


def _assert_no_db() -> object:
    """A get_session replacement that fails if the DB path is reached."""
    raise AssertionError("DB path should not be reached on a validation rejection")


class TestValidationPaths:
    """Each rejection returns before touching the DB."""

    @pytest.mark.asyncio
    async def test_disabled_is_noop(self) -> None:
        with patch("src.config.settings.get_settings", lambda: _settings(enabled=False)), patch(
            "src.db.session.get_session", _assert_no_db
        ):
            result = await create_scheduled_task("d", "0 9 * * *", "g")
        assert "disabled" in result.lower()

    @pytest.mark.asyncio
    async def test_rejects_empty_required_field(self) -> None:
        with patch("src.config.settings.get_settings", lambda: _settings()), patch(
            "src.db.session.get_session", _assert_no_db
        ):
            result = await create_scheduled_task("", "0 9 * * *", "g")
        assert "required" in result.lower()

    @pytest.mark.asyncio
    async def test_rejects_bad_cron(self) -> None:
        with patch("src.config.settings.get_settings", lambda: _settings()), patch(
            "src.db.session.get_session", _assert_no_db
        ):
            result = await create_scheduled_task("d", "not a cron", "g")
        assert "invalid cron" in result.lower()

    @pytest.mark.asyncio
    async def test_rejects_oversized_goal(self) -> None:
        with patch("src.config.settings.get_settings", lambda: _settings(max_goal_chars=10)), patch(
            "src.db.session.get_session", _assert_no_db
        ):
            result = await create_scheduled_task("d", "0 9 * * *", "x" * 50)
        assert "cap is 10" in result.lower()


class TestUpsertAndCap:
    @pytest.mark.asyncio
    async def test_creates_new_task_row(self) -> None:
        # select-by-name → None (no existing); count(enabled) → 0.
        session = _FakeSession(responses=[None, 0])
        with patch("src.config.settings.get_settings", lambda: _settings()), patch(
            "src.db.session.get_session", lambda: session
        ):
            result = await create_scheduled_task(
                "weekday-report", "0 9 * * 1-5", "refresh the daily report"
            )
        assert result.startswith("Created")
        assert len(session.added) == 1
        row = session.added[0]
        assert isinstance(row, ScheduledTask)
        assert row.name == "weekday-report"
        assert row.cron == "0 9 * * 1-5"
        assert row.goal == "refresh the daily report"
        assert row.model is None
        assert row.timezone == "UTC"
        assert row.enabled is True

    @pytest.mark.asyncio
    async def test_model_pin_is_stored(self) -> None:
        session = _FakeSession(responses=[None, 0])
        with patch("src.config.settings.get_settings", lambda: _settings()), patch(
            "src.db.session.get_session", lambda: session
        ):
            await create_scheduled_task("d", "0 9 * * *", "g", model="glm-4.7")
        assert session.added[0].model == "glm-4.7"

    @pytest.mark.asyncio
    async def test_revises_existing_row_by_name(self) -> None:
        existing = ScheduledTask(
            name="weekday-report",
            cron="0 8 * * 1-5",
            goal="old goal",
            model="glm-4.7",
            timezone="UTC",
            enabled=False,
        )
        # select-by-name → the existing row (no count query on the update path).
        session = _FakeSession(responses=[existing])
        with patch("src.config.settings.get_settings", lambda: _settings()), patch(
            "src.db.session.get_session", lambda: session
        ):
            result = await create_scheduled_task(
                "weekday-report", "0 9 * * 1-5", "new goal"
            )
        assert result.startswith("Updated")
        assert session.added == []  # update path never calls session.add()
        # The existing row was revised in place.
        assert existing.cron == "0 9 * * 1-5"
        assert existing.goal == "new goal"
        assert existing.model is None  # empty model clears the pin
        assert existing.enabled is True  # re-enabled

    @pytest.mark.asyncio
    async def test_cap_reached_rejects_new(self) -> None:
        # select-by-name → None; count(enabled) → at the cap (25).
        session = _FakeSession(responses=[None, 25])
        with patch("src.config.settings.get_settings", lambda: _settings(max_tasks=25)), patch(
            "src.db.session.get_session", lambda: session
        ):
            result = await create_scheduled_task("new", "0 9 * * *", "g")
        assert "cap reached" in result.lower()
        assert session.added == []

    @pytest.mark.asyncio
    async def test_cap_does_not_block_revision(self) -> None:
        # Revising an EXISTING name must not trip the cap (it consumes no new slot)
        # even when enabled count is at the cap.
        existing = ScheduledTask(
            name="existing", cron="0 8 * * *", goal="old", timezone="UTC", enabled=True
        )
        session = _FakeSession(responses=[existing])  # update path → no count query
        with patch(
            "src.config.settings.get_settings", lambda: _settings(max_tasks=1)
        ), patch("src.db.session.get_session", lambda: session):
            result = await create_scheduled_task("existing", "0 9 * * *", "new")
        assert result.startswith("Updated")

    @pytest.mark.asyncio
    async def test_unknown_timezone_falls_back_to_utc(self) -> None:
        session = _FakeSession(responses=[None, 0])
        with patch("src.config.settings.get_settings", lambda: _settings()), patch(
            "src.db.session.get_session", lambda: session
        ):
            result = await create_scheduled_task(
                "d", "0 9 * * *", "g", timezone="Mars/Olympus"
            )
        assert result.startswith("Created")
        assert "UTC" in result
        assert session.added[0].timezone == "UTC"


def test_tool_definition_shape() -> None:
    """The builtin ships a valid, cacheable=False, fully-described definition."""
    from src.tools.builtin.schedule_task import TOOL_DEFINITION

    assert TOOL_DEFINITION["name"] == "create_scheduled_task"
    assert callable(TOOL_DEFINITION["handler"])
    assert TOOL_DEFINITION["cacheable"] is False
    params = TOOL_DEFINITION["parameters"]
    assert set(params["required"]) == {"name", "cron", "goal"}
    # model + timezone are optional (have defaults).
    props = params["properties"]
    assert props["model"]["default"] == ""
    assert props["timezone"]["default"] == "UTC"
