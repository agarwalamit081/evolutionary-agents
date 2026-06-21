"""ToolMetricsRecorder (M4): roll_rate math, gating, non-fatal guarantee.

The incremental-mean math is exercised directly (``roll_rate`` is the pure form
of the server-side UPDATE). The recorder's DB path is verified with a capturing
fake session — never a real DB — and the CostTracker-resilience guarantee (a
poisoned write logs and returns rather than re-raising) is asserted explicitly.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from src.tools.metrics import ToolMetricsRecorder, roll_rate


class _FakeResult:
    def __init__(self, rows: list[Any]) -> None:
        self._rows = rows

    def all(self) -> list[Any]:
        return self._rows


class _FakeSession:
    """Captures every execute() statement; never touches a DB."""

    def __init__(self, rows: list[Any] | None = None) -> None:
        self.stmts: list[Any] = []
        self._rows = rows or []

    async def execute(self, stmt: Any) -> _FakeResult:
        self.stmts.append(stmt)
        return _FakeResult(self._rows)

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, *_args: Any) -> bool:
        return False


class _RaisingSession:
    async def execute(self, _stmt: Any) -> Any:
        raise RuntimeError("db down")

    async def __aenter__(self) -> _RaisingSession:
        return self

    async def __aexit__(self, *_args: Any) -> bool:
        return False


def _enabled(enabled: bool = True) -> object:
    return SimpleNamespace(agent=SimpleNamespace(tool_metrics_enabled=enabled))


class TestRollRate:
    def test_first_sample_ignores_seed(self) -> None:
        # seed 1.0 with 0 prior calls → result is just the sample
        assert roll_rate(1.0, 0, 0.0) == pytest.approx(0.0)
        assert roll_rate(1.0, 0, 1.0) == pytest.approx(1.0)

    def test_running_mean_matches_plain_average(self) -> None:
        # mean of [1, 1, 0, 0] == 0.5, folded one sample at a time
        r = roll_rate(0.0, 0, 1.0)
        r = roll_rate(r, 1, 1.0)
        r = roll_rate(r, 2, 0.0)
        r = roll_rate(r, 3, 0.0)
        assert r == pytest.approx(0.5)

    def test_long_history_resists_single_outlier(self) -> None:
        # 99 prior successes, one failure → 0.99, not washed out
        assert roll_rate(1.0, 99, 0.0) == pytest.approx(0.99)


class TestRecorderRecord:
    @pytest.mark.asyncio
    async def test_disabled_never_opens_session(self) -> None:
        with patch("src.config.settings.get_settings", lambda: _enabled(False)), patch(
            "src.db.session.get_session"
        ) as mock_gs:
            await ToolMetricsRecorder().record("t", success=True, empty_output=False)
            mock_gs.assert_not_called()

    @pytest.mark.asyncio
    async def test_enabled_emits_update_then_insert(self) -> None:
        session = _FakeSession()
        with patch("src.config.settings.get_settings", lambda: _enabled(True)), patch(
            "src.db.session.get_session", lambda: session
        ):
            await ToolMetricsRecorder().record(
                "t", success=False, empty_output=False, run_id="r1", latency_ms=42
            )
        kinds = {type(s).__name__ for s in session.stmts}
        assert kinds == {"Update", "Insert"}
        assert len(session.stmts) == 2

    @pytest.mark.asyncio
    async def test_db_failure_is_swallowed(self) -> None:
        # Observability-only: a poisoned session must never break the tool call.
        with patch("src.config.settings.get_settings", lambda: _enabled(True)), patch(
            "src.db.session.get_session", lambda: _RaisingSession()
        ):
            await ToolMetricsRecorder().record("t", success=True, empty_output=False)  # no raise


class TestRecordResult:
    @pytest.mark.asyncio
    async def test_success_blank_is_empty_output(self) -> None:
        recorded: dict[str, Any] = {}

        class _Result:
            success = True
            output = "   "
            error = None

        async def _fake_record(_self, tool_name: str, **kw: Any) -> None:
            recorded.update(kw)

        with patch("src.config.settings.get_settings", lambda: _enabled(True)):
            with patch.object(ToolMetricsRecorder, "record", _fake_record):
                await ToolMetricsRecorder().record_result("t", _Result())
        assert recorded["success"] is True
        assert recorded["empty_output"] is True

    @pytest.mark.asyncio
    async def test_error_is_not_empty_output(self) -> None:
        recorded: dict[str, Any] = {}

        class _Result:
            success = False
            output = ""
            error = "boom"

        async def _fake_record(_self, tool_name: str, **kw: Any) -> None:
            recorded.update(kw)

        with patch("src.config.settings.get_settings", lambda: _enabled(True)):
            with patch.object(ToolMetricsRecorder, "record", _fake_record):
                await ToolMetricsRecorder().record_result("t", _Result())
        assert recorded["success"] is False
        assert recorded["empty_output"] is False
