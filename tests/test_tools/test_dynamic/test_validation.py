"""Regression for Fix 1a — the blocking ``ruff`` subprocess runs OFF the event loop.

``validate_tool_code`` (async) calls ``_lint_code`` (sync, shells out to
``ruff`` with a timeout up to 30s). Before Fix 1a the call was synchronous, so a
generated-tool lint stalled every concurrent task on the worker event loop for
the duration of the subprocess (the lint runs in-process on the worker, not in
the no-DinD runner). The fix wraps it in ``asyncio.to_thread``.

This test monkeypatches ``_lint_code`` with a stub that records whether it ran in
a non-main thread, then asserts the offload happened and the result contract is
unchanged — proving the lint now runs off the loop without altering behavior.
"""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.tools.dynamic.validation import validate_tool_code


_VALID_HANDLER = "async def t() -> str:\n    return 'ok'\n"
_VALID_TEST = "assert True\n"


@pytest.mark.asyncio
class TestLintOffloaded:
    async def test_lint_runs_in_worker_thread_not_on_event_loop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}

        def _lint_stub(combined: str) -> dict[str, Any]:
            # ``asyncio.to_thread`` runs this in a default-executor worker thread,
            # NOT the main thread the event loop runs on.
            seen["off_loop"] = threading.current_thread() is not threading.main_thread()
            seen["combined"] = combined
            return {"passed": True, "issues": []}

        monkeypatch.setattr("src.tools.dynamic.validation._lint_code", _lint_stub)

        fake_safety = MagicMock()
        fake_safety.validate = AsyncMock(return_value={"passed": True, "issues": []})

        result = await validate_tool_code(
            handler_code=_VALID_HANDLER,
            test_code=_VALID_TEST,
            tool_name="t",
            safety_pipeline=fake_safety,  # type: ignore[arg-type]
        )

        # The offload IS the fix — the stub must have run off the main thread.
        assert seen.get("off_loop") is True
        # The combined source was threaded through unchanged.
        assert _VALID_HANDLER in str(seen["combined"])
        # Result contract unchanged: success carries the lint layer dict verbatim.
        assert result.passed is True
        assert result.lint_result == {"passed": True, "issues": []}
