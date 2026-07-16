"""Regression for Fix 1b — the blocking import-smoke invariant pass runs OFF
the event loop.

``SelfEvolutionEngine._verify_invariants`` (async) calls
``verify_graph_invariants`` (sync — shells out to an import-smoke subprocess
with a timeout up to 30s, plus several file-read/AST passes). Before Fix 1b the
call was synchronous, stalling the worker event loop per CODE mutation. The fix
wraps it in ``asyncio.to_thread``.

This test monkeypatches the verifier with a stub that records whether it ran in
a non-main thread, then asserts the offload happened AND the
``(result, failed)`` fails-open tuple contract is preserved.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.evolution.engine import SelfEvolutionEngine


def _make_engine() -> SelfEvolutionEngine:
    """An engine whose deps are mocks.

    ``_verify_invariants`` reads only the live ``state.py`` baseline (via
    ``Path(__file__)``, not ``self``), so mocked deps keep the test hermetic.
    """
    return SelfEvolutionEngine(  # type: ignore[arg-type]
        safety_pipeline=MagicMock(),
        gateway=MagicMock(),
        persister=MagicMock(),
    )


@pytest.mark.asyncio
class TestInvariantVerifyOffloaded:
    async def test_verify_runs_in_worker_thread_and_preserves_contract(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        seen: dict[str, Any] = {}

        def _verify_stub(repo_dir: Any, *, baseline_state_keys: Any = None) -> Any:
            seen["off_loop"] = threading.current_thread() is not threading.main_thread()
            seen["repo_dir"] = repo_dir
            seen["baseline"] = baseline_state_keys
            # A passing report: no failures, no checks.
            return SimpleNamespace(passed=True, failures=[], checks=[])

        # The engine imports verify_graph_invariants lazily inside the method, so
        # patching the module attribute is what the in-function import observes.
        monkeypatch.setattr(
            "src.evolution.invariants.verify_graph_invariants", _verify_stub
        )

        engine = _make_engine()
        tracker = MagicMock()
        tracker.repo_dir = str(tmp_path)  # truthy → the verify branch runs

        result, failed = await engine._verify_invariants(tracker)

        # The offload IS the fix — the verifier ran off the main thread.
        assert seen.get("off_loop") is True
        # The live state.py baseline was threaded through (proves extract_state_keys ran).
        assert seen.get("baseline") is not None
        # The fails-open tuple contract is preserved: a passed report ⇒ failed=False.
        assert failed is False
        assert result["passed"] is True
        assert result["reason"] == "invariants hold"

    async def test_skips_cleanly_when_no_shadow_repo(self) -> None:
        """Fails-open contract: no ``repo_dir`` ⇒ skipped (passed, not failed)."""
        engine = _make_engine()
        tracker = MagicMock()
        tracker.repo_dir = None

        result, failed = await engine._verify_invariants(tracker)

        assert failed is False
        assert result["passed"] is True
        assert "skipped" in result["reason"]
