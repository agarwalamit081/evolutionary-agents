"""Unit tests for scripts/run_e2e_validation.py — the parallel validation runner.

These are fast, hermetic tests of the runner's *logic* (settings isolation under
concurrency, failure-metrics shape, gather ordering) — NOT real agent runs. The
graph, gateway, memory, and DB are all stubbed so no provider key or
infrastructure is required.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# Load the standalone script (it lives in scripts/, not the src package).
_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_e2e_validation.py"
_spec = importlib.util.spec_from_file_location("e2e_run_validation", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
e2e = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e2e)


# ─── Fixtures & stubs ────────────────────────────────────────────────


@pytest.fixture
def isolated_settings() -> Any:
    """A deep-copied Settings so tests never mutate the lru_cache singleton."""
    from src.config import get_settings

    return get_settings().model_copy(deep=True)


async def _async_none(*_args: Any, **_kwargs: Any) -> None:
    """Async no-op for stubbing awaited helpers."""
    return None


def _fake_gateway() -> MagicMock:
    """A gateway stub exposing the surface _run_single_query touches."""
    gw = MagicMock()
    gw.set_cache = MagicMock()
    gw.get_cost_records = MagicMock(return_value=[])
    return gw


def _fake_compiled_graph() -> MagicMock:
    """A compiled-graph stub whose ainvoke returns a minimal terminal state."""
    compiled = MagicMock()
    compiled.ainvoke = AsyncMock(
        return_value={
            "is_complete": True,
            "iteration_count": 1,
            "final_output": "stubbed output",
        }
    )
    return compiled


def _patch_infra(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub every external dependency so a query runs without LLM/DB/Redis.

    The one real call kept is ``initial_state`` (pure dict builder). Everything
    that opens a socket or runs the graph is replaced.
    """
    monkeypatch.setattr(
        "src.graph.task_graph.compile_task_graph",
        lambda **_kw: _fake_compiled_graph(),
    )
    monkeypatch.setattr("src.observability.logging.add_query_log_sink", lambda *a, **k: None)
    monkeypatch.setattr(e2e, "_create_gateway", lambda _s: _fake_gateway())
    monkeypatch.setattr(e2e, "_create_memory_manager", _async_none)
    monkeypatch.setattr(e2e, "_create_tool_registry", lambda: None)
    monkeypatch.setattr(e2e, "_create_checkpointer", _async_none)
    monkeypatch.setattr(e2e, "_load_sub_agents", _async_none)


# ─── Settings isolation under concurrency ────────────────────────────


class TestSettingsIsolation:
    """The shared settings object must never be mutated by a query.

    This is the safety guarantee that makes the parallel (gather) path correct:
    without it, N concurrent queries race on ``agent.results_root`` and clobber
    each other's artifact directories.
    """

    @pytest.mark.asyncio
    async def test_single_query_does_not_mutate_shared_results_root(
        self, isolated_settings: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """One query redirects artifacts into its subfolder, leaving shared root intact."""
        shared_root = tmp_path / "shared_results"
        isolated_settings.agent.results_root = str(shared_root)
        original_root = isolated_settings.agent.results_root

        _patch_infra(monkeypatch)

        metrics = await e2e._run_single_query(e2e.QUERIES[0], isolated_settings, max_iterations=18)

        # The shared settings object is untouched — the query mutated a copy.
        assert isolated_settings.agent.results_root == original_root
        # Artifacts were redirected into the per-query subfolder.
        assert (shared_root / e2e.QUERIES[0]["id"]).exists()
        assert metrics["query_id"] == e2e.QUERIES[0]["id"]
        assert metrics["complete"] is True

    @pytest.mark.asyncio
    async def test_concurrent_queries_do_not_cross_contaminate_roots(
        self, isolated_settings: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Two queries run via gather each land in their own subfolder, in order."""
        import asyncio

        shared_root = tmp_path / "shared_results"
        isolated_settings.agent.results_root = str(shared_root)
        original_root = isolated_settings.agent.results_root

        _patch_infra(monkeypatch)

        # Mirror run_validation's bounded-gather shape (order-preserving).
        two = e2e.QUERIES[:2]
        results = await asyncio.gather(
            *[e2e._run_single_query(q, isolated_settings, max_iterations=18) for q in two]
        )

        # gather preserves QUERIES order.
        assert [r["query_id"] for r in results] == [q["id"] for q in two]
        # Shared settings untouched despite concurrent mutation of per-query copies.
        assert isolated_settings.agent.results_root == original_root
        # Each query got its own isolated artifact directory.
        assert (shared_root / two[0]["id"]).exists()
        assert (shared_root / two[1]["id"]).exists()


# ─── Failure metrics ─────────────────────────────────────────────────


class TestFailureMetrics:
    """_failure_metrics must match the success-metrics shape (no missing keys)."""

    def test_shape_matches_success_metrics_keys(self) -> None:
        """Every key present in a success record is present (zeroed) on failure."""
        fm = e2e._failure_metrics(e2e.QUERIES[0], RuntimeError("boom"), max_iterations=18)

        required = {
            "query_id", "query_name", "query_text", "complete", "iterations",
            "max_iterations", "duration_seconds", "total_tokens", "total_cost",
            "model_usage", "cost_records_count", "sub_agents_spawned",
            "sub_agent_details", "delegations_succeeded", "delegations_total",
            "tools_called_count", "tools_created_count", "tool_creation_details",
            "tool_results_count", "fold_count", "fold_details", "errors",
            "results_file", "final_output_preview",
        }
        assert required.issubset(fm.keys()), f"missing: {required - fm.keys()}"
        assert fm["complete"] is False
        assert fm["max_iterations"] == 18
        assert fm["errors"] == ["FATAL: boom"]
        assert fm["results_file"] is None
        assert fm["total_cost"] == 0.0

    def test_exception_message_preserved(self) -> None:
        """The triggering exception's message survives into the error list."""
        fm = e2e._failure_metrics(e2e.QUERIES[1], ValueError("bad input"), max_iterations=18)
        assert any("bad input" in e for e in fm["errors"])
