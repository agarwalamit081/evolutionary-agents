"""Regression for the deployed worker→``execute_run`` run_id seam (#256).

``default_agent_executor`` is the ONLY thing between a queued ``RunJob`` and
``src.runner.execute_run``. Per-run results subfoldering
(``results_root/<run_id>/…``) depends entirely on ``run_id`` reaching
``execute_run``, which binds the ``_active_run_id`` contextvar (Phase 7). If the
executor ever dropped ``run_id`` — or ``origin="api"`` (so the run's checkpoint
thread is ``api-{run_id}`` and never collides with a CLI run) — deployed runs
would silently write flat: the exact #256 symptom ("no subdir in the deployed
container"). These lock the forwarding contract.

``execute_run`` is replaced with a recorder so no live graph/LLM fires. The real
binding+routing is unit-covered by ``tests/test_tools/test_paths.py``
(``set_active_run_id`` → ``normalize`` → ``results_root/<run_id>/``); this file
closes the one untested link in that chain — the executor forwarding ``run_id``.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.worker import executors
from src.worker.schema import RunJob


def _recorder(sink: dict[str, Any]) -> Any:
    """Build an async ``execute_run`` stand-in that captures its kwargs."""

    async def fake_execute_run(**kwargs: Any) -> dict[str, Any]:
        sink.update(kwargs)
        return {"final_output": "ok", "is_complete": True, "iteration_count": 1}

    return fake_execute_run


class TestDefaultAgentExecutorSeam:
    """The executor must forward the run identity + deployed-path flags verbatim."""

    @pytest.mark.asyncio
    async def test_forwards_run_id_to_execute_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """run_id is the key the whole per-run subfolder path keys on (#256)."""
        captured: dict[str, Any] = {}
        monkeypatch.setattr(executors, "execute_run", _recorder(captured))

        await executors.default_agent_executor(RunJob(run_id="deploy-123", goal="g"))

        assert captured.get("run_id") == "deploy-123"

    @pytest.mark.asyncio
    async def test_forwards_origin_api_and_resume_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Deployed runs use the api- thread prefix and never auto-resume."""
        captured: dict[str, Any] = {}
        monkeypatch.setattr(executors, "execute_run", _recorder(captured))

        await executors.default_agent_executor(RunJob(run_id="r1", goal="g"))

        assert captured.get("origin") == "api"
        assert captured.get("resume") is False

    @pytest.mark.asyncio
    async def test_forwards_job_fields_verbatim(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """goal / max_iterations / no_evolution / model thread straight through."""
        captured: dict[str, Any] = {}
        monkeypatch.setattr(executors, "execute_run", _recorder(captured))

        await executors.default_agent_executor(
            RunJob(
                run_id="r2",
                goal="do the thing",
                max_iterations=7,
                no_evolution=True,
                model="glm-4.7-flash",
            )
        )

        assert captured["goal_text"] == "do the thing"
        assert captured["max_iterations"] == 7
        assert captured["no_evolution"] is True
        assert captured["model"] == "glm-4.7-flash"

    @pytest.mark.asyncio
    async def test_forwards_results_per_run_subdir_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The battery's flat-root opt-out (#575) reaches ``execute_run``."""
        captured: dict[str, Any] = {}
        monkeypatch.setattr(executors, "execute_run", _recorder(captured))

        await executors.default_agent_executor(
            RunJob(run_id="battery04_q02-20260630", goal="g", results_per_run_subdir=False)
        )

        assert captured.get("results_per_run_subdir") is False

    @pytest.mark.asyncio
    async def test_clears_flat_subdirs_before_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A battery job's flat write-dir is cleared on the worker BEFORE the run
        (#575) — the worker mounts the results volume (the scheduler does not)."""
        captured: dict[str, Any] = {}
        monkeypatch.setattr(executors, "execute_run", _recorder(captured))
        cleared: list[list[str]] = []
        monkeypatch.setattr(
            executors,
            "clear_flat_results_subdirs",
            lambda subs: (cleared.append(subs), 0)[1],  # sync; record + return 0
        )

        await executors.default_agent_executor(
            RunJob(run_id="battery04_q02-20260630", goal="g", clear_flat_subdirs=["q02"])
        )

        assert cleared == [["q02"]]  # cleared exactly once, with the job's dirs

    @pytest.mark.asyncio
    async def test_skips_clear_when_no_flat_subdirs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-battery run (default empty list) never invokes the clear."""
        monkeypatch.setattr(executors, "execute_run", _recorder({}))
        cleared: list[list[str]] = []
        monkeypatch.setattr(
            executors,
            "clear_flat_results_subdirs",
            lambda subs: (cleared.append(subs), 0)[1],
        )

        await executors.default_agent_executor(RunJob(run_id="r1", goal="g"))

        assert cleared == []
