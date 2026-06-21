"""Phase 6 — cross-process ``--resume``.

Covers the testable surface without a live agent run:
* ``_thread_id_for_run`` derives a stable, run_id-keyed thread_id (so a later
  ``--resume <run_id>`` reuses it) and a process-local fallback otherwise.
* ``_require_resumable_checkpoint`` refuses cleanly (no checkpointer / no
  checkpoint / missing run_id) and returns the tuple when a checkpoint exists.
* the ``--resume`` click option exists, parses, threads ``resume=True`` +
  ``run_id`` into ``execute_run`` without requiring ``--goal``, and surfaces a
  clean CLI error (exit 1) when the checkpoint can't be resumed.

Full live resume (real AsyncPostgresSaver + a halted run) is validated in the
Phase-10 battery.
"""

from __future__ import annotations

from typing import Any

import click.testing
import pytest

import main as main_mod
import src.runner as runner


class TestThreadIdForRun:
    def test_run_id_keyed_thread_is_stable(self) -> None:
        """A run_id maps to a deterministic, resumable thread_id."""
        assert runner._thread_id_for_run("q01", "any goal") == "cli-q01"
        # Stable across calls (no pid/obj-id entropy).
        assert runner._thread_id_for_run("q01", "different goal") == "cli-q01"

    def test_no_run_id_falls_back_to_process_local_key(self) -> None:
        """Without a run_id the thread_id is process-local (non-resumable)."""
        tid = runner._thread_id_for_run(None, "some goal")
        assert tid.startswith("cli-")
        assert tid != "cli-q01"  # not accidentally a run_id mapping


class _FakeCheckpointer:
    """Minimal checkpointer double exposing aget_tuple / adelete_thread."""

    def __init__(self, tuple_: Any = None) -> None:
        self._tuple = tuple_
        self.deleted: list[str] = []

    async def aget_tuple(self, _config: Any) -> Any:
        return self._tuple

    async def adelete_thread(self, thread_id: str) -> None:
        self.deleted.append(thread_id)


class TestRequireResumableCheckpoint:
    @pytest.mark.asyncio
    async def test_returns_tuple_when_checkpoint_exists(self) -> None:
        cp = _FakeCheckpointer(tuple_={"checkpoint": {"channel_values": {}}})
        result = await runner._require_resumable_checkpoint(cp, "cli-q01", "q01")
        assert result == {"checkpoint": {"channel_values": {}}}

    @pytest.mark.asyncio
    async def test_raises_when_no_checkpoint(self) -> None:
        cp = _FakeCheckpointer(tuple_=None)
        with pytest.raises(RuntimeError, match="no checkpoint found"):
            await runner._require_resumable_checkpoint(cp, "cli-q01", "q01")

    @pytest.mark.asyncio
    async def test_raises_when_no_checkpointer(self) -> None:
        with pytest.raises(RuntimeError, match="no checkpointer"):
            await runner._require_resumable_checkpoint(None, "cli-q01", "q01")

    @pytest.mark.asyncio
    async def test_raises_when_no_run_id(self) -> None:
        cp = _FakeCheckpointer(tuple_=object())
        with pytest.raises(ValueError, match="--run-id"):
            await runner._require_resumable_checkpoint(cp, "cli-None", None)


class TestResumeCliOption:
    def test_resume_option_default_is_none(self) -> None:
        """The click command exposes --resume, defaulting to None."""
        cmd = main_mod.main
        param = next(p for p in cmd.params if p.name == "resume_run_id")
        assert param.default is None

    def test_resume_threads_resume_flag_and_run_id_without_goal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--resume q01 (no --goal) does NOT exit, and calls execute_run with
        resume=True + run_id=q01 + empty goal_text."""
        captured: dict[str, Any] = {}

        async def _fake_run_agent(*args: Any, **kwargs: Any) -> dict[str, Any]:
            captured["args"] = args
            captured["kwargs"] = kwargs
            return {"final_output": "ok", "iteration_count": 1, "is_complete": True}

        monkeypatch.setattr(main_mod, "execute_run", _fake_run_agent)
        runner = click.testing.CliRunner()
        result = runner.invoke(main_mod.main, ["--resume", "q01"])

        assert result.exit_code == 0, result.output
        # goal_text (args[0]) empty, run_id (args[3]) == q01, resume kwarg True.
        assert captured["args"][0] == ""
        assert captured["args"][3] == "q01"
        assert captured["kwargs"]["resume"] is True
        assert "Resume: run_id=q01" in result.output

    def test_resume_surfaces_clean_error_when_unresumable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A RuntimeError from execute_run (no checkpoint) → clean exit 1."""

        async def _fake_run_agent(*args: Any, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("no checkpoint found for run_id=q01")

        monkeypatch.setattr(main_mod, "execute_run", _fake_run_agent)
        runner = click.testing.CliRunner()
        result = runner.invoke(main_mod.main, ["--resume", "q01"])

        assert result.exit_code == 1
        assert "cannot resume" in result.output
        assert "no checkpoint found" in result.output
