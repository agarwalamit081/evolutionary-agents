"""Runner-level regression for the disk-contamination fix (Vector B).

``execute_run`` clears THIS run's ``results/<run_id>/`` subdir at the start of a
FRESH attempt (so a re-enqueued run_id does not inherit a prior attempt's
deliverables) — but MUST NOT on resume (the subdir deliverables are part of the
resumable run state). The clean sits in the non-resume branch, beside the prior
checkpoint clear.

These run the REAL ``execute_run`` with the heavy internals patched to no-ops
(gateway=None → heuristic path; tools=None → skips dynamic-tool load; a fake
compiled graph whose ``ainvoke`` returns a terminal state). The decisive
assertion is behavioral, not a call-count spy: a stale deliverable seeded before
the run is GONE after a fresh run (clean fired) and SURVIVES a resume (clean did
not fire) — which no spy could fake, because the run's own end-of-run summary
write re-creates the subdir either way; only the stale file's absence/presence
distinguishes the two paths.
"""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

import src.runner as runner


# -- no-op stand-ins for the heavy dependency-instantiation helpers ------------


async def _async_none(*_a: Any, **_k: Any) -> None:
    """Async no-op returning None (gateway/memory/tools/checkpointer → None)."""
    return None


async def _resumable_tuple(*_a: Any, **_k: Any) -> SimpleNamespace:
    """A checkpoint tuple stand-in so the resume branch proceeds without a DB.

    ``channel_values`` has no ``current_goal`` → goal_text stays as passed.
    """
    return SimpleNamespace(checkpoint={"channel_values": {}})


async def _fake_ainvoke(_state: Any, *, config: Any = None) -> dict[str, Any]:
    """The compiled graph's terminal state — no real graph/LLM fires."""
    return {
        "final_output": "ok",
        "is_complete": True,
        "iteration_count": 1,
        "cost_records": [],
        "total_tokens_used": 0,
    }


def _fake_compiled() -> SimpleNamespace:
    return SimpleNamespace(ainvoke=_fake_ainvoke)


def _wire_run_nops(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> Any:
    """Patch the run's heavy internals to no-ops + a tmp results root.

    Returns the fake settings pinned via ``src.config.get_settings`` so
    ``_subdir_active`` / ``results_root`` / ``normalize`` all resolve under
    ``tmp_path`` with per-run subfoldering ON.
    """
    fake_settings = SimpleNamespace(
        agent=SimpleNamespace(
            results_root=str(tmp_path / "results"),
            workspace_root=str(tmp_path / "workspace"),
            results_per_run_subdir=True,
            max_iterations=12,
        ),
        langsmith=SimpleNamespace(is_configured=False),
        redis=SimpleNamespace(redis_url="redis://localhost:6380"),
        eval=SimpleNamespace(eval_enabled=False, eval_enforce=False),
        logging=SimpleNamespace(),
    )
    # execute_run resolves ``src.config.get_settings`` (the __init__ re-export),
    # while ``src.tools._paths`` resolves ``src.config.settings.get_settings``
    # (its captured ``_settings`` module). Patch BOTH so the run AND the path
    # resolver/cleaner agree on the tmp results root + subfoldering flag.
    monkeypatch.setattr("src.config.get_settings", lambda: fake_settings)
    monkeypatch.setattr("src.config.settings.get_settings", lambda: fake_settings)
    # gateway=None → heuristic path, skips the Redis-cache + cost-tracker blocks.
    monkeypatch.setattr(runner, "_create_gateway", lambda *_a, **_k: None)
    monkeypatch.setattr(runner, "_async_create_memory_manager", _async_none)
    # tools=None → skips the dynamic-tool load.
    monkeypatch.setattr(runner, "_create_tool_registry", lambda: None)
    monkeypatch.setattr(runner, "_load_sub_agents", _async_none)
    # checkpointer=None (fresh) → skips adelete_thread; clean branch still enters.
    monkeypatch.setattr(runner, "_create_checkpointer", _async_none)
    # compile → fake compiled graph (no real graph build / ainvoke LLM loop).
    monkeypatch.setattr(
        "src.graph.task_graph.compile_task_graph", lambda **_k: _fake_compiled()
    )
    # Per-query log sink — avoid filesystem side effects in the test cwd.
    monkeypatch.setattr("src.observability.logging.add_query_log_sink", lambda *_a, **_k: None)
    return fake_settings


@pytest.fixture(autouse=True)
def _reset_active_run_id() -> Iterator[None]:
    """execute_run binds the _active_run_id contextvar; clear it between tests."""
    from src.tools._paths import set_active_run_id

    set_active_run_id(None)
    yield
    set_active_run_id(None)


class TestExecuteRunSubdirClean:
    async def test_cleans_subdir_on_fresh_run(self, monkeypatch, tmp_path) -> None:
        """Fresh (non-resume) attempt + subfoldering on → clean fires: a stale
        deliverable seeded under ``results/<run_id>/`` is removed BEFORE the run
        re-creates the subdir for its own summary write."""
        _wire_run_nops(monkeypatch, tmp_path)
        results = tmp_path / "results"
        stale = results / "fresh-test" / "stale_deliverable.csv"
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_text("prior attempt")
        (results / "sibling-run" / "keep.csv").parent.mkdir(parents=True, exist_ok=True)
        (results / "sibling-run" / "keep.csv").write_text("another run")

        await runner.execute_run(
            "goal", 12, no_evolution=True, run_id="fresh-test"
        )

        # The stale deliverable is gone — clean fired and cleared the subdir.
        assert not stale.exists()
        # The run's end-of-run summary write re-created the subdir for its own
        # output, so the dir is back but ONLY the run's artifacts are in it.
        assert (results / "fresh-test").exists()
        assert not (results / "fresh-test" / "stale_deliverable.csv").exists()
        # Sibling run + the shared results root are untouched (scoped to run_id).
        assert (results / "sibling-run" / "keep.csv").exists()
        assert results.exists()

    async def test_does_not_clean_on_resume(self, monkeypatch, tmp_path) -> None:
        """Resume MUST preserve the subdir — its deliverables are part of the
        resumable run state. The clean branch lives in the non-resume ``else``,
        so a resume attempt leaves a seeded stale deliverable intact."""
        _wire_run_nops(monkeypatch, tmp_path)
        # Resume takes the ``if resume:`` branch → _require_resumable_checkpoint
        # is consulted; stub it so no real checkpointer/DB is needed.
        monkeypatch.setattr(runner, "_require_resumable_checkpoint", _resumable_tuple)

        results = tmp_path / "results"
        deliverable = results / "resume-test" / "partial_work.csv"
        deliverable.parent.mkdir(parents=True, exist_ok=True)
        deliverable.write_text("resumable state")

        await runner.execute_run(
            "goal", 12, no_evolution=True, run_id="resume-test", resume=True
        )

        # The resumable deliverable SURVIVES — clean did not fire on resume.
        assert deliverable.exists()
        assert deliverable.read_text() == "resumable state"
