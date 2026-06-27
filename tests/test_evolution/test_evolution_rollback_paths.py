"""Rollback paths in SelfEvolutionEngine (G1 invariant + post-deploy smoke).

These complement ``test_engine.py`` (which covers the analyze→deploy happy path
and the sandbox/ab rejection *before* deploy) by exercising the post-deploy
safety nets that REVERT a deployed CODE mutation:

* ``_verify_invariants`` — a graph-invariant FAILURE over the deployed shadow
  repo (broken ``task_graph`` import via ``imports_clean``, or a removed
  ``AgentState`` key) trips ``rollback_deployment`` and the cycle reports
  ``rolled_back``.
* a passing invariant lets the cycle proceed to the post-deploy smoke (no
  rollback).
* a post-deploy SMOKE failure (a CODE mutation that re-runs and raises in the
  sandbox after deploy) also trips ``rollback_deployment``.
* ``rollback_deployment`` is a faithful no-op-when-safe guard: a deployment
  with no ``pre_deploy_hash`` never calls ``git_tracker.rollback``, and a
  tracker whose ``rollback`` raises is reported ``rolled_back=False`` without
  re-raising.
* the whole ``_verify_invariants`` call is FAIL-OPEN: a verifier exception is
  swallowed and reported ``passed`` (never aborts the cycle) — distinct from
  the ``test_invariants.TestEngineInvariantHook`` test, which patches
  ``verify_graph_invariants`` to raise; here we prove the fail-open fires from
  inside a full ``run_cycle`` (the deployed mutation is NOT rolled back because
  the verifier silently passed).

The shadow-repo plant pattern (``_write_repo_with_live_graph`` +
``SimpleNamespace(repo_dir=...)``) is reused from ``test_invariants.py``. No
src/ file is modified.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.evolution.engine import SelfEvolutionEngine
from src.graph.enums import MutationType
from src.sandbox.executor import SandboxResult

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ─── shared helpers ──────────────────────────────────────────────────────


def _write_repo_with_live_graph(tmp_path: Path) -> Path:
    """Plant a repo whose ``src/`` is a full copy of the live source tree.

    Reused verbatim from ``test_invariants.py``: a full copy (not just the
    three graph files) is required because ``imports_clean`` spawns a
    subprocess that imports the real ``src.graph.task_graph``, which
    transitively resolves the whole orchestration stack.
    """
    root = tmp_path / "repo"
    root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        _REPO_ROOT / "src",
        root / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    return root


def _safety_pass() -> MagicMock:
    mock_safety = MagicMock()
    mock_safety.validate = AsyncMock(return_value={"passed": True, "layers": {}})
    return mock_safety


def _sandbox_result(success: bool, **kw: object) -> SandboxResult:
    return SandboxResult(
        success=success,
        exit_code=int(kw.get("exit_code", 0)),
        stdout=str(kw.get("stdout", "ok")),
        stderr=str(kw.get("stderr", "")),
        duration_seconds=float(kw.get("duration_seconds", 0.1)),
        memory_mb=None,
        timed_out=bool(kw.get("timed_out", False)),
    )


def _tracker_with_repo(repo: Path) -> MagicMock:
    """A GitTracker-shaped mock bound to a planted shadow ``repo_dir``.

    The engine reads ``getattr(git_tracker, "repo_dir", None)`` so a
    SimpleNamespace would also work; a MagicMock keeps the apply/snapshot/
    rollback call recording for the assertions.
    """
    tracker = MagicMock()
    tracker.repo_dir = repo
    tracker.apply_mutation = AsyncMock(return_value=None)
    tracker.snapshot = AsyncMock(return_value="postdeploy1230000")
    tracker.get_current_hash = AsyncMock(return_value="predeploy000000")
    tracker.get_diff = AsyncMock(return_value="diff blob")
    tracker.rollback = AsyncMock(return_value=True)
    return tracker


def _engine_with_code_mutation(
    engine: SelfEvolutionEngine, mutated_content: str = "print('new code')"
) -> None:
    """Patch ``generate`` so the cycle emits a CODE mutation (executable →
    routes through sandbox/ab/post-deploy smoke + invariants)."""
    patch.object(
        engine,
        "generate",
        new_callable=AsyncMock,
        return_value={
            "mutation_type": MutationType.CODE,
            "description": "test code mutation",
            "original_content": "old",
            "mutated_content": mutated_content,
            "target_path": "test.py",
            "priority": "high",
            "rationale": "testing rollback paths",
            "model_used": None,
            "tokens_used": 0,
        },
    ).start()


# ─── _verify_invariants: invariant failure vs success ─────────────────────


class TestVerifyInvariantsRollbackDecision:
    """``_verify_invariants`` returns ``failed=True`` for a graph-breaking
    mutation and ``failed=False`` for a clean one — the gate the run_cycle
    rollback branch reads."""

    @pytest.mark.asyncio
    async def test_broken_import_fails_and_reports_reason(self, tmp_path: Path) -> None:
        repo = _write_repo_with_live_graph(tmp_path)
        # A task_graph that imports a NON-EXISTENT submodule. ``compile()``
        # misses this; ``imports_clean`` surfaces it (ModuleNotFoundError).
        (repo / "src" / "graph" / "task_graph.py").write_text(
            "from src.graph.does_not_exist import missing_symbol\n",
            encoding="utf-8",
        )
        tracker = SimpleNamespace(repo_dir=repo)

        result, failed = await SelfEvolutionEngine()._verify_invariants(tracker)

        assert failed is True
        assert result["passed"] is False
        assert "import failed" in result["reason"]
        assert "does_not_exist" in result["reason"]

    @pytest.mark.asyncio
    async def test_removed_state_key_fails_schema(self, tmp_path: Path) -> None:
        repo = _write_repo_with_live_graph(tmp_path)
        # Rewrite state.py dropping a baseline key the live graph reads.
        (repo / "src" / "graph" / "state.py").write_text(
            "from __future__ import annotations\n"
            "import operator\n"
            "from typing import Annotated, TypedDict\n\n"
            "class AgentState(TypedDict, total=False):\n"
            "    goal: str\n",
            encoding="utf-8",
        )
        tracker = SimpleNamespace(repo_dir=repo)

        result, failed = await SelfEvolutionEngine()._verify_invariants(tracker)

        assert failed is True
        assert "removed baseline AgentState key" in result["reason"]

    @pytest.mark.asyncio
    async def test_clean_repo_passes(self, tmp_path: Path) -> None:
        """A clean copy of the live graph is invariant-clean — the cycle must
        NOT roll back a deploy over it."""
        tracker = SimpleNamespace(repo_dir=_write_repo_with_live_graph(tmp_path))

        result, failed = await SelfEvolutionEngine()._verify_invariants(tracker)

        assert failed is False
        assert result["passed"] is True
        assert result["reason"] == "invariants hold"


# ─── run_cycle: invariant failure → rollback_deployment → rolled_back ─────


class TestInvariantFailureRollsBackInCycle:
    """End-to-end through ``run_cycle``: a CODE mutation that deploys, then
    trips an invariant, reverts the shadow repo and reports ``rolled_back``."""

    @pytest.mark.asyncio
    async def test_broken_import_rolls_back(
        self, tmp_path: Path
    ) -> None:
        repo = _write_repo_with_live_graph(tmp_path)
        # Pre-break the shadow repo's import graph so the post-deploy invariant
        # verify over the *deployed* snapshot fails. The deploy applies the
        # mutation to a different file (test.py), so this break survives deploy.
        (repo / "src" / "graph" / "task_graph.py").write_text(
            "from src.graph.does_not_exist import missing_symbol\n",
            encoding="utf-8",
        )
        tracker = _tracker_with_repo(repo)
        sandbox = MagicMock()
        sandbox.execute_code = AsyncMock(return_value=_sandbox_result(True))

        engine = SelfEvolutionEngine(safety_pipeline=_safety_pass())
        _engine_with_code_mutation(engine)
        try:
            result = await engine.run_cycle(
                execution_history=[],
                failure_patterns=["test"],
                sandbox=sandbox,
                git_tracker=tracker,
            )
        finally:
            patch.stopall()

        # The invariant failure short-circuits the post-deploy smoke and rolls
        # the shadow repo back to pre_deploy_hash.
        assert result["status"] == "rolled_back"
        assert result["deployed"] is False
        assert result["rollback"]["rolled_back"] is True
        tracker.rollback.assert_awaited_once_with("predeploy000000")
        assert result["smoke_result"]["passed"] is False
        assert "graph-invariant failed" in result["smoke_result"]["reason"]

    @pytest.mark.asyncio
    async def test_clean_invariant_proceeds_no_rollback(
        self, tmp_path: Path
    ) -> None:
        """A CODE mutation deployed over a CLEAN shadow repo passes invariants,
        runs the post-deploy smoke, and stays deployed (no rollback call)."""
        tracker = _tracker_with_repo(_write_repo_with_live_graph(tmp_path))
        sandbox = MagicMock()
        sandbox.execute_code = AsyncMock(return_value=_sandbox_result(True))

        engine = SelfEvolutionEngine(safety_pipeline=_safety_pass())
        _engine_with_code_mutation(engine)
        try:
            result = await engine.run_cycle(
                execution_history=[],
                failure_patterns=["test"],
                sandbox=sandbox,
                git_tracker=tracker,
            )
        finally:
            patch.stopall()

        assert result["status"] == "deployed"
        assert result["deployed"] is True
        assert result["rollback"] == {}
        tracker.rollback.assert_not_awaited()


class TestPostDeploySmokeFailureRollsBack:
    """When invariants PASS but the post-deploy smoke re-run fails, the cycle
    rolls back via the same ``rollback_deployment`` path."""

    @pytest.mark.asyncio
    async def test_smoke_failure_rolls_back(
        self, tmp_path: Path
    ) -> None:
        tracker = _tracker_with_repo(_write_repo_with_live_graph(tmp_path))
        sandbox = MagicMock()
        # sandbox_test + ab_test pass; post_deploy_verify (3rd call) fails.
        sandbox.execute_code = AsyncMock(
            side_effect=[
                _sandbox_result(True),   # sandbox_test
                _sandbox_result(True),   # ab_test control
                _sandbox_result(True),   # ab_test treatment
                _sandbox_result(False, exit_code=1, stderr="boom"),  # post-deploy
            ]
        )

        engine = SelfEvolutionEngine(safety_pipeline=_safety_pass())
        _engine_with_code_mutation(engine)
        try:
            result = await engine.run_cycle(
                execution_history=[],
                failure_patterns=["test"],
                sandbox=sandbox,
                git_tracker=tracker,
            )
        finally:
            patch.stopall()

        assert result["status"] == "rolled_back"
        assert result["deployed"] is False
        assert result["rollback"]["rolled_back"] is True
        tracker.rollback.assert_awaited_once_with("predeploy000000")
        assert "post-deploy smoke failed" in result["smoke_result"]["reason"]


# ─── _verify_invariants is fail-open from inside a full cycle ─────────────


class TestVerifyInvariantsFailsOpenInCycle:
    """A verifier EXCEPTION (not a real invariant failure) must never abort an
    evolution cycle: ``_verify_invariants`` swallows it and reports ``passed``,
    so the deployed mutation is NOT rolled back. This is the load-bearing
    resilience contract — distinct from a genuine invariant break above."""

    @pytest.mark.asyncio
    async def test_verifier_exception_does_not_roll_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src.evolution.invariants as invariants

        def _boom(*_a: object, **_kw: object) -> None:
            raise RuntimeError("forced verifier crash")

        monkeypatch.setattr(invariants, "verify_graph_invariants", _boom)

        tracker = _tracker_with_repo(_write_repo_with_live_graph(tmp_path))
        sandbox = MagicMock()
        sandbox.execute_code = AsyncMock(return_value=_sandbox_result(True))

        engine = SelfEvolutionEngine(safety_pipeline=_safety_pass())
        _engine_with_code_mutation(engine)
        try:
            result = await engine.run_cycle(
                execution_history=[],
                failure_patterns=["test"],
                sandbox=sandbox,
                git_tracker=tracker,
            )
        finally:
            patch.stopall()

        # The verifier bug was swallowed → no invariant failure → smoke passed
        # → deployed, NOT rolled back.
        assert result["status"] == "deployed"
        assert result["deployed"] is True
        tracker.rollback.assert_not_awaited()


# ─── rollback_deployment: direct unit behavior ────────────────────────────


class TestRollbackDeploymentDirect:
    """``rollback_deployment`` is a safe guard: it captures the reverted diff
    BEFORE resetting, no-ops without a pre_deploy_hash, and reports
    ``rolled_back=False`` (never re-raises) when the tracker raises."""

    @pytest.mark.asyncio
    async def test_no_pre_deploy_hash_is_a_safe_noop(self) -> None:
        tracker = MagicMock()
        tracker.rollback = AsyncMock(return_value=True)
        tracker.get_diff = AsyncMock(return_value="")
        result = await SelfEvolutionEngine().rollback_deployment(
            {"pre_deploy_hash": None}, tracker
        )
        assert result["rolled_back"] is False
        assert "no pre_deploy_hash" in result["reason"]
        tracker.rollback.assert_not_awaited()
        tracker.get_diff.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_captures_diff_then_rolls_back(self) -> None:
        tracker = MagicMock()
        tracker.get_diff = AsyncMock(return_value="BIG DIFF")
        tracker.rollback = AsyncMock(return_value=True)

        result = await SelfEvolutionEngine().rollback_deployment(
            {"pre_deploy_hash": "abc12345"}, tracker
        )

        assert result["rolled_back"] is True
        assert result["pre_deploy_hash"] == "abc12345"
        assert result["reverted_diff"] == "BIG DIFF"
        # get_diff is captured BEFORE rollback (ordering asserted via await counts).
        tracker.get_diff.assert_awaited_once_with(since_hash="abc12345")
        tracker.rollback.assert_awaited_once_with("abc12345")

    @pytest.mark.asyncio
    async def test_tracker_rollback_raising_is_swallowed(self) -> None:
        tracker = MagicMock()
        tracker.get_diff = AsyncMock(side_effect=RuntimeError("diff boom"))
        tracker.rollback = AsyncMock(side_effect=RuntimeError("reset boom"))

        result = await SelfEvolutionEngine().rollback_deployment(
            {"pre_deploy_hash": "abc12345"}, tracker
        )

        # Never re-raises — reports rolled_back=False with the reason.
        assert result["rolled_back"] is False
        assert "reset boom" in result["reason"]
        assert result["pre_deploy_hash"] == "abc12345"
