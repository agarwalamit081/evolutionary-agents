"""Tests for src.evolution.invariants — stage-1 graph-invariant verifier (Phase 5 G1).

Four checks are pure static-AST over planted repo fixtures (no LLM, no
execution); the fifth (``imports_clean``) spawns a subprocess that imports the
mutated graph modules in isolation. The engine hook wiring
(``SelfEvolutionEngine._verify_invariants``) is exercised against planted
shadow-repo snapshots and the fail-open guard.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.evolution.engine import SelfEvolutionEngine
from src.evolution.invariants import (
    extract_state_keys,
    verify_graph_invariants,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

# ─── Planted fixtures ───────────────────────────────────────────────────

_GOOD_STATE = """from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict, total=False):
    goal: str
    messages: Annotated[list, operator.add]
    errors: Annotated[list, operator.add]
"""

_STATE_MISSING_KEY = """from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict, total=False):
    goal: str
    messages: Annotated[list, operator.add]
"""

_GOOD_ROUTERS = """from __future__ import annotations


def route_after_execute(state) -> str:
    return "reflect"


def route_after_store(state) -> str:
    return "complete"
"""

_BAD_ROUTERS_UNKNOWN_NODE = """from __future__ import annotations


def route_after_execute(state) -> str:
    return "nonexistent_node"
"""

_GOOD_TASK_GRAPH = """from __future__ import annotations


def build():
    graph = None
    graph.add_node("classify", None)
    graph.add_node("execute", None)
    graph.add_node("reflect", None)
    graph.add_edge("classify", "execute")
    return graph
"""

_BAD_TASK_GRAPH_SELF_LOOP = """from __future__ import annotations


def build():
    graph = None
    graph.add_node("execute", None)
    graph.add_edge("execute", "execute")
    return graph
"""

# A task_graph that compiles cleanly (passes the ``compiles`` check) but raises
# at IMPORT time — ``ModuleNotFoundError`` here is invisible to ``compile()``
# and only surfaces when the module is actually imported (what imports_clean
# catches). Names an attribute on a missing submodule, not a bare raise, so the
# failure is representative of a real rename/missing-symbol mutation.
_BROKEN_IMPORT_TASK_GRAPH = """from src.graph.does_not_exist import missing_symbol
"""

_BASELINE_KEYS: set[str] = {"goal", "messages", "errors"}
_CHECK_NAMES = {
    "compiles",
    "imports_clean",
    "state_schema_compatible",
    "routers_valid",
    "no_self_loops",
}


def _write_repo(
    tmp_path: Path,
    *,
    state: str = _GOOD_STATE,
    routers: str = _GOOD_ROUTERS,
    task_graph: str = _GOOD_TASK_GRAPH,
    include_graph: bool = True,
) -> Path:
    """Plant a minimal shadow-repo tree under ``tmp_path/repo/src/...``."""
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "src" / "__init__.py").write_text("", encoding="utf-8")
    if include_graph:
        gdir = root / "src" / "graph"
        gdir.mkdir(parents=True, exist_ok=True)
        (gdir / "__init__.py").write_text("", encoding="utf-8")
        (gdir / "state.py").write_text(state, encoding="utf-8")
        (gdir / "routers.py").write_text(routers, encoding="utf-8")
        (gdir / "task_graph.py").write_text(task_graph, encoding="utf-8")
    return root


def _write_repo_with_live_graph(tmp_path: Path) -> Path:
    """Plant a repo whose ``src/`` is a full COPY of the live source tree.

    The engine computes its baseline from the live state.py, so a repo whose
    state.py IS the live one is an exact superset (0 removed) → all checks
    pass. Used to validate the hook's good-repo path against real files.

    The full tree (not just the three graph files) is required because the
    ``imports_clean`` check spawns a subprocess that imports the real
    ``src.graph.task_graph``, which transitively imports the nodes package,
    config, llm gateway, memory, and tools — only a full copy resolves that
    import graph. The ~5s subprocess cost is inherent to importing the live
    orchestration stack and is expected for these real-file validation tests.
    """
    root = tmp_path / "repo"
    root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        _REPO_ROOT / "src",
        root / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    return root


# ─── verify_graph_invariants: planted fixtures ─────────────────────────


class TestVerifyGraphInvariants:
    """The five stage-1 checks against planted-good / planted-bad repos."""

    def test_good_repo_passes_all_checks(self, tmp_path: Path) -> None:
        report = verify_graph_invariants(
            _write_repo(tmp_path), baseline_state_keys=_BASELINE_KEYS
        )
        assert report.passed, [c.detail for c in report.checks]
        assert {c.name for c in report.checks} == _CHECK_NAMES

    def test_syntax_error_fails_compiles(self, tmp_path: Path) -> None:
        root = _write_repo(tmp_path)
        (root / "src" / "other.py").write_text("(unclosed paren\n", encoding="utf-8")
        report = verify_graph_invariants(root, baseline_state_keys=_BASELINE_KEYS)
        assert not report.passed
        assert any(c.name == "compiles" for c in report.failures)

    def test_removed_state_key_fails_schema(self, tmp_path: Path) -> None:
        report = verify_graph_invariants(
            _write_repo(tmp_path, state=_STATE_MISSING_KEY),
            baseline_state_keys=_BASELINE_KEYS,
        )
        assert not report.passed
        assert any(c.name == "state_schema_compatible" for c in report.failures)

    def test_added_state_key_still_passes(self, tmp_path: Path) -> None:
        # A superset (fields added) is a non-breaking change → must pass.
        report = verify_graph_invariants(
            _write_repo(tmp_path), baseline_state_keys={"goal", "messages"}
        )
        assert report.passed, [c.detail for c in report.checks]

    def test_unknown_router_target_fails(self, tmp_path: Path) -> None:
        report = verify_graph_invariants(
            _write_repo(tmp_path, routers=_BAD_ROUTERS_UNKNOWN_NODE),
            baseline_state_keys=_BASELINE_KEYS,
        )
        assert not report.passed
        assert any(c.name == "routers_valid" for c in report.failures)

    def test_self_loop_fails(self, tmp_path: Path) -> None:
        report = verify_graph_invariants(
            _write_repo(tmp_path, task_graph=_BAD_TASK_GRAPH_SELF_LOOP),
            baseline_state_keys=_BASELINE_KEYS,
        )
        assert not report.passed
        assert any(c.name == "no_self_loops" for c in report.failures)

    def test_missing_graph_files_skip_checks(self, tmp_path: Path) -> None:
        # No graph files at all → structural checks skip (pass); a valid .py
        # keeps compiles green. All-skipped ⇒ report passes.
        root = _write_repo(tmp_path, include_graph=False)
        (root / "src" / "m.py").write_text("x = 1\n", encoding="utf-8")
        report = verify_graph_invariants(root, baseline_state_keys=_BASELINE_KEYS)
        assert report.passed
        details = " ".join(c.detail for c in report.checks)
        assert "skipped" in details

    def test_no_baseline_skips_state_check(self, tmp_path: Path) -> None:
        report = verify_graph_invariants(
            _write_repo(tmp_path), baseline_state_keys=None
        )
        assert report.passed
        state_check = next(
            c for c in report.checks if c.name == "state_schema_compatible"
        )
        assert "skipped" in state_check.detail


# ─── imports_clean: dynamic subprocess import smoke ─────────────────────


class TestImportSmoke:
    """The ``imports_clean`` check — a subprocess imports the mutated graph.

    Uses the minimal planted fixtures (fast: the minimal task_graph has no
    imports, so the subprocess resolves instantly). The ``_check_imports``
    skip/fail/pass branches are each exercised, plus a subprocess-isolation
    assertion proving the smoke reads the SHADOW src without poisoning the live
    process's ``sys.modules``.
    """

    def test_imports_clean_passes_on_minimal_graph(self, tmp_path: Path) -> None:
        report = verify_graph_invariants(_write_repo(tmp_path))
        smoke = next(c for c in report.checks if c.name == "imports_clean")
        assert smoke.passed, smoke.detail

    def test_imports_clean_fails_on_broken_import(self, tmp_path: Path) -> None:
        report = verify_graph_invariants(
            _write_repo(tmp_path, task_graph=_BROKEN_IMPORT_TASK_GRAPH)
        )
        # The broken import trips imports_clean → the whole report fails, so a
        # graph-breaking mutation would roll back in the engine.
        assert not report.passed
        smoke = next(c for c in report.failures if c.name == "imports_clean")
        assert "import failed" in smoke.detail
        assert "does_not_exist" in smoke.detail

    def test_imports_clean_skips_when_task_graph_absent(self, tmp_path: Path) -> None:
        # No task_graph.py ⇒ the smoke cannot run ⇒ skip-as-pass (never
        # false-positive on a repo shape it didn't plant).
        root = _write_repo(tmp_path, include_graph=False)
        (root / "src" / "m.py").write_text("x = 1\n", encoding="utf-8")
        report = verify_graph_invariants(root)
        smoke = next(c for c in report.checks if c.name == "imports_clean")
        assert smoke.passed
        assert "skipped" in smoke.detail

    def test_imports_clean_reads_shadow_not_live(self, tmp_path: Path) -> None:
        # The shadow state defines a marker the LIVE state lacks; task_graph
        # imports it. A passing smoke PROVES the subprocess resolved the SHADOW
        # src (the live state has no such attribute → import would fail), and
        # the post-run check PROVES the subprocess did not poison this process.
        root = tmp_path / "repo"
        gdir = root / "src" / "graph"
        gdir.mkdir(parents=True)
        (root / "src" / "__init__.py").write_text("", encoding="utf-8")
        (gdir / "__init__.py").write_text("", encoding="utf-8")
        (gdir / "state.py").write_text('SHADOW_MARKER = "isolated"\n', encoding="utf-8")
        (gdir / "routers.py").write_text("def route():\n    return 1\n", encoding="utf-8")
        (gdir / "task_graph.py").write_text(
            "from src.graph.state import SHADOW_MARKER  # lives ONLY in shadow\n"
            "_ = SHADOW_MARKER\n",
            encoding="utf-8",
        )
        report = verify_graph_invariants(root)
        smoke = next(c for c in report.checks if c.name == "imports_clean")
        assert smoke.passed, smoke.detail
        import src.graph.state as live_state

        assert not hasattr(live_state, "SHADOW_MARKER")  # live unpoisoned


# ─── extract_state_keys ─────────────────────────────────────────────────


class TestExtractStateKeys:
    def test_extracts_live_agent_state_keys(self) -> None:
        live = _REPO_ROOT / "src" / "graph" / "state.py"
        keys = extract_state_keys(live)
        assert keys is not None
        assert "messages" in keys
        assert "submitted_goal" in keys

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert extract_state_keys(tmp_path / "absent.py") is None


# ─── Live-graph sanity (no regression on the baseline) ──────────────────


class TestLiveGraphWellFormed:
    """The current live graph is invariant-clean — guards against a future
    edit that silently breaks the verifier's own baseline."""

    def test_live_repo_is_invariant_clean(self) -> None:
        live_state = _REPO_ROOT / "src" / "graph" / "state.py"
        report = verify_graph_invariants(
            _REPO_ROOT, baseline_state_keys=extract_state_keys(live_state) or set()
        )
        assert report.passed, [c.detail for c in report.checks]
        # The live graph registers 15 nodes; its routers only target those + END.
        routers = next(c for c in report.checks if c.name == "routers_valid")
        assert routers.passed


# ─── Engine hook wiring ─────────────────────────────────────────────────


class TestEngineInvariantHook:
    """SelfEvolutionEngine._verify_invariants: good repo passes, bad repo
    fails, missing shadow repo skips, verifier error fails open."""

    @staticmethod
    def _engine() -> SelfEvolutionEngine:
        # No gateway needed — the invariant check is pure static AST analysis.
        return SelfEvolutionEngine()

    @pytest.mark.asyncio
    async def test_good_repo_passes(self, tmp_path: Path) -> None:
        tracker = SimpleNamespace(repo_dir=_write_repo_with_live_graph(tmp_path))
        result, failed = await self._engine()._verify_invariants(tracker)
        assert failed is False
        assert result["passed"] is True

    @pytest.mark.asyncio
    async def test_syntax_error_repo_fails(self, tmp_path: Path) -> None:
        root = _write_repo_with_live_graph(tmp_path)
        (root / "src" / "other.py").write_text("(unclosed paren\n", encoding="utf-8")
        tracker = SimpleNamespace(repo_dir=root)
        result, failed = await self._engine()._verify_invariants(tracker)
        assert failed is True
        assert result["passed"] is False
        assert "compile" in result["reason"]

    @pytest.mark.asyncio
    async def test_missing_repo_dir_skips(self) -> None:
        # A tracker with no repo_dir attribute ⇒ skipped (fail-open).
        tracker = SimpleNamespace()
        result, failed = await self._engine()._verify_invariants(tracker)
        assert failed is False
        assert result["passed"] is True

    @pytest.mark.asyncio
    async def test_verifier_error_fails_open(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src.evolution.invariants as invariants

        def _boom(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("forced verifier failure")

        monkeypatch.setattr(invariants, "verify_graph_invariants", _boom)
        tracker = SimpleNamespace(repo_dir=_write_repo_with_live_graph(tmp_path))
        result, failed = await self._engine()._verify_invariants(tracker)
        # Fails OPEN: the verifier bug is swallowed, never aborts the cycle.
        assert failed is False
        assert result["passed"] is True
        assert "open" in result["reason"]
