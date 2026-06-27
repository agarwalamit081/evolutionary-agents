"""Graph-invariant verifier, stage-1 (Phase 5 G1).

Deterministic checks over a deployed mutation's shadow-repo snapshot. Four are
**static-AST** (no LLM, no execution); one (``imports_clean``) runs a hermetic
import smoke in a subprocess. Together they are fast, hermetic, and
unit-testable with planted fixtures. The evolution engine runs these between
``deploy()`` and ``post_deploy_verify()`` for CODE mutations; a failed
invariant trips the existing ``rollback_deployment()`` path so a
graph-breaking mutation never reaches the running agent.

Five checks (mapped to findings.md's cycle / bad-router / breaking-state-change
classes):

1. ``compiles`` — every ``.py`` under the repo ``src/`` tree parses and
   compiles (``ast.parse`` + ``compile``; no file written, no import). Catches
   a mutation that left the agent's own source in a syntax-broken state.
2. ``imports_clean`` — the mutated ``src/graph/{state,routers,task_graph}``
   modules import successfully in a **subprocess** that shares the live
   interpreter + environment but isolates ``sys.modules``. Catches import-time
   breakage ``compile`` cannot: ``NameError`` / ``AttributeError`` /
   ``ModuleNotFoundError`` / circular-import / missing-symbol raised while the
   mutated graph's import graph is resolved. Never run in-process — importing
   the shadow ``src`` would clobber the running agent's own modules.
3. ``state_schema_compatible`` — the mutated ``AgentState`` TypedDict is a
   superset of the live baseline (no baseline key removed). Adding fields is
   safe; removing/renaming one a live node reads is a breaking change.
4. ``routers_valid`` — every ``return "<literal>"`` in a ``route_*`` function
   is a registered node name (``add_node("...", ...)``) or the sentinel
   ``"complete"`` (which maps to END). Catches a router rewired to a node the
   graph never registered (LangGraph would raise at compile time — we catch it
   earlier, without compiling).
5. ``no_self_loops`` — no ``graph.add_edge("X", "X")`` with identical string
   literals (a degenerate progress-free cycle; the intentional
   ``execute→execute`` retry goes through ``add_conditional_edges`` and is
   bounded by the iteration cap, so it is exempt).

Checks that cannot run (the target file is absent from the shadow repo, or no
baseline was supplied) are reported ``passed`` with a ``skipped`` note rather
than failed — the verifier must never false-positive on a repo shape it didn't
plant. The engine wraps the whole call in a fail-open guard, so a verifier bug
never aborts an evolution cycle (the post-deploy sandbox smoke remains as the
second barrier).

Note: termination and budget are **runtime-enforced** (``effective_max_iterations``
+ the budget hard-cap live inside ``route_after_*`` router logic, not graph
structure), so a sound *static* proof that a mutated graph still terminates is
deferred — the runtime sandbox smoke (``post_deploy_verify``) is the dynamic
backstop.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

# Router return literals that are NOT node names but are valid routing targets:
# ``"complete"`` maps to END in every ``add_conditional_edges`` mapping dict in
# ``task_graph.py``. ``route_after_store`` / ``route_after_hitl`` /
# ``route_after_error`` all legitimately return it.
_ROUTER_SENTINELS: frozenset[str] = frozenset({"complete"})


@dataclass(frozen=True)
class InvariantCheck:
    """The outcome of one invariant check.

    Attributes:
        name: Stable check identifier (``compiles``, ``imports_clean``,
            ``state_schema_compatible``, ``routers_valid``, ``no_self_loops``).
        passed: ``True`` iff the invariant holds (or was deliberately skipped).
        detail: Human-readable explanation / first offending item.
    """

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class InvariantReport:
    """Aggregate result of all stage-1 invariant checks.

    Attributes:
        checks: Ordered tuple of every ``InvariantCheck``.
    """

    checks: tuple[InvariantCheck, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        """``True`` iff every check passed (skipped checks count as passed)."""
        return all(check.passed for check in self.checks)

    @property
    def failures(self) -> list[InvariantCheck]:
        """Only the failing checks (empty ⇒ all green)."""
        return [check for check in self.checks if not check.passed]


# ─── AST extractors ─────────────────────────────────────────────────────


def _safe_parse(path: Path) -> ast.AST | None:
    """Parse ``path`` to an AST, returning ``None`` on any read/syntax error.

    ``None`` is the universal "could not inspect this file" signal: callers
    treat it as "skip the check" (pass) rather than "fail", so a shadow repo
    that doesn't contain a given graph file never false-positives.
    """
    if not path.is_file():
        return None
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        return ast.parse(source, filename=str(path))
    except SyntaxError:
        # A syntax error HERE is itself a finding, but it is reported by the
        # dedicated ``compiles`` check (which names the file). The structural
        # extractors should not double-report it.
        return None


def _iter_python_src(root: Path) -> Iterator[Path]:
    """Yield every ``.py`` under ``root/src`` (or ``root`` if no ``src/``).

    Scoped to ``src/`` so a large shadow repo (which may mirror the whole
    project) is not byte-compiled in its entirety — only the agent's own
    source matters for the ``compiles`` invariant.
    """
    src = root / "src"
    base = src if src.is_dir() else root
    yield from sorted(base.rglob("*.py"))


def extract_state_keys(path: Path, cls_name: str = "AgentState") -> set[str] | None:
    """Return the set of annotated field names of a TypedDict class.

    Public so the engine can compute the LIVE (unmutated) ``AgentState``
    baseline from the running agent's ``state.py`` before running the
    invariants — the check compares the mutated repo's key set against this.

    Returns ``None`` if the file is absent/unparseable or the class isn't
    defined in it (→ check skipped). TypedDict fields are ``ast.AnnAssign``
    nodes (``name: Type``) whose target is a plain ``Name``.
    """
    tree = _safe_parse(path)
    if tree is None:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            keys: set[str] = set()
            for stmt in node.body:
                if (
                    isinstance(stmt, ast.AnnAssign)
                    and isinstance(stmt.target, ast.Name)
                ):
                    keys.add(stmt.target.id)
            return keys
    return None


def _extract_node_names(path: Path) -> set[str] | None:
    """Return node names registered via ``add_node("<name>", ...)``.

    Returns ``None`` if ``task_graph.py`` is absent/unparseable (→ check
    skipped). The live graph registers 15 nodes this way.
    """
    tree = _safe_parse(path)
    if tree is None:
        return None
    names: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_node"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            names.add(node.args[0].value)
    return names or None


def _extract_router_literals(path: Path) -> dict[str, set[str]] | None:
    """Return ``{route_fn_name: {returned_string_literals}}``.

    Collects every ``return "<str>"`` inside a ``def route_*`` function.
    Returns ``None`` if ``routers.py`` is absent/unparseable (→ check skipped).

    Over-collects literals from any nested helper inside a router (there are
    none today); that only makes the check *stricter* (more literals to
    validate against the node set), which is the safe direction for a guard.
    """
    tree = _safe_parse(path)
    if tree is None:
        return None
    result: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("route_"):
            literals: set[str] = set()
            for sub in node.body:  # walk only THIS function's body (no nested-fn bleed)
                for inner in ast.walk(sub):
                    if (
                        isinstance(inner, ast.Return)
                        and inner.value is not None
                        and isinstance(inner.value, ast.Constant)
                        and isinstance(inner.value.value, str)
                    ):
                        literals.add(inner.value.value)
            result[node.name] = literals
    return result or None


def _detect_self_loops(path: Path) -> list[tuple[str, int]]:
    """Return ``[(node_name, line_no)]`` for each ``add_edge("X", "X")``.

    A literal self-loop (both args the same string ``Constant``) is a
    progress-free cycle. ``add_edge(START, "classify")`` is exempt (START is a
    Name, not a Constant); the intentional ``execute→execute`` retry is wired
    via ``add_conditional_edges`` (bounded by the iteration cap), also exempt.
    """
    tree = _safe_parse(path)
    if tree is None:
        return []
    loops: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_edge"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[0].value, str)
            and isinstance(node.args[1].value, str)
            and node.args[0].value == node.args[1].value
        ):
            loops.append((node.args[0].value, node.lineno))
    return loops


# ─── Individual checks ──────────────────────────────────────────────────


def _check_compiles(repo_path: Path) -> InvariantCheck:
    """Compile-check every ``.py`` under the repo ``src/`` tree."""
    targets = list(_iter_python_src(repo_path))
    errors: list[str] = []
    for py in targets:
        try:
            source = py.read_text(encoding="utf-8")
        except OSError as exc:
            errors.append(f"{py.relative_to(repo_path)}: {exc}")
            continue
        try:
            # ast.parse surfaces SyntaxError; compile() turns the AST into a
            # code object WITHOUT writing a .pyc and WITHOUT importing — a true
            # "does it compile" probe with no side effects on the shadow repo.
            compile(ast.parse(source, filename=str(py)), str(py), "exec")
        except SyntaxError as exc:
            loc = f"{py.relative_to(repo_path)}:{exc.lineno}"
            errors.append(f"{loc}: {exc.msg}")
    if errors:
        return InvariantCheck(
            "compiles",
            passed=False,
            detail=f"{len(errors)} file(s) fail to compile — {errors[0]}",
        )
    return InvariantCheck(
        "compiles", passed=True, detail=f"{len(targets)} file(s) compiled"
    )


# ─── Dynamic import smoke ────────────────────────────────────────────────

# Importing these three in a subprocess transitively pulls the whole node
# package (src/graph/nodes/* → llm/gateway, memory/manager, tools/registry),
# so a broken import anywhere in the orchestration layer's import graph is
# surfaced — far broader than the ``compiles`` AST check.
_IMPORT_SMOKE_MODULES: tuple[str, ...] = (
    "src.graph.state",
    "src.graph.routers",
    "src.graph.task_graph",
)

# Bounded wall-clock: a pathological mutated import (an accidental blocking
# call / infinite loop at module top) must not hang the evolution cycle.
_IMPORT_SMOKE_TIMEOUT_S: int = 30


def _check_imports(repo_path: Path) -> InvariantCheck:
    """Dynamic import smoke — import the mutated graph modules in a subprocess.

    ``compile`` (the ``compiles`` check) catches ``SyntaxError`` but nothing
    that fires when the module is *resolved*: ``NameError`` / ``AttributeError``
    at module scope, ``ModuleNotFoundError`` for a renamed/missing dependency,
    circular-import errors, or a symbol a downstream ``from … import …``
    expects but no longer exists. Those only surface on actual import.

    Runs ``sys.executable -c "<smoke>"`` in a **child process** with
    ``PYTHONPATH=<repo>`` and the live environment inherited. The shadow
    ``src/`` resolves first on that path, so the *mutated* modules import —
    but in isolation: importing them in-process would replace the live agent's
    ``src.graph.*`` entries in our own ``sys.modules``, poisoning this process.
    Sharing the interpreter + env + deps means an import failure here is a
    genuine breakage the running agent would hit, so there is no
    false-positive surface.

    Skip-pass semantics (mirroring the sibling checks — never false-positive on
    a repo shape it didn't plant): no ``task_graph.py`` ⇒ skip; no interpreter
    found (``FileNotFoundError``) ⇒ skip (fail-open). A timeout or nonzero exit
    ⇒ fail, the detail naming the offending error (last stderr/stdout line).
    """
    task_graph = repo_path / "src" / "graph" / "task_graph.py"
    if not task_graph.is_file():
        return InvariantCheck(
            "imports_clean",
            passed=True,
            detail="skipped (task_graph.py absent from repo)",
        )

    imports = ", ".join(f"importlib.import_module({m!r})" for m in _IMPORT_SMOKE_MODULES)
    smoke_code = f"import importlib; {imports}\n"
    # Inherit the live environment (interpreter, deps, config) so the subprocess
    # matches the running agent's import conditions exactly; prepend the repo so
    # the shadow src/ takes precedence over any installed copy.
    env = {**os.environ, "PYTHONPATH": str(repo_path), "PYTHONUNBUFFERED": "1"}
    try:
        proc = subprocess.run(  # noqa: S603 -- argv is fully literal; code is a hardcoded import probe
            [sys.executable, "-c", smoke_code],
            cwd=str(repo_path),
            env=env,
            capture_output=True,
            text=True,
            timeout=_IMPORT_SMOKE_TIMEOUT_S,
            check=False,
        )
    except FileNotFoundError as exc:
        return InvariantCheck(
            "imports_clean",
            passed=True,
            detail=f"skipped (interpreter unavailable: {exc})",
        )
    except subprocess.TimeoutExpired:
        return InvariantCheck(
            "imports_clean",
            passed=False,
            detail=f"import smoke timed out (>{_IMPORT_SMOKE_TIMEOUT_S}s) — pathological import",
        )

    if proc.returncode != 0:
        stream = (proc.stderr or proc.stdout or "").strip().splitlines()
        last = stream[-1] if stream else f"exit code {proc.returncode}"
        return InvariantCheck(
            "imports_clean",
            passed=False,
            detail=f"import failed: {last}",
        )
    return InvariantCheck(
        "imports_clean",
        passed=True,
        detail="src/graph/{state,routers,task_graph} import clean in subprocess",
    )


def _check_state_schema(
    state_path: Path,
    baseline_keys: set[str] | None,
) -> InvariantCheck:
    """AgentState is a superset of the live baseline (no key removed)."""
    mutated_keys = extract_state_keys(state_path)
    if baseline_keys is None or mutated_keys is None:
        return InvariantCheck(
            "state_schema_compatible",
            passed=True,
            detail="skipped (no baseline or state.py absent from repo)",
        )
    removed = baseline_keys - mutated_keys
    if removed:
        return InvariantCheck(
            "state_schema_compatible",
            passed=False,
            detail=f"removed baseline AgentState key(s): {sorted(removed)}",
        )
    added = mutated_keys - baseline_keys
    return InvariantCheck(
        "state_schema_compatible",
        passed=True,
        detail=f"superset of baseline (+{len(added)} added, 0 removed)",
    )


def _check_routers(
    routers_path: Path, task_graph_path: Path
) -> InvariantCheck:
    """Every router return-literal is a registered node or a known sentinel."""
    router_map = _extract_router_literals(routers_path)
    known_nodes = _extract_node_names(task_graph_path)
    if router_map is None or known_nodes is None:
        return InvariantCheck(
            "routers_valid",
            passed=True,
            detail="skipped (routers.py or task_graph.py absent from repo)",
        )
    valid_targets = known_nodes | _ROUTER_SENTINELS
    bad: list[str] = []
    for fn_name, literals in router_map.items():
        for literal in sorted(literals):
            if literal not in valid_targets:
                bad.append(f"{fn_name} -> {literal!r}")
    if bad:
        return InvariantCheck(
            "routers_valid",
            passed=False,
            detail=f"router(s) target unknown node(s): {bad}",
        )
    return InvariantCheck(
        "routers_valid",
        passed=True,
        detail=(
            f"{len(router_map)} router(s) ⊆ {len(known_nodes)} registered node(s)"
            f" + {sorted(_ROUTER_SENTINELS)}"
        ),
    )


def _check_no_self_loops(task_graph_path: Path) -> InvariantCheck:
    """No ``add_edge("X", "X")`` literal self-loops."""
    loops = _detect_self_loops(task_graph_path)
    if loops:
        rendered = ", ".join(f"{name}@L{line}" for name, line in loops)
        return InvariantCheck(
            "no_self_loops",
            passed=False,
            detail=f"literal self-loop(s): {rendered}",
        )
    return InvariantCheck(
        "no_self_loops", passed=True, detail="no add_edge(X, X) self-loops"
    )


# ─── Public API ─────────────────────────────────────────────────────────


def verify_graph_invariants(
    repo_path: Path,
    *,
    baseline_state_keys: set[str] | None = None,
) -> InvariantReport:
    """Run all stage-1 graph-invariant checks over a deployed shadow-repo snapshot.

    Args:
        repo_path: The shadow-repo root (``GitTracker.repo_dir``) with the CODE
            mutation already applied.
        baseline_state_keys: The live (unmutated) ``AgentState`` field set, used
            as the no-breaking-change baseline. ``None`` ⇒ the
            ``state_schema_compatible`` check is skipped (the engine computes
            this from the running agent's ``state.py``; tests pass it explicitly).

    Returns:
        An :class:`InvariantReport`; ``report.passed`` is ``True`` iff every
        check passed (or was safely skipped). Inspect ``report.failures`` for
        the offending checks.
    """
    root = Path(repo_path)
    state_path = root / "src" / "graph" / "state.py"
    routers_path = root / "src" / "graph" / "routers.py"
    task_graph_path = root / "src" / "graph" / "task_graph.py"

    checks: tuple[InvariantCheck, ...] = (
        _check_compiles(root),
        _check_imports(root),
        _check_state_schema(state_path, baseline_state_keys),
        _check_routers(routers_path, task_graph_path),
        _check_no_self_loops(task_graph_path),
    )
    return InvariantReport(checks=checks)
