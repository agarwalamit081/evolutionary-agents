"""Shared path resolution for the file-touching tools.

Single source of truth so ``file_writer`` / ``code_executor`` /
``terminal_command`` / ``file_reader`` / ``verify`` / ``execute`` all agree on
where a logical path like ``results/foo.md`` lands on disk.

Historical bug this centralizes: each tool resolved its own working directory
(``results_root`` vs ``workspace_root``) and de-nested redundant ``results/``
prefixes independently. The same string then resolved to *different* files
across tools — so a ``code_executor`` script globbing ``results/*.md`` from
inside ``results/`` found ``results/results/*.md`` (nothing) and the LLM
fabricated output. Aligning every tool to ``project_root`` (parent of
``results_root``) and routing all de-nesting through here makes a path resolve
identically whether written, executed, or cat'd.
"""

from __future__ import annotations

import contextvars
import re
import shutil
from pathlib import Path

from loguru import logger

# Reference the *module object* (not the symbol) so tests can monkeypatch
# ``src.config.settings.get_settings`` (the single source) and have it take
# effect here — a module-level ``from … import get_settings`` would capture the
# original and ignore the patch. This keeps settings reading in exactly one place.
from src.config import settings as _settings

# Named roots a caller can ask ``normalize`` to resolve against.
_ROOT_BY_NAME = {
    "results": lambda: _results_root(),
    "workspace": lambda: _workspace_root(),
    "project": lambda: _project_root(),
}

# Phase 7: active run identifier. Bound by the runner (``src.runner.execute_run``)
# so the shared resolver routes a run's deliverables under ``results_root / <run_id> / ...``,
# isolating each run on disk. Backed by a ``ContextVar`` (not a process global)
# so it is scoped per async task: two concurrent runs in the same event loop each
# see their own run_id. The global it replaced bled the last writer's id into
# every concurrent run — a horizontal-scaling blocker once the API enqueues runs
# to a worker (Phase 2b). Async tasks copy their context at creation, so a run_id
# set inside ``execute_run`` propagates to every node/tool coroutine it awaits but
# NOT to sibling runs. ``None`` when no run_id is in play → every path resolves
# flat exactly as before (non-regression for non-run-id runs).
_active_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "turing_active_run_id", default=None
)


def set_active_run_id(run_id: str | None) -> None:
    """Bind the active run_id for per-run results subfoldering (main.py entry).

    Scopes to the current async context (task). Pass ``None`` to reset (e.g.
    between independent runs / in tests).
    """
    _active_run_id.set(run_id)


def get_active_run_id() -> str | None:
    """Current active run_id for this context, or ``None`` when none is bound."""
    return _active_run_id.get()


def _subdir_active() -> bool:
    """Per-run subfoldering is on only with a run_id AND the setting enabled."""
    if _active_run_id.get() is None:
        return False
    try:
        return bool(_agent().results_per_run_subdir)  # type: ignore[attr-defined]
    except AttributeError:
        # Minimal agent fakes / older settings without the field → treat as off.
        return False


def _maybe_inject_run_subdir(parts: tuple[str, ...], root: Path) -> tuple[str, ...]:
    """Prefix a run_id component so writes land under ``results_root/<run_id>/``.

    Skipped unless ``root`` is the results root (workspace/project/explicit-bases
    are never subfoldered), when subfoldering is off, or when the (already
    de-nested) path already starts with the run_id — so a goal that names
    ``results/<run_id>/x`` is NOT double-nested to ``results/<run_id>/<run_id>/x``.
    """
    run_id = _active_run_id.get()
    if run_id is None or not _subdir_active():
        return parts
    if root != _results_root():
        return parts  # only the results base is per-run subfoldered
    if parts and parts[0] == run_id:
        return parts
    return (run_id, *parts)


def _agent() -> "object":
    # ``AgentSettings`` — typed loosely to avoid an import cycle with config.
    return _settings.get_settings().agent


def _results_root() -> Path:
    return Path(_agent().results_root).resolve()  # type: ignore[attr-defined]


def _workspace_root() -> Path:
    return Path(_agent().workspace_root).resolve()  # type: ignore[attr-defined]


def _project_root() -> Path:
    """Parent of ``results_root`` — the common ancestor every tool runs from.

    On host-run this is the repo root (``results_root="results"`` relative to
    CWD). In the container (``RESULTS_ROOT=/home/turing/.turing/results``) it is
    the persisted volume. Aligning cwd here is what makes ``results/foo.md``
    resolve the same for reads, writes, and subprocess globs.
    """
    return _results_root().parent


def results_root() -> Path:
    """Resolved ``results_root`` (where ``file_writer`` deliverables live)."""
    return _results_root()


def workspace_root() -> Path:
    """Resolved ``workspace_root`` (inputs/fixtures scratch dir)."""
    return _workspace_root()


def project_root() -> Path:
    """Resolved project root — the single cwd for all file-touching tools."""
    return _project_root()


# A run_id is a single safe path component (the CLI/worker pass ``q09`` /
# ``battery04_q09-20260624``). No separators, no ``.``/``..`` — so it can never
# name a traversal target when joined under results_root.
_RUN_ID_RE = re.compile(r"[A-Za-z0-9_.\-]+")


def run_subdir_path(run_id: str) -> Path:
    """Resolve a run's results subfolder ``<results_root>/<run_id>`` safely.

    Validates ``run_id`` is a single safe path component and that the resolved
    folder stays inside ``results_root``, so a caller cannot name a traversal
    target. Mirrors the safety the CLI ``--clean`` applies
    (``main._clean_run_results``) but returns the Path so the worker path can
    reuse it without CLI coupling.

    Raises:
        ValueError: ``run_id`` is empty, contains separators/``..``, or the
            resolved folder escapes ``results_root``.
    """
    if (
        not isinstance(run_id, str)
        or not run_id
        or run_id in {".", ".."}
        or _RUN_ID_RE.fullmatch(run_id) is None
    ):
        raise ValueError(f"unsafe run_id: {run_id!r}")
    base = _results_root()
    sub = (base / run_id).resolve()
    if sub == base or not sub.is_relative_to(base):
        raise ValueError(f"run subdir escapes results_root: {sub}")
    return sub


def clean_run_subdir(run_id: str) -> bool:
    """Delete a run's results subfolder if present (best-effort, never raises).

    Removes ONLY ``<results_root>/<run_id>`` — never the whole ``results_root``
    and never anything that resolves outside it. Returns ``True`` if a subfolder
    was removed, ``False`` if none existed or the run_id was refused. Any failure
    is logged at DEBUG and swallowed so a run never aborts on cleanup —
    contamination prevention is best-effort (the same resilience posture as the
    cost-ledger / curve-gate writes).
    """
    try:
        sub = run_subdir_path(run_id)
    except ValueError as exc:
        logger.debug(f"clean_run_subdir refused unsafe run_id: {exc}")
        return False
    if not sub.exists():
        return False
    try:
        shutil.rmtree(sub)
    except OSError as exc:
        logger.debug(f"clean_run_subdir could not remove {sub}: {exc}")
        return False
    logger.info(f"Cleared prior results subfolder: {sub}")
    return True


def _strip_names(*extra: str) -> set[str]:
    agent = _agent()
    return {
        n.lower()
        for n in (
            "results",
            Path(agent.results_root).name,  # type: ignore[attr-defined]
            Path(agent.workspace_root).name,  # type: ignore[attr-defined]
            *extra,
        )
        if n
    }


def strip_results_prefix(parts: tuple[str, ...]) -> tuple[str, ...]:
    """Drop redundant leading ``results/``/workspace components from path parts.

    Goals say "save to results/<file>" but ``file_writer`` already writes under
    ``results_root``, which would double-nest to ``results/results/<file>``. Strip
    a leading component matching the literal ``results``, the configured
    ``results_root`` name, or the workspace name so ``results/x``,
    ``results/results/x`` and bare ``x`` collapse to one target. Never strips a
    lone filename (``len(parts) > 1`` guard).
    """
    return _strip(parts)


def normalize(path: str, *, base: str | Path = "results") -> Path:
    """Resolve ``path`` under ``base`` as a WRITE target, with per-run isolation.

    ``base`` is a named root (``"results"``/``"workspace"``/``"project"``) or an
    explicit absolute ``Path`` (used when a caller supplies its own
    ``sandbox_root``). The resolved base-root name is added to the strip set so
    an explicit root de-nests its own name too.

    Phase 7: when per-run subfoldering is active (an ``_active_run_id`` is bound
    AND ``results_per_run_subdir`` is on), a results write is routed under
    ``results_root / <run_id> / <path>`` so each run's deliverables are isolated
    on disk. A goal that already names ``results/<run_id>/x`` is de-duplicated
    (never double-nested). Workspace/project bases are never subfoldered. Raises
    ``ValueError`` on path traversal outside ``base``; callers translate that to
    their existing ``ERROR:`` strings to preserve behavior.

    Use ``resolve_existing`` for reads — it prefers the run subdir but falls back
    to the flat root so legacy/cross-run deliverables still recall.
    """
    root = _resolve_base(base)
    parts = _strip(Path(path).parts, root.name)
    parts = _maybe_inject_run_subdir(parts, root)
    target = (root / Path(*parts)).resolve() if parts else root
    if not target.is_relative_to(root):
        raise ValueError(f"Path traversal blocked: {path}")
    return target


def resolve_existing(path: str, *, base: str | Path = "results") -> Path:
    """Resolve ``path`` as a READ target: run subdir first, flat fallback.

    The primary candidate is ``normalize(path, base)`` — the run subdir when
    per-run subfoldering is active, the flat root otherwise. If it exists on
    disk it is returned. Otherwise we re-resolve *without* the run_id injection
    (the flat root) and return that when it exists — so recall of older flat
    deliverables (battery-03) and cross-run reads still work. When neither
    exists, the primary candidate is returned (the canonical "where it would be"
    location), so a caller's ``.exists()`` check simply reports absent.

    Raises ``ValueError`` only when the flat resolution escapes ``base`` — the
    primary (subdir) candidate is already traversal-guarded by ``normalize``.
    """
    primary = normalize(path, base=base)
    if primary.exists():
        return primary
    flat_root = _resolve_base(base)
    flat_parts = _strip(Path(path).parts, flat_root.name)
    flat = (flat_root / Path(*flat_parts)).resolve() if flat_parts else flat_root
    if flat != primary and flat.is_relative_to(flat_root) and flat.exists():
        return flat
    return primary


def _strip(parts: tuple[str, ...], *extra: str) -> tuple[str, ...]:
    """Shared de-nest loop. ``extra`` adds names (e.g. an explicit base root)."""
    names = _strip_names(*extra)
    parts = tuple(parts)
    while len(parts) > 1 and parts[0].lower() in names:
        parts = parts[1:]
    return parts


def _resolve_base(base: str | Path) -> Path:
    if isinstance(base, Path):
        return base.resolve()
    try:
        resolver = _ROOT_BY_NAME[base]
    except KeyError as exc:  # pragma: no cover — defensive; callers pass literals
        raise ValueError(f"Unknown root base: {base!r}") from exc
    return resolver()
