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

from pathlib import Path

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
    """Resolve ``path`` under ``base`` after de-nesting, with a traversal guard.

    ``base`` is a named root (``"results"``/``"workspace"``/``"project"``) or an
    explicit absolute ``Path`` (used when a caller supplies its own
    ``sandbox_root``). The resolved base-root name is added to the strip set so
    an explicit root de-nests its own name too. Raises ``ValueError`` on path
    traversal outside ``base``; callers translate that to their existing
    ``ERROR:`` strings to preserve behavior.
    """
    root = _resolve_base(base)
    parts = _strip(Path(path).parts, root.name)
    target = (root / Path(*parts)).resolve() if parts else root
    if not target.is_relative_to(root):
        raise ValueError(f"Path traversal blocked: {path}")
    return target


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
