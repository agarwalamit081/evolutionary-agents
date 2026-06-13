"""Tool listing directory entries within a sandboxed root.

The agent is otherwise blind to the filesystem and must guess filenames. This
read-only listing (dirs first, then files, alphabetized) is gated by the same
sandbox-root + path-traversal guard as ``file_reader``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from loguru import logger

from src.config.settings import get_settings


async def list_directory(
    path: str = ".",
    max_entries: int = 100,
    sandbox_root: Optional[str] = None,
) -> str:
    """List entries in a directory within the sandboxed root.

    Args:
        path: Relative path to the directory (default: the sandbox root).
        max_entries: Maximum number of entries to return (default: 100).
        sandbox_root: Root directory for sandboxing (prevents path traversal).
            Defaults to ``settings.agent.workspace_root``.

    Returns:
        One line per entry: ``[DIR|FILE] name (size)``. Directories listed
        first, then files, each alphabetized.
    """
    if sandbox_root is None:
        sandbox_root = get_settings().agent.workspace_root
    root = Path(sandbox_root).resolve()
    target = (root / path).resolve()

    # Security: prevent path traversal outside the sandbox root.
    if not target.is_relative_to(root):
        return f"ERROR: Path traversal blocked: {path}"
    if not target.exists():
        return f"ERROR: Directory not found: {path}"
    if not target.is_dir():
        return f"ERROR: Not a directory: {path}"

    logger.info(f"list_directory: {path}")

    try:
        entries = sorted(
            target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())
        )
    except OSError as exc:
        return f"ERROR: Cannot read directory: {exc}"

    if not entries:
        return f"(empty directory: {path})"

    lines: list[str] = []
    for entry in entries[:max_entries]:
        try:
            kind = "DIR " if entry.is_dir() else "FILE"
            size = "" if entry.is_dir() else f" ({entry.stat().st_size} bytes)"
        except OSError:
            kind, size = "?   ", ""
        lines.append(f"[{kind}] {entry.name}{size}")

    if len(entries) > max_entries:
        lines.append(
            f"... ({len(entries)} entries total, showing first {max_entries})"
        )
    return "\n".join(lines)


TOOL_DEFINITION = {
    "name": "list_directory",
    "handler": list_directory,
    "description": (
        "List files and subdirectories at a path inside the sandboxed project "
        "directory. Directories are shown first, then files, alphabetized, "
        "with file sizes. Use this to discover what exists before reading or "
        "writing files, instead of guessing filenames. Path traversal is blocked."
    ),
    # Listings mutate as files are written during a run — never cache.
    "cacheable": False,
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path to the directory (default: project root).",
                "default": ".",
            },
            "max_entries": {
                "type": "integer",
                "description": "Maximum number of entries to return (default: 100).",
                "default": 100,
            },
        },
        "required": [],
    },
}
