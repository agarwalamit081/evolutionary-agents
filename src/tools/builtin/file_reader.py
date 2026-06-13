"""File reader tool — reads files within a sandboxed directory."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from loguru import logger

from src.config.settings import get_settings


def _results_root_fallback(file_path: str) -> Optional[Path]:
    """Resolve a relative path under ``results_root`` (the file_writer output dir).

    The read tools default their sandbox to ``workspace_root`` (inputs/fixtures),
    but ``file_writer`` writes deliverables to ``results_root``. Without this
    fallback the agent cannot read back its own outputs — it writes a file, then
    ``file_reader`` reports "not found", which verify misreads as a missing
    deliverable (F13). Returns the resolved Path if it exists there and stays
    within ``results_root``; absolute paths return ``None`` (kept blocked).
    """
    if file_path is None or Path(file_path).is_absolute():
        return None
    try:
        results_root = Path(get_settings().agent.results_root).resolve()
    except Exception:
        return None
    candidate = (results_root / file_path).resolve()
    if not str(candidate).startswith(str(results_root)):
        return None
    return candidate if candidate.exists() and candidate.is_file() else None


async def file_reader(
    file_path: str,
    sandbox_root: Optional[str] = None,
    max_lines: int = 200,
    encoding: str = "utf-8",
) -> str:
    """Read a file within the sandboxed directory.

    Args:
        file_path: Relative path to the file to read.
        sandbox_root: Root directory for sandboxing (prevents path traversal).
            Defaults to ``settings.agent.workspace_root``
            (``.turing/workspace``).
        max_lines: Maximum number of lines to return.
        encoding: File encoding.

    Returns:
        File contents as a string.
    """
    default_root_used = sandbox_root is None
    if sandbox_root is None:
        sandbox_root = get_settings().agent.workspace_root
    root = Path(sandbox_root).resolve()
    target = (root / file_path).resolve()

    # Security: prevent path traversal
    if not str(target).startswith(str(root)):
        return f"ERROR: Path traversal blocked: {file_path}"

    if not target.exists():
        # F13: when using the default workspace sandbox, also look under
        # results_root so the agent can read back files it wrote via
        # file_writer (which targets results_root). Explicit sandbox_roots
        # (e.g. in tests) are left untouched.
        if default_root_used:
            fallback = _results_root_fallback(file_path)
            if fallback is not None:
                target = fallback
            else:
                return f"ERROR: File not found: {file_path}"
        else:
            return f"ERROR: File not found: {file_path}"

    if not target.is_file():
        return f"ERROR: Not a file: {file_path}"

    # Size check (max 1MB)
    size = target.stat().st_size
    if size > 1_000_000:
        return f"ERROR: File too large ({size} bytes, max 1MB): {file_path}"

    logger.info(f"Reading file: {file_path} ({size} bytes)")

    try:
        content = target.read_text(encoding=encoding)
    except UnicodeDecodeError:
        return f"ERROR: Cannot decode file as {encoding}: {file_path}"

    lines = content.splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines.append(f"\n... (truncated at {max_lines} lines, total {len(content.splitlines())} lines)")

    return "\n".join(lines)


TOOL_DEFINITION = {
    "name": "file_reader",
    "handler": file_reader,
    "description": (
        "Read a file from the sandboxed project directory. "
        "Supports text files with configurable line limits. "
        "Path traversal attacks are blocked."
    ),
    # Read-only file access — content is deterministic per (path, max_lines),
    # so caching repeated reads within/across runs is safe.
    "cacheable": True,
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Relative path to the file to read.",
            },
            "max_lines": {
                "type": "integer",
                "description": "Maximum number of lines to return (default: 200).",
                "default": 200,
            },
        },
        "required": ["file_path"],
    },
}
