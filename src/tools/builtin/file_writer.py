"""File writer tool — writes files within a sandboxed directory."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from loguru import logger

from src.tools._paths import normalize, results_root


async def file_writer(
    file_path: str,
    content: str,
    sandbox_root: Optional[str] = None,
    create_dirs: bool = True,
    encoding: str = "utf-8",
) -> str:
    """Write content to a file within the sandboxed directory.

    Args:
        file_path: Relative path to the file to write.
        content: Content to write.
        sandbox_root: Root directory for sandboxing. Defaults to
            ``settings.agent.results_root`` (``results``).
        create_dirs: Create parent directories if they don't exist. Defaults
            to ``True`` so deliverables land under nested run subfolders
            (e.g. ``results/q03/retention.csv``) without a prior mkdir — a
            ``False`` default silently failed writes to missing parents.
        encoding: File encoding.

    Returns:
        Confirmation message or error.
    """
    # Resolve the sandbox root, then de-nest + traverse-guard via the shared
    # resolver so file_writer, code_executor, and terminal_command all agree on
    # where "results/<file>" lands. A None sandbox_root means the configured
    # results_root (where deliverables live).
    root = results_root() if sandbox_root is None else Path(sandbox_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    try:
        target = normalize(file_path, base=root)
    except ValueError:
        # normalize raises on path traversal outside the root.
        return f"ERROR: Path traversal blocked: {file_path}"

    # Size check (max 1MB)
    if len(content) > 1_000_000:
        return f"ERROR: Content too large ({len(content)} bytes, max 1MB)"

    # Create parent directories if requested
    if create_dirs:
        target.parent.mkdir(parents=True, exist_ok=True)

    # Check parent directory exists
    if not target.parent.exists():
        return f"ERROR: Parent directory does not exist: {target.parent}"

    logger.info(f"Writing file: {file_path} ({len(content)} bytes)")

    try:
        target.write_text(content, encoding=encoding)
        return f"Successfully wrote {len(content)} bytes to {file_path}"
    except Exception as exc:
        return f"ERROR: Failed to write file: {exc}"


TOOL_DEFINITION = {
    "name": "file_writer",
    "handler": file_writer,
    "description": (
        "Write content to a file in the sandboxed project directory. "
        "Can create parent directories. Path traversal attacks are blocked. "
        "Maximum file size is 1MB."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Relative path to the file to write.",
            },
            "content": {
                "type": "string",
                "description": "Content to write to the file.",
            },
            "create_dirs": {
                "type": "boolean",
                "description": "Create parent directories if they don't exist (default: true).",
                "default": True,
            },
        },
        "required": ["file_path", "content"],
    },
}
