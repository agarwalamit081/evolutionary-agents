"""File writer tool — writes files within a sandboxed directory."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from loguru import logger

from src.config.settings import get_settings


async def file_writer(
    file_path: str,
    content: str,
    sandbox_root: Optional[str] = None,
    create_dirs: bool = False,
    encoding: str = "utf-8",
) -> str:
    """Write content to a file within the sandboxed directory.

    Args:
        file_path: Relative path to the file to write.
        content: Content to write.
        sandbox_root: Root directory for sandboxing. Defaults to
            ``settings.agent.results_root`` (``results``).
        create_dirs: Create parent directories if they don't exist.
        encoding: File encoding.

    Returns:
        Confirmation message or error.
    """
    if sandbox_root is None:
        sandbox_root = get_settings().agent.results_root
    root = Path(sandbox_root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    # Goals embed the workspace name in the save path (e.g. "save to
    # results/<file>"). Strip a leading workspace-name component so the path
    # resolves under sandbox_root instead of double-nesting to
    # results/results/<file>. Covers the literal "results" plus the configured
    # and resolved results_root names, so per-query workspaces de-nest too.
    ws_names = {
        n.lower()
        for n in (
            "results",
            Path(get_settings().agent.results_root).name,
            root.name,
        )
    }
    parts = Path(file_path).parts
    while len(parts) > 1 and parts[0].lower() in ws_names:
        parts = parts[1:]
    target = (root / Path(*parts)).resolve()

    # Security: prevent path traversal
    if not str(target).startswith(str(root)):
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
                "description": "Create parent directories if they don't exist (default: false).",
                "default": False,
            },
        },
        "required": ["file_path", "content"],
    },
}
