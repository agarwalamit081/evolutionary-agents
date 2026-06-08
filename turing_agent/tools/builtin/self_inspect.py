"""Self-inspect tool — reads agent source code for self-awareness."""

from __future__ import annotations

from pathlib import Path

from loguru import logger


async def self_inspect(module_path: str = "", max_lines: int = 100) -> str:
    """Read the agent's own source code for self-inspection.

    Used by the evolution engine and reflective processes to analyze
    and potentially modify the agent's own code.

    Args:
        module_path: Relative path within the turing_agent package (e.g., "graph/nodes/execute.py").
        max_lines: Maximum lines to return.

    Returns:
        Source code contents.
    """
    # Determine the turing_agent package root
    agent_root = Path(__file__).parent.parent.parent

    if not module_path:
        # List available modules
        modules: list[str] = []
        for py_file in agent_root.rglob("*.py"):
            rel = py_file.relative_to(agent_root)
            modules.append(str(rel))
        modules.sort()
        return f"Turing Agent modules ({len(modules)} files):\n" + "\n".join(f"  {m}" for m in modules[:50])

    target = (agent_root / module_path).resolve()

    # Security: ensure within agent root
    if not str(target).startswith(str(agent_root.resolve())):
        return f"ERROR: Path traversal blocked: {module_path}"

    if not target.exists():
        return f"ERROR: Module not found: {module_path}"

    if not target.is_file() or not target.suffix == ".py":
        return f"ERROR: Not a Python file: {module_path}"

    logger.info(f"Self-inspecting: {module_path}")

    content = target.read_text(encoding="utf-8")
    lines = content.splitlines()

    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines.append(f"\n... (truncated at {max_lines} lines)")

    return "\n".join(lines)


TOOL_DEFINITION = {
    "name": "self_inspect",
    "handler": self_inspect,
    "description": (
        "Read the agent's own source code for self-analysis. "
        "Call without arguments to list all modules. "
        "Provide a module path (e.g., 'graph/nodes/execute.py') to read specific files."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "module_path": {
                "type": "string",
                "description": "Relative path within turing_agent package (e.g., 'graph/nodes/execute.py').",
                "default": "",
            },
            "max_lines": {
                "type": "integer",
                "description": "Maximum lines to return (default: 100).",
                "default": 100,
            },
        },
        "required": [],
    },
}
