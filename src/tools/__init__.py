"""Tools package — tool registry and built-in tools."""

from src.tools.builtin import ALL_TOOL_DEFINITIONS, TOOL_ANNOTATIONS
from src.tools.metrics import ToolMetricsRecorder
from src.tools.registry import ToolRegistry
from src.tools.result_cache import ToolResultCache


def create_default_registry() -> ToolRegistry:
    """Create a ToolRegistry with all built-in tools registered.

    Each tool's category ``tags``, MCP ``mcp_hints`` (F3), and success contract
    (#11) are sourced from the tool definition when present, otherwise from the
    central ``TOOL_ANNOTATIONS`` map — a single source of truth rather than
    per-file edits. Tools absent from the map register with empty tags/hints
    and no contract.

    Returns:
        ToolRegistry with 7 built-in tools ready for use.
    """
    registry = ToolRegistry()
    for tool_def in ALL_TOOL_DEFINITIONS:
        name = tool_def["name"]
        annotation = TOOL_ANNOTATIONS.get(name, {})
        # Definition-supplied annotations win; otherwise fall back to the
        # central map. isinstance-narrowed so a mis-shaped map entry never
        # crashes register() (it just registers un-annotated).
        raw_tags = tool_def.get("tags") or annotation.get("tags")
        raw_hints = tool_def.get("mcp_hints") or annotation.get("mcp_hints")
        tags = list(raw_tags) if isinstance(raw_tags, list) else None
        mcp_hints = dict(raw_hints) if isinstance(raw_hints, dict) else None
        # Per-tool success contract (#11) — additive; None when neither the
        # definition nor the central map supplies one (today's behavior, where
        # a non-raising handler is recorded as success).
        raw_contract = tool_def.get("success_contract") or annotation.get(
            "success_contract"
        )
        success_contract = dict(raw_contract) if isinstance(raw_contract, dict) else None
        registry.register(
            name=name,
            handler=tool_def["handler"],
            description=tool_def["description"],
            parameters=tool_def["parameters"],
            cacheable=tool_def.get("cacheable", False),
            tags=tags,
            mcp_hints=mcp_hints,
            success_contract=success_contract,
        )
    return registry


__all__ = [
    "ToolRegistry",
    "create_default_registry",
    "ALL_TOOL_DEFINITIONS",
    "TOOL_ANNOTATIONS",
    "ToolResultCache",
    "ToolMetricsRecorder",
]
