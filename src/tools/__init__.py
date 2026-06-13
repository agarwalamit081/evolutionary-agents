"""Tools package — tool registry and built-in tools."""

from src.tools.builtin import ALL_TOOL_DEFINITIONS
from src.tools.registry import ToolRegistry
from src.tools.result_cache import ToolResultCache


def create_default_registry() -> ToolRegistry:
    """Create a ToolRegistry with all built-in tools registered.

    Returns:
        ToolRegistry with 7 built-in tools ready for use.
    """
    registry = ToolRegistry()
    for tool_def in ALL_TOOL_DEFINITIONS:
        registry.register(
            name=tool_def["name"],
            handler=tool_def["handler"],
            description=tool_def["description"],
            parameters=tool_def["parameters"],
            cacheable=tool_def.get("cacheable", False),
        )
    return registry


__all__ = [
    "ToolRegistry",
    "create_default_registry",
    "ALL_TOOL_DEFINITIONS",
    "ToolResultCache",
]
