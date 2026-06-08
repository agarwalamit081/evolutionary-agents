"""Tools package — tool registry and built-in tools."""

from turing_agent.tools.builtin import ALL_TOOL_DEFINITIONS
from turing_agent.tools.registry import ToolRegistry


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
        )
    return registry


__all__ = [
    "ToolRegistry",
    "create_default_registry",
    "ALL_TOOL_DEFINITIONS",
]
