"""Dynamic tool registry with @tool decorator support."""

from __future__ import annotations

from typing import Any, Callable

from loguru import logger


class ToolRegistry:
    """Registry for managing available tools.

    Tools can be registered via decorator or direct registration.
    Each tool must have a name, description, and callable handler.
    """

    def __init__(self) -> None:
        self._tools: dict[str, dict[str, Any]] = {}

    def register(
        self,
        name: str,
        handler: Callable[..., Any],
        description: str = "",
        parameters: dict[str, Any] | None = None,
    ) -> None:
        """Register a tool in the registry.

        Args:
            name: Unique tool identifier.
            handler: Async callable that implements the tool.
            description: Human-readable description for LLM tool selection.
            parameters: JSON Schema describing tool parameters.
        """
        if name in self._tools:
            logger.warning(f"Tool '{name}' already registered, overwriting")

        self._tools[name] = {
            "name": name,
            "handler": handler,
            "description": description,
            "parameters": parameters or {},
        }
        logger.debug(f"Tool registered: {name}")

    def tool(
        self,
        name: str | None = None,
        description: str = "",
        parameters: dict[str, Any] | None = None,
    ) -> Callable[..., Any]:
        """Decorator to register a function as a tool.

        Args:
            name: Optional tool name (defaults to function name).
            description: Tool description for LLM.
            parameters: JSON Schema for parameters.

        Returns:
            Decorator function.
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = name or func.__name__
            tool_desc = description or func.__doc__ or ""
            self.register(tool_name, func, tool_desc, parameters)
            return func

        return decorator

    def get(self, name: str) -> dict[str, Any] | None:
        """Get a tool by name.

        Args:
            name: Tool identifier.

        Returns:
            Tool dict with name, handler, description, parameters or None.
        """
        return self._tools.get(name)

    def get_handler(self, name: str) -> Callable[..., Any] | None:
        """Get just the handler function for a tool."""
        tool = self._tools.get(name)
        return tool["handler"] if tool else None

    def list_tools(self) -> list[dict[str, Any]]:
        """List all registered tools (without handlers, for LLM consumption).

        Returns:
            List of tool descriptors suitable for bind_tools().
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
            for t in self._tools.values()
        ]

    def list_names(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def unregister(self, name: str) -> bool:
        """Remove a tool from the registry.

        Returns:
            True if the tool was found and removed.
        """
        if name in self._tools:
            del self._tools[name]
            logger.debug(f"Tool unregistered: {name}")
            return True
        return False

    @property
    def count(self) -> int:
        """Number of registered tools."""
        return len(self._tools)
