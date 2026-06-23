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
        cacheable: bool = False,
        generated: bool = False,
        handler_code: str | None = None,
    ) -> None:
        """Register a tool in the registry.

        Args:
            name: Unique tool identifier.
            handler: Async callable that implements the tool.
            description: Human-readable description for LLM tool selection.
            parameters: JSON Schema describing tool parameters.
            cacheable: When True, successful results of this tool are eligible
                for the Redis result cache. Reserve this for idempotent,
                read-only tools (e.g. ``web_search``, ``file_reader``). NEVER
                mark mutating tools (``file_writer``) cacheable.
            generated: True for an LLM-generated dynamic tool (``tool_create`` →
                ``ToolGenerator``) whose ``handler`` is materialized from
                ``handler_code``. Such tools' handler_code is UNTRUSTED LLM
                output; in a sandboxed code-exec mode (docker/runner) the
                execute node routes their invocation through that sandbox (the
                SAME surface ``code_executor`` uses) instead of calling the
                in-process ``handler`` — closing the gap where generated code
                otherwise runs inside the worker with full DB/Redis/FS access.
                Hand-written builtins leave this False and always run in-process
                (they are trusted and need gateway/Redis access).
            handler_code: The Python source the handler was materialized from.
                Required when ``generated=True`` (the dispatch needs it to build
                the sandboxed driver); ignored otherwise.
        """
        if name in self._tools:
            logger.warning(f"Tool '{name}' already registered, overwriting")

        self._tools[name] = {
            "name": name,
            "handler": handler,
            "description": description,
            "parameters": parameters or {},
            "cacheable": cacheable,
            "generated": generated,
            "handler_code": handler_code if generated else None,
        }
        logger.debug(f"Tool registered: {name}")

    def tool(
        self,
        name: str | None = None,
        description: str = "",
        parameters: dict[str, Any] | None = None,
        cacheable: bool = False,
    ) -> Callable[..., Any]:
        """Decorator to register a function as a tool.

        Args:
            name: Optional tool name (defaults to function name).
            description: Tool description for LLM.
            parameters: JSON Schema for parameters.
            cacheable: Whether successful results may be cached.

        Returns:
            Decorator function.
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = name or func.__name__
            tool_desc = description or func.__doc__ or ""
            self.register(tool_name, func, tool_desc, parameters, cacheable)
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

    def is_cacheable(self, name: str) -> bool:
        """Return whether a tool's successful results may be cached.

        Unknown tools (and any tool registered without ``cacheable=True``)
        return False, so the cache hook is a safe no-op for everything but
        the explicit idempotent read-only tools.
        """
        tool = self._tools.get(name)
        return bool(tool and tool.get("cacheable"))

    def get_handler(self, name: str) -> Callable[..., Any] | None:
        """Get just the handler function for a tool."""
        tool = self._tools.get(name)
        return tool["handler"] if tool else None

    def is_generated(self, name: str) -> bool:
        """Return whether ``name`` is an LLM-generated dynamic tool.

        Generated tools carry untrusted ``handler_code``; in a sandboxed
        code-exec mode the execute node routes their invocation through that
        sandbox instead of the in-process ``handler``. Hand-written builtins
        and MCP-loaded tools are False and always run in-process.
        """
        tool = self._tools.get(name)
        return bool(tool and tool.get("generated"))

    def get_handler_code(self, name: str) -> str | None:
        """Return the materialized source of a generated tool, or None.

        None for unknown tools and non-generated tools (builtins have no
        source — their handler is hand-written in the worker package).
        """
        tool = self._tools.get(name)
        if not tool or not tool.get("generated"):
            return None
        return tool.get("handler_code")

    def list_tools(self, names: list[str] | None = None) -> list[dict[str, Any]]:
        """List registered tools (without handlers, for LLM consumption).

        Args:
            names: When given, restrict the result to these tool names (in
                registry order). Unknown names are silently skipped. ``None``
                (default) returns every registered tool — the historical
                behavior, used everywhere retrieval is disabled.

        Returns:
            List of tool descriptors suitable for bind_tools().
        """
        wanted = set(names) if names is not None else None
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
            if wanted is None or t["name"] in wanted
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

    @property
    def generated_count(self) -> int:
        """Number of LLM-generated dynamic tools currently registered.

        This is the active *dynamic-tool* population: builtins and MCP-loaded
        tools (``generated=False``) are excluded, so it mirrors exactly what
        ``AgentSettings.max_active_tools`` caps at the governance layer. Used by
        ``ToolGenerator.validate_and_register``'s pre-register active-population
        gate (findings.md A3) so a single run cannot push the generated-tool
        count past the cap mid-run.
        """
        return sum(1 for t in self._tools.values() if t.get("generated"))
