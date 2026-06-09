"""MCP adapter — bridges MCP tool servers into the ToolRegistry.

Uses langchain-mcp-adapters to connect to MCP servers and register
their tools in the agent's ToolRegistry for use during execution.
"""

from __future__ import annotations

from typing import Any

from loguru import logger


class MCPToolAdapter:
    """Adapts MCP server tools into the ToolRegistry format.

    Connects to MCP servers via langchain-mcp-adapters and converts
    each MCP tool into a registered tool callable by the agent.
    """

    def __init__(self, registry: Any) -> None:
        self._registry = registry

    async def load_server(
        self,
        server_command: list[str],
        server_name: str = "mcp",
        env: dict[str, str] | None = None,
    ) -> list[str]:
        """Load tools from an MCP server and register them.

        Args:
            server_command: Command to start the MCP server (e.g., ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"]).
            server_name: Name prefix for the tools.
            env: Optional environment variables for the server process.

        Returns:
            List of registered tool names.
        """
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient

            client = MultiServerMCPClient(
                {server_name: {"command": server_command[0], "args": server_command[1:], "env": env}},  # type: ignore[arg-type]
            )

            tools = await client.get_tools()  # type: ignore[misc]
            registered: list[str] = []

            for tool in tools:
                name = f"mcp_{server_name}_{tool.name}"
                description = tool.description or f"MCP tool: {tool.name}"

                self._registry.register(
                    name=name,
                    handler=self._wrap_mcp_tool(client, tool.name),
                    description=description,
                    parameters=tool.inputSchema if hasattr(tool, "inputSchema") else getattr(tool, "args_schema", {}),  # type: ignore[union-attr]
                )
                registered.append(name)

            logger.info(f"Loaded {len(registered)} tools from MCP server '{server_name}'")
            return registered
        except ImportError:
            logger.warning("langchain-mcp-adapters not installed, MCP tools unavailable")
            return []
        except Exception as e:
            logger.warning(f"Failed to load MCP server '{server_name}': {e}")
            return []

    @staticmethod
    def _wrap_mcp_tool(client: Any, tool_name: str) -> Any:
        """Create an async handler that calls an MCP tool via the client."""

        async def _handler(**kwargs: Any) -> str:
            try:
                result = await client.call_tool(tool_name, kwargs)
                if isinstance(result, list):
                    parts = []
                    for item in result:
                        if hasattr(item, "text"):
                            parts.append(item.text)
                        else:
                            parts.append(str(item))
                    return "\n".join(parts)
                return str(result)
            except Exception as e:
                return f"Error calling MCP tool {tool_name}: {e}"

        return _handler
