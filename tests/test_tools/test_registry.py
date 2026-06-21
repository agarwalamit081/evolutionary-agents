"""Tests for src.tools.registry.ToolRegistry and create_default_registry."""

from __future__ import annotations



from src.tools import create_default_registry
from src.tools.registry import ToolRegistry


# ─── Helpers ──────────────────────────────────────────────────────────


async def _dummy_handler(code: str) -> str:
    """A minimal async handler used for test registrations."""
    return f"executed: {code}"


# ─── ToolRegistry Tests ──────────────────────────────────────────────


class TestToolRegistry:
    """Unit tests for ToolRegistry CRUD operations."""

    def test_register_tool(self) -> None:
        """Register a tool and verify it appears in list_tools."""
        registry = ToolRegistry()
        registry.register(
            name="test_tool",
            handler=_dummy_handler,
            description="A test tool",
            parameters={"type": "object", "properties": {"code": {"type": "string"}}},
        )

        tools = registry.list_tools()
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "test_tool"
        assert tools[0]["function"]["description"] == "A test tool"

    def test_get_handler_returns_callable(self) -> None:
        """get_handler returns the registered function."""
        registry = ToolRegistry()
        registry.register(
            name="test_tool",
            handler=_dummy_handler,
            description="A test tool",
        )

        handler = registry.get_handler("test_tool")
        assert handler is not None
        assert callable(handler)
        assert handler is _dummy_handler

    def test_get_handler_unknown_tool_returns_none(self) -> None:
        """Unregistered tool returns None from get_handler."""
        registry = ToolRegistry()
        result = registry.get_handler("nonexistent_tool")
        assert result is None


class TestCreateDefaultRegistry:
    """Tests for the create_default_registry factory function."""

    def test_create_default_registry_has_sixteen_tools(self) -> None:
        """create_default_registry returns a registry with all 16 built-in tools."""
        registry = create_default_registry()
        assert isinstance(registry, ToolRegistry)
        assert registry.count == 16

        names = registry.list_names()
        assert "code_executor" in names
        assert "code_validator" in names
        assert "file_reader" in names
        assert "file_writer" in names
        assert "memory_search" in names
        assert "self_inspect" in names
        assert "web_search" in names
        # New tools (WS2/WS3/WS4)
        assert "get_current_time" in names
        assert "environment_inspect" in names
        assert "list_directory" in names
        assert "web_scraper" in names
        assert "document_parser" in names
        assert "http_request" in names
        assert "terminal_command" in names
