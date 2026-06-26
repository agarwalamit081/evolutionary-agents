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

    def test_generated_count_excludes_non_generated(self) -> None:
        """A3: generated_count is the active dynamic-tool population — it counts
        only LLM-generated tools (generated=True), excluding builtins/MCP
        (generated=False). This is the measure ToolGenerator's pre-register cap
        gates on, mirroring AgentSettings.max_active_tools."""
        registry = ToolRegistry()
        registry.register(name="builtin_a", handler=_dummy_handler)
        registry.register(name="builtin_b", handler=_dummy_handler, generated=False)
        registry.register(name="gen_a", handler=_dummy_handler, generated=True)
        registry.register(name="gen_b", handler=_dummy_handler, generated=True)

        assert registry.count == 4
        assert registry.generated_count == 2


class TestCreateDefaultRegistry:
    """Tests for the create_default_registry factory function."""

    def test_create_default_registry_has_twenty_tools(self) -> None:
        """create_default_registry returns a registry with all 20 built-in tools."""
        registry = create_default_registry()
        assert isinstance(registry, ToolRegistry)
        assert registry.count == 20

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


# ─── F3: tags / MCP hints / destructive accessors ────────────────────


class TestToolAnnotations:
    """F3 — category tags + MCP hints + destructive accessor behavior."""

    def test_get_tags_and_hints_round_trip(self) -> None:
        """Registered tags/hints are returned verbatim by the accessors."""
        registry = ToolRegistry()
        registry.register(
            name="my_tool",
            handler=_dummy_handler,
            tags=["read", "network"],
            mcp_hints={"readOnlyHint": True, "openWorldHint": True},
        )
        assert registry.get_tags("my_tool") == ["read", "network"]
        assert registry.get_mcp_hints("my_tool") == {
            "readOnlyHint": True,
            "openWorldHint": True,
        }
        assert registry.is_destructive("my_tool") is False

    def test_destructive_flag_from_hint(self) -> None:
        """is_destructive reflects the destructiveHint MCP annotation."""
        registry = ToolRegistry()
        registry.register(
            name="rmrf",
            handler=_dummy_handler,
            mcp_hints={"destructiveHint": True, "openWorldHint": True},
        )
        assert registry.is_destructive("rmrf") is True

    def test_unknown_tool_accessors_are_safe_defaults(self) -> None:
        """Unknown / un-annotated tools return empty tags + hints + non-destructive."""
        registry = ToolRegistry()
        registry.register(name="plain", handler=_dummy_handler)
        assert registry.get_tags("plain") == []
        assert registry.get_mcp_hints("plain") == {}
        assert registry.is_destructive("plain") is False
        # And for a name that isn't registered at all:
        assert registry.get_tags("nope") == []
        assert registry.get_mcp_hints("nope") == {}
        assert registry.is_destructive("nope") is False

    def test_create_default_registry_applies_annotations(self) -> None:
        """The central TOOL_ANNOTATIONS map is threaded into the default registry."""
        registry = create_default_registry()
        # terminal_command is flagged destructive in the central map.
        assert registry.is_destructive("terminal_command") is True
        assert registry.is_destructive("http_request") is True
        assert registry.is_destructive("index_corpus") is True
        # file_writer is path-confined and intentionally NOT destructive.
        assert registry.is_destructive("file_writer") is False
        assert registry.is_destructive("code_executor") is False
        # Tags are populated from the map.
        assert "search" in registry.get_tags("web_search")
        assert "read" in registry.get_tags("file_reader")
        assert "write" in registry.get_tags("file_writer")
        # Read-only hint present on a pure read tool.
        assert registry.get_mcp_hints("file_reader").get("readOnlyHint") is True
