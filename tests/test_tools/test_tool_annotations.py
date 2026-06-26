"""F3 — completeness + invariants for the central TOOL_ANNOTATIONS map.

Locks the single source of truth for builtin category tags + MCP hints so a
future edit can't silently drop a tool, add a bogus hint key, or widen the
destructive set without this test forcing a deliberate update.
"""

from __future__ import annotations

from src.tools.builtin import ALL_TOOL_DEFINITIONS, TOOL_ANNOTATIONS

# The four canonical MCP boolean annotations (per the MCP spec); any other key
# in the map is almost certainly a typo.
_MCP_HINT_KEYS = {"readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint"}

# The deliberately-flagged destructive tools. file_writer (path-confined) and
# code_executor (runner-sandboxed) are intentionally excluded — their blast
# radius is already bounded.
_EXPECTED_DESTRUCTIVE = {"terminal_command", "http_request", "index_corpus"}


def test_every_builtin_tool_is_annotated() -> None:
    """Each builtin tool must have an entry in the central annotation map."""
    annotated = set(TOOL_ANNOTATIONS)
    for tool_def in ALL_TOOL_DEFINITIONS:
        assert tool_def["name"] in annotated, f"missing annotation for {tool_def['name']}"


def test_no_annotation_keys_outside_the_four_mcp_hints() -> None:
    """Every mcp_hints map uses only the four canonical MCP hint keys."""
    for name, annotation in TOOL_ANNOTATIONS.items():
        hints = annotation.get("mcp_hints", {})
        assert isinstance(hints, dict), f"{name} mcp_hints not a dict"
        bad = set(hints) - _MCP_HINT_KEYS
        assert not bad, f"{name} has unknown MCP hint keys: {bad}"


def test_destructive_set_is_exactly_the_intended_three() -> None:
    """Only terminal_command/http_request/index_corpus are flagged destructive."""
    flagged: set[str] = set()
    for name, annotation in TOOL_ANNOTATIONS.items():
        hints = annotation.get("mcp_hints", {})
        if isinstance(hints, dict) and hints.get("destructiveHint"):
            flagged.add(name)
    assert flagged == _EXPECTED_DESTRUCTIVE


def test_tags_are_non_empty_strings() -> None:
    """Tag values are non-empty strings (for scope-injection recall)."""
    for name, annotation in TOOL_ANNOTATIONS.items():
        tags = annotation.get("tags", [])
        assert isinstance(tags, list), f"{name} tags not a list"
        for tag in tags:
            assert isinstance(tag, str) and tag, f"{name} has empty/non-str tag: {tag!r}"
