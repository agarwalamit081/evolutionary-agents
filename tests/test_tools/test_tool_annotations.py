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


# ── Phase 3.5 A8 (rec #16): tool-description/schema token-budget invariant ──
# The ~23 always-on builtins are injected into EVERY execute/plan prompt, so
# their aggregate schema is a fixed per-call token tax. Rec #16 proposed trimming
# descriptions from "~500 tok"; this audit found they are ALREADY lean
# (max code_executor @ 797c ≈ 199 tok; web_search full-schema ≈ 610 tok incl. its
# 10 params). These budgets lock that leanness so a future edit can't silently
# re-bloat a description and inflate every call without forcing a deliberate
# update. The real per-call lever is the always-on COUNT (governance/retire) +
# RAG-over-tools top-K (A7), not description prose. Headroom is generous so a
# legitimate new param doesn't trip the gate.
_DESC_BUDGET_CHARS = 900      # ≈ 225 tok (code_executor 797c today)
_SCHEMA_BUDGET_CHARS = 2600   # ≈ 650 tok (web_search 2433c today)


def _schema_view(td: dict) -> dict:
    return {k: v for k, v in td.items() if k in ("name", "description", "parameters")}


def test_builtin_descriptions_under_char_budget() -> None:
    """No builtin description exceeds the token budget (locks rec #16 leanness)."""
    over = []
    for td in ALL_TOOL_DEFINITIONS:
        desc = td.get("description", "")
        if len(desc) > _DESC_BUDGET_CHARS:
            over.append((td.get("name", "?"), len(desc)))
    assert not over, f"descriptions over {_DESC_BUDGET_CHARS}c: {over}"


def test_builtin_schemas_under_char_budget() -> None:
    """No builtin's full LLM-facing schema (name+description+parameters) exceeds
    the budget — the per-call token tax stays bounded."""
    import json

    over = []
    for td in ALL_TOOL_DEFINITIONS:
        size = len(json.dumps(_schema_view(td), default=str))
        if size > _SCHEMA_BUDGET_CHARS:
            over.append((td.get("name", "?"), size))
    assert not over, f"schemas over {_SCHEMA_BUDGET_CHARS}c: {over}"


def test_descriptions_leak_no_internal_env_flags() -> None:
    """Descriptions must not leak internal implementation detail (mcp-patterns
    rule: no internal stack traces / config-flag names). Catches the
    DEEP_CRAWL_ENABLED-style leak A8 removed from web_search. Only flags
    ALL-CAPS underscored names (env-var style); lowercase parameter hints like
    ``multi_query=true`` are legitimate and left through."""
    import re

    _ENV_FLAG = re.compile(r"\b[A-Z][A-Z0-9_]*_[A-Z0-9_]+=(?:true|false)\b")
    leaks = []
    for td in ALL_TOOL_DEFINITIONS:
        desc = td.get("description", "")
        # parameter descriptions too
        params = td.get("parameters", {})
        blobs = [desc]
        if isinstance(params, dict):
            for p in params.get("properties", {}).values():
                if isinstance(p, dict) and isinstance(p.get("description"), str):
                    blobs.append(p["description"])
        for i, blob in enumerate(blobs):
            m = _ENV_FLAG.search(blob)
            if m:
                leaks.append((td.get("name", "?"), i, m.group(0)))
    assert not leaks, f"internal env-flag leaks in descriptions: {leaks}"
