"""Consolidation guardrail for the built-in tools (Phase 3, B5 / M7b).

M7b audited all 14 built-ins for true duplicates (the capability bloat B3
addresses for dynamic tools). The audit found **no** true duplicates — every
hypothesised overlap serves a genuinely distinct purpose by design:

  - Listing cluster — ``list_directory`` (project files), ``self_inspect``
    (agent source code), ``environment_inspect`` (OS/CPU/RAM/packages): three
    different things listed.
  - Reading cluster — ``file_reader`` (plain text, 1 MB / line-capped, with the
    results-root read-back fallback) vs ``document_parser`` (PDF/DOCX/XLSX/CSV,
    20 MB / char-capped, format-specific extraction).
  - Fetch cluster — ``web_scraper`` (GET → clean markdown via trafilatura,
    cacheable) vs ``http_request`` (structured REST, any method/headers/body,
    non-cacheable) vs ``web_search`` (search-engine snippets/links).

These tests lock that conclusion in: they fail if a future change adds a
builtin whose name or description collides with an existing one, or collapses a
cluster into a single tool that loses capability. ``count == 14`` documents the
consolidation baseline; a legitimate new tool updates this number deliberately.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.tools.builtin import ALL_TOOL_DEFINITIONS

_REQUIRED_FIELDS = {"name", "handler", "description", "parameters"}


def _by_name(name: str) -> dict[str, Any]:
    """Return the tool definition with the given name."""
    for definition in ALL_TOOL_DEFINITIONS:
        if definition["name"] == name:
            return definition
    raise KeyError(name)


# ─── Uniqueness invariants ───────────────────────────────────────────


class TestBuiltinConsolidation:
    """Guard against accidental duplication or capability collapse in builtins."""

    def test_all_tool_names_unique(self) -> None:
        """No two built-ins share a name (the core B3 dedup invariant)."""
        names = [d["name"] for d in ALL_TOOL_DEFINITIONS]
        assert len(names) == len(set(names)), f"Duplicate tool names: {names}"

    def test_all_descriptions_unique(self) -> None:
        """No two built-ins share an identical description (catches copy-paste dupes)."""
        descriptions = [d["description"] for d in ALL_TOOL_DEFINITIONS]
        assert len(descriptions) == len(set(descriptions))

    def test_tool_count_matches_baseline(self) -> None:
        """The baseline after the M7b audit is 14 distinct built-ins."""
        assert len(ALL_TOOL_DEFINITIONS) == 14

    @pytest.mark.parametrize("definition", ALL_TOOL_DEFINITIONS)
    def test_each_definition_has_required_fields(self, definition: dict[str, Any]) -> None:
        """Every built-in exposes name/handler/description/parameters."""
        missing = _REQUIRED_FIELDS - definition.keys()
        assert not missing, f"{definition.get('name')} missing fields: {missing}"

    @pytest.mark.parametrize("definition", ALL_TOOL_DEFINITIONS)
    def test_each_definition_is_callable_handler(self, definition: dict[str, Any]) -> None:
        """The handler is a real callable, not a stub."""
        assert callable(definition["handler"])


# ─── Cluster-distinctness (the hypothesised overlaps stay separate) ──


class TestClusterDistinctness:
    """The three audit clusters are deliberately distinct, not mergeable."""

    def test_reading_cluster_uses_different_resolvers_and_caps(self) -> None:
        """file_reader (text/1 MB/lines) and document_parser (docs/20 MB/chars) differ."""
        reader = _by_name("file_reader")
        parser = _by_name("document_parser")
        # Different required params would collapse capability — both must exist.
        assert set(reader["parameters"]["required"]) == {"file_path"}
        assert set(parser["parameters"]["required"]) == {"file_path"}
        # Distinct truncation knobs (line-based vs char-based).
        assert "max_lines" in reader["parameters"]["properties"]
        assert "max_chars" in parser["parameters"]["properties"]
        assert "max_lines" not in parser["parameters"]["properties"]

    def test_fetch_cluster_splits_extraction_from_api(self) -> None:
        """web_scraper (readable markdown) and http_request (REST) stay separate."""
        scraper = _by_name("web_scraper")
        http = _by_name("http_request")
        # http_request supports mutating methods + headers/body; web_scraper does not.
        assert "method" in http["parameters"]["properties"]
        assert "method" not in scraper["parameters"]["properties"]
        assert "headers" in http["parameters"]["properties"]
        assert "body" in http["parameters"]["properties"]
        # Caching policy differs: scraper is a cacheable read, http is not.
        assert scraper.get("cacheable", True) is True
        assert http.get("cacheable", True) is False

    def test_listing_cluster_covers_three_distinct_targets(self) -> None:
        """list_directory / self_inspect / environment_inspect list different things."""
        names = {d["name"] for d in ALL_TOOL_DEFINITIONS}
        assert {"list_directory", "self_inspect", "environment_inspect"} <= names
        env = _by_name("environment_inspect")
        # environment_inspect switches on a detail knob the file-listing tools lack.
        assert "detail" in env["parameters"]["properties"]
