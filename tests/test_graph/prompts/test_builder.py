"""Tests for src.graph.prompts.builder — §5 prompt-construction layer.

Covers: passthrough with no techniques, splicing above the JSON-schema
footer (schema stays intact), and the no-marker fallback (insert after the
first paragraph). Also confirms build_messages returns a [system, user] pair.
"""

from __future__ import annotations

from src.graph.enums import TaskComplexity
from src.graph.prompts.builder import build_messages, splice_techniques
from src.graph.prompts.technique_selector import (
    JSON_SCHEMA_MARKER,
    NODE_PLAN,
    Technique,
    TechniqueSelector,
)

_SYSTEM_WITH_MARKER = (
    "You are a planning system.\n\n"
    "Plan methodically.\n\n"
    "Respond with a JSON object matching this schema:\n"
    "- steps: array\n"
)

_SYSTEM_NO_MARKER = (
    "You are an executor.\n\n"
    "Use the available tools to make progress.\n"
)


def _techniques_for_critical_plan() -> list[Technique]:
    return TechniqueSelector().select(
        complexity=TaskComplexity.CRITICAL, node=NODE_PLAN, goal_pattern="math",
    )


class TestSpliceTechniques:
    """splice_techniques injects bodies without breaking the schema footer."""

    def test_no_techniques_is_passthrough(self) -> None:
        assert splice_techniques(_SYSTEM_WITH_MARKER, []) == _SYSTEM_WITH_MARKER

    def test_bodies_appear_above_json_marker(self) -> None:
        techniques = _techniques_for_critical_plan()
        result = splice_techniques(_SYSTEM_WITH_MARKER, techniques)

        marker_pos = result.find(JSON_SCHEMA_MARKER)
        block_pos = result.find("Reasoning techniques to apply:")
        assert marker_pos != -1
        assert block_pos != -1
        # The block must precede the schema footer.
        assert block_pos < marker_pos

    def test_schema_footer_remains_intact(self) -> None:
        """The JSON schema lines survive the splice unchanged (extract() works)."""
        techniques = _techniques_for_critical_plan()
        result = splice_techniques(_SYSTEM_WITH_MARKER, techniques)

        assert "Respond with a JSON object matching this schema:" in result
        assert "- steps: array" in result

    def test_technique_bodies_present(self) -> None:
        techniques = _techniques_for_critical_plan()
        result = splice_techniques(_SYSTEM_WITH_MARKER, techniques)
        for technique in techniques:
            assert technique.body in result

    def test_no_marker_inserts_after_first_paragraph(self) -> None:
        """Without a JSON footer, guidance lands after the opening paragraph."""
        techniques = _techniques_for_critical_plan()
        result = splice_techniques(_SYSTEM_NO_MARKER, techniques)

        block_pos = result.find("Reasoning techniques to apply:")
        # Opening line stays first; the block follows it.
        assert result.startswith("You are an executor.")
        assert block_pos != -1
        # The original tool-use line still appears after the block.
        assert "Use the available tools to make progress." in result


class TestBuildMessages:
    """build_messages returns a [system, user] pair with techniques spliced in."""

    def test_returns_system_user_pair(self) -> None:
        messages = build_messages(_SYSTEM_WITH_MARKER, "do the thing", None)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "do the thing"

    def test_no_techniques_passes_system_through(self) -> None:
        messages = build_messages(_SYSTEM_WITH_MARKER, "u")
        assert messages[0]["content"] == _SYSTEM_WITH_MARKER

    def test_techniques_spliced_into_system(self) -> None:
        techniques = _techniques_for_critical_plan()
        messages = build_messages(_SYSTEM_WITH_MARKER, "u", techniques)
        assert "Reasoning techniques to apply:" in messages[0]["content"]
        # User content untouched.
        assert messages[1]["content"] == "u"
