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


class TestExecuteSystemRunFilesGuidance:
    """battery-04 q4 (Part C): the execute system prompt must instruct the agent
    to invoke a file it wrote by its EXACT written path and to record REAL
    pass/fail counts — never a status=failed placeholder.

    The q4 orchestrator wrote ``test_suite.py`` but ran ``pytest
    test_retention_csv`` (filename mismatch), then shipped a
    ``{"status": "failed"}`` stub. This guidance (shared by the main agent and
    every sub-agent, which reuse ``execute_node``) is the prompt-level fix;
    the eval golden check (Part B) is the hard backstop. Guards against
    accidental removal of the guidance.
    """

    def _rendered(self) -> str:
        from src.graph.prompts import EXECUTE_SYSTEM

        return EXECUTE_SYSTEM.format(
            goal_text="g",
            completed_count=0,
            total_steps=1,
            step_description="s",
            memory_context="",
            tool_results_context="",
        )

    def test_requires_exact_written_path(self) -> None:
        rendered = self._rendered()
        assert "EXACT written path" in rendered
        # The "No module named" symptom must be called out as a filename error.
        assert "No module named" in rendered

    def test_forbids_failure_status_placeholder(self) -> None:
        rendered = self._rendered()
        assert "real pass/fail counts" in rendered
        assert "placeholder" in rendered
        assert '"status": "failed"' in rendered or "status\": \"failed" in rendered

    def test_states_project_root_default_cwd(self) -> None:
        rendered = self._rendered()
        # Relative results/ paths resolve because the tools run from project root.
        assert "project root" in rendered


# ─── A2a (Phase 3.5): builder prefix-stability for provider-native caching ──


class TestBuildMessagesPrefixStability:
    """Provider-native PREFIX caching (OpenAI/DeepSeek/Z.AI auto-cache) engages
    only when the system-prompt prefix is byte-identical across calls in a run.

    ``build_messages`` is a pure function of (base_system, techniques, node): it
    injects no per-call timestamp / run_id / randomness (builder.py has none of
    datetime/uuid/random). These lock that purity — the prerequisite that lets
    every auto-caching provider engage on the shared system block, the dominant
    per-call input-token lever for A2. Companion to the cache-hit observability in
    gateway.py (record_prompt_cache_tokens parses OpenAI ``cached_tokens`` /
    DeepSeek ``prompt_cache_hit_tokens`` / Anthropic ``_cache_read_input_tokens``).
    """

    def test_identical_args_yield_byte_identical_system_block(self) -> None:
        from src.graph.prompts.builder import build_messages, clear_evolved_candidate
        from src.graph.prompts.technique_selector import NODE_EXECUTE

        clear_evolved_candidate(None)  # promotion OFF by default → deterministic
        techniques = _techniques_for_critical_plan()
        m1 = build_messages(
            _SYSTEM_WITH_MARKER, "user turn A", techniques=techniques, node=NODE_EXECUTE
        )
        m2 = build_messages(
            _SYSTEM_WITH_MARKER, "user turn A", techniques=techniques, node=NODE_EXECUTE
        )
        assert m1[0]["content"] == m2[0]["content"]  # system block identical
        assert m1 == m2  # whole message list identical

    def test_system_prefix_invariant_to_varying_user_turn(self) -> None:
        """The cacheable system prefix stays byte-identical as the per-call user
        turn varies — the whole point of prefix caching."""
        from src.graph.prompts.builder import build_messages, clear_evolved_candidate
        from src.graph.prompts.technique_selector import NODE_PLAN

        clear_evolved_candidate(None)
        m1 = build_messages(_SYSTEM_WITH_MARKER, "user turn 1", node=NODE_PLAN)
        m2 = build_messages(_SYSTEM_WITH_MARKER, "DIFFERENT user turn 2", node=NODE_PLAN)
        assert m1[0]["content"] == m2[0]["content"]  # system prefix invariant
        assert m1[1]["content"] != m2[1]["content"]  # only the user tail differs

    def test_no_techniques_is_byte_stable(self) -> None:
        from src.graph.prompts.builder import build_messages, clear_evolved_candidate

        clear_evolved_candidate(None)
        m1 = build_messages(_SYSTEM_NO_MARKER, "u1")
        m2 = build_messages(_SYSTEM_NO_MARKER, "u1")
        assert m1[0]["content"] == m2[0]["content"]

