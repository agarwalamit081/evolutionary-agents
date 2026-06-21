"""Tests for src.evolution.templates — mutation template generators."""

from __future__ import annotations

import json


from src.evolution.templates import (
    generate_code_improvement,
    generate_config_tuning,
    generate_memory_config,
    generate_prompt_improvement,
    generate_tool_config,
    generate_workflow_config,
)


class TestPromptImprovement:
    """Tests for generate_prompt_improvement."""

    def test_returns_valid_structure(self) -> None:
        """Template returns content, target_path, and rationale."""
        result = generate_prompt_improvement(patterns=["json parse error"])
        assert "content" in result
        assert "target_path" in result
        assert "rationale" in result

    def test_content_is_valid_json(self) -> None:
        """Content field contains parseable JSON."""
        result = generate_prompt_improvement(patterns=["json parse error"])
        parsed = json.loads(result["content"])
        assert "suffixes" in parsed
        assert "reason" in parsed
        assert "target_node" in parsed

    def test_json_pattern_matches_json_fix(self) -> None:
        """Pattern containing 'json' matches the JSON prompt fix."""
        result = generate_prompt_improvement(patterns=["invalid JSON from LLM"])
        parsed = json.loads(result["content"])
        assert len(parsed["suffixes"]) >= 1
        assert "JSON" in parsed["suffixes"][0]

    def test_timeout_pattern_matches_timeout_fix(self) -> None:
        """Pattern containing 'timeout' matches the timeout prompt fix."""
        result = generate_prompt_improvement(patterns=["timeout on tool call"])
        parsed = json.loads(result["content"])
        assert any("sub-tasks" in s or "retrying" in s for s in parsed["suffixes"])

    def test_unknown_pattern_uses_default_fix(self) -> None:
        """Unknown pattern uses the default prompt improvement."""
        result = generate_prompt_improvement(patterns=["something totally unknown"])
        parsed = json.loads(result["content"])
        assert len(parsed["suffixes"]) >= 1

    def test_empty_patterns_uses_default_fix(self) -> None:
        """Empty patterns list uses the default prompt improvement."""
        result = generate_prompt_improvement(patterns=[])
        parsed = json.loads(result["content"])
        assert len(parsed["suffixes"]) == 1
        assert "reasoning" in parsed["suffixes"][0].lower() or "careful" in parsed["suffixes"][0].lower()

    def test_multiple_patterns_produce_multiple_suffixes(self) -> None:
        """Multiple distinct patterns produce multiple prompt suffixes."""
        result = generate_prompt_improvement(patterns=["json parse error", "timeout on call"])
        parsed = json.loads(result["content"])
        assert len(parsed["suffixes"]) >= 2

    def test_target_path_is_set(self) -> None:
        """Target path is always set (never None)."""
        result = generate_prompt_improvement(patterns=["error"])
        assert result["target_path"] == "evolution/prompt_improvements.json"

    def test_rationale_is_non_empty(self) -> None:
        """Rationale describes the improvement."""
        result = generate_prompt_improvement(patterns=["json error"])
        assert len(result["rationale"]) > 20


class TestWorkflowConfig:
    """Tests for generate_workflow_config."""

    def test_speed_description_uses_reduce_time_strategy(self) -> None:
        """Description mentioning 'time' uses reduce_execution_time strategy."""
        result = generate_workflow_config("Reduce execution time")
        parsed = json.loads(result["content"])
        assert parsed["strategy"] == "reduce_execution_time"

    def test_accuracy_description_uses_improve_accuracy_strategy(self) -> None:
        """Description mentioning 'accuracy' uses improve_accuracy strategy."""
        result = generate_workflow_config("Improve accuracy of results")
        parsed = json.loads(result["content"])
        assert parsed["strategy"] == "improve_accuracy"

    def test_generic_description_uses_balance_strategy(self) -> None:
        """Generic description uses balance_speed_accuracy strategy."""
        result = generate_workflow_config("General optimization")
        parsed = json.loads(result["content"])
        assert parsed["strategy"] == "balance_speed_accuracy"

    def test_target_path_is_set(self) -> None:
        result = generate_workflow_config("test")
        assert result["target_path"] == "evolution/workflow_config.json"


class TestToolConfig:
    """Tests for generate_tool_config."""

    def test_code_executor_selection(self) -> None:
        result = generate_tool_config("Optimize code execution speed")
        parsed = json.loads(result["content"])
        assert parsed["tool"] == "code_executor"

    def test_web_search_selection(self) -> None:
        result = generate_tool_config("Improve web search results")
        parsed = json.loads(result["content"])
        assert parsed["tool"] == "web_search"

    def test_default_memory_search_selection(self) -> None:
        result = generate_tool_config("General improvement")
        parsed = json.loads(result["content"])
        assert parsed["tool"] == "memory_search"

    def test_target_path_is_set(self) -> None:
        result = generate_tool_config("test")
        assert result["target_path"] == "evolution/tool_config.json"


class TestMemoryConfig:
    """Tests for generate_memory_config."""

    def test_precision_strategy(self) -> None:
        result = generate_memory_config("Reduce noise in memory precision results")
        parsed = json.loads(result["content"])
        assert parsed["strategy"] == "precision_focused"
        assert parsed["min_fitness"] == 0.6

    def test_recall_strategy(self) -> None:
        result = generate_memory_config("Miss important context recall issues")
        parsed = json.loads(result["content"])
        assert parsed["strategy"] == "recall_focused"
        assert parsed["max_results"] == 7

    def test_balanced_strategy(self) -> None:
        result = generate_memory_config("General memory tuning")
        parsed = json.loads(result["content"])
        assert parsed["strategy"] == "balanced"

    def test_target_path_is_set(self) -> None:
        result = generate_memory_config("test")
        assert result["target_path"] == "evolution/memory_config.json"


class TestCodeImprovement:
    """Tests for generate_code_improvement (real, loadable code — B2)."""

    def test_emits_valid_python(self) -> None:
        import ast

        result = generate_code_improvement("Optimize loop")
        assert result["target_path"].endswith(".py")
        ast.parse(result["content"])  # raises if not valid Python

    def test_wraps_async_function_with_memoization(self) -> None:
        code = "async def foo(x):\n    return x * 2\n"
        result = generate_code_improvement("memoize", current_content=code)
        assert "async def foo_cached" in result["content"]
        assert "_EVOLUTION_MEMO_CACHE" in result["content"]
        assert "await foo(*args, **kwargs)" in result["content"]
        assert code in result["content"]  # original function preserved

    def test_wraps_sync_function_without_await(self) -> None:
        code = "def bar(n):\n    return n + 1\n"
        result = generate_code_improvement("memoize", current_content=code)
        assert "result = bar(*args, **kwargs)" in result["content"]
        assert "await bar(" not in result["content"]

    def test_standalone_helper_when_no_function(self) -> None:
        result = generate_code_improvement(
            "improve", current_content="for i in range(10): pass"
        )
        assert "async def memoized_compute" in result["content"]

    def test_target_path_is_python(self) -> None:
        result = generate_code_improvement("test")
        assert result["target_path"] == "evolution/code_improvement.py"

    def test_emitted_code_passes_safety_pipeline(self) -> None:
        """The heuristic mutation is real code the safety pipeline accepts."""
        import asyncio

        from src.safety.pipeline import SafetyPipeline

        code = "async def compute(x):\n    return x\n"
        result = generate_code_improvement("memoize", current_content=code)
        safety = asyncio.run(SafetyPipeline().validate(result["content"]))
        assert safety["passed"] is True


class TestConfigTuning:
    """Tests for generate_config_tuning."""

    def test_returns_valid_json(self) -> None:
        result = generate_config_tuning("Tune temperature")
        parsed = json.loads(result["content"])
        assert "adjustments" in parsed
        assert "temperature" in parsed["adjustments"]

    def test_target_path_is_set(self) -> None:
        result = generate_config_tuning("test")
        assert result["target_path"] == "evolution/config_tuning.json"
