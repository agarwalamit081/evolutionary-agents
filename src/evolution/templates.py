"""Mutation templates — generate real, structured content for each mutation type.

Each template function produces a dict with:
- content: JSON-serialisable payload with the actual improvement
- target_path: file path in the shadow repo for git tracking
- rationale: human-readable explanation of the change

These replace the comment-only heuristic mutations that produced no
observable effect at runtime.
"""

from __future__ import annotations

import json
from typing import Any

# ─── Failure-pattern → prompt-fix mapping ────────────────────────────────

_PROMPT_FIXES: dict[str, str] = {
    "json": (
        "IMPORTANT: Always respond with valid JSON matching the requested schema. "
        "Do NOT wrap JSON in markdown code fences. If the schema requires specific "
        "field types, respect them exactly."
    ),
    "timeout": (
        "When performing multi-step reasoning, break complex tasks into smaller "
        "sub-tasks. If a step is taking too long, summarize intermediate results "
        "and proceed rather than retrying indefinitely."
    ),
    "format": (
        "Follow the requested output format precisely. Use the exact field names "
        "and types specified in the schema. Do not add extra commentary outside "
        "the structured response."
    ),
    "tool": (
        "Before invoking a tool, verify its parameters match the expected schema. "
        "After receiving tool results, extract the key information rather than "
        "repeating the raw output verbatim."
    ),
    "error": (
        "When an error occurs during execution, analyze the error message to "
        "determine if it is transient (retry) or permanent (try an alternative "
        "approach). Include the error context in your reasoning."
    ),
    "context": (
        "When the conversation context is large, focus on the most recent "
        "instructions and results. Summarize earlier steps concisely rather "
        "than repeating them in full."
    ),
    "plan": (
        "When generating execution plans, ensure each step has a clear success "
        "criterion. Mark steps as completed only when their output matches the "
        "expected result."
    ),
}

_DEFAULT_PROMPT_FIX = (
    "Apply careful reasoning to each step. Verify intermediate results before "
    "proceeding. If a step fails, analyze the failure and adjust the approach."
)

# ─── Workflow parameter adjustments ──────────────────────────────────────

_WORKFLOW_ADJUSTMENTS: dict[str, dict[str, Any]] = {
    "reduce_execution_time": {
        "early_stop_on_confidence": True,
        "confidence_threshold": 0.85,
        "max_iterations": 8,
        "parallel_tool_calls": True,
    },
    "improve_accuracy": {
        "verification_enabled": True,
        "reflection_after_steps": 3,
        "max_iterations": 15,
        "require_lesson_extraction": True,
    },
    "balance_speed_accuracy": {
        "early_stop_on_confidence": True,
        "confidence_threshold": 0.75,
        "reflection_after_steps": 4,
        "max_iterations": 10,
    },
}

# ─── Tool parameter adjustments ──────────────────────────────────────────

_TOOL_ADJUSTMENTS: dict[str, dict[str, Any]] = {
    "code_executor": {
        "timeout_seconds": 60,
        "max_output_lines": 200,
        "capture_stderr": True,
    },
    "code_validator": {
        "check_security": True,
        "check_style": True,
        "max_complexity": 15,
    },
    "web_search": {
        "max_results": 5,
        "timeout_seconds": 30,
        "snippet_length": 500,
    },
    "memory_search": {
        "min_fitness": 0.4,
        "max_results": 5,
        "include_cold": True,
    },
}

# ─── Memory retrieval strategy adjustments ───────────────────────────────

_MEMORY_STRATEGIES: dict[str, dict[str, Any]] = {
    "precision_focused": {
        "min_fitness": 0.6,
        "max_results": 3,
        "include_cold": False,
        "tag_overlap_threshold": 2,
    },
    "recall_focused": {
        "min_fitness": 0.3,
        "max_results": 7,
        "include_cold": True,
        "tag_overlap_threshold": 1,
    },
    "balanced": {
        "min_fitness": 0.4,
        "max_results": 5,
        "include_cold": True,
        "tag_overlap_threshold": 1,
    },
}


def generate_prompt_improvement(
    patterns: list[str],
    current_content: str | None = None,
) -> dict[str, Any]:
    """Generate a concrete prompt improvement addressing failure patterns.

    Maps common failure keywords to specific prompt suffixes that guide
    the LLM toward better behaviour. Returns structured JSON that the
    retrieve_memory_node can load at runtime.

    Args:
        patterns: Failure pattern descriptions from analysis.
        current_content: Current prompt text (unused by heuristic but
            kept for interface consistency with LLM generation).

    Returns:
        Dict with content, target_path, and rationale.
    """
    suffixes: list[str] = []
    matched_keywords: list[str] = []

    for pattern in patterns:
        pattern_lower = pattern.lower()
        for keyword, fix in _PROMPT_FIXES.items():
            if keyword in pattern_lower and fix not in suffixes:
                suffixes.append(fix)
                matched_keywords.append(keyword)

    if not suffixes:
        suffixes.append(_DEFAULT_PROMPT_FIX)

    content = {
        "target_node": "execute",
        "suffixes": suffixes,
        "reason": f"Addresses failure patterns: {', '.join(matched_keywords) or 'general improvement'}",
        "generation_source": "heuristic",
    }

    rationale = (
        f"Added {len(suffixes)} prompt improvement(s) addressing: "
        f"{', '.join(matched_keywords) if matched_keywords else 'general performance'}. "
        f"These will be injected as additional context during task execution."
    )

    return {
        "content": json.dumps(content, indent=2),
        "target_path": "evolution/prompt_improvements.json",
        "rationale": rationale,
    }


def generate_workflow_config(
    description: str,
) -> dict[str, Any]:
    """Generate workflow parameter adjustments based on the opportunity description.

    Args:
        description: The opportunity description from analysis.

    Returns:
        Dict with content, target_path, and rationale.
    """
    desc_lower = description.lower()

    if "time" in desc_lower or "speed" in desc_lower or "fast" in desc_lower:
        strategy = "reduce_execution_time"
    elif "accura" in desc_lower or "quality" in desc_lower or "correct" in desc_lower:
        strategy = "improve_accuracy"
    else:
        strategy = "balance_speed_accuracy"

    config = _WORKFLOW_ADJUSTMENTS[strategy]

    rationale = (
        f"Applied '{strategy}' workflow configuration: "
        f"early_stop={config.get('early_stop_on_confidence', False)}, "
        f"max_iterations={config.get('max_iterations', 10)}. "
        f"Based on opportunity: {description[:80]}"
    )

    return {
        "content": json.dumps({"strategy": strategy, **config}, indent=2),
        "target_path": "evolution/workflow_config.json",
        "rationale": rationale,
    }


def generate_tool_config(
    description: str,
) -> dict[str, Any]:
    """Generate tool parameter adjustments.

    Args:
        description: The opportunity description from analysis.

    Returns:
        Dict with content, target_path, and rationale.
    """
    desc_lower = description.lower()

    # Pick the most relevant tool based on description keywords
    if "code" in desc_lower and "exec" in desc_lower:
        tool_name = "code_executor"
    elif "valid" in desc_lower:
        tool_name = "code_validator"
    elif "search" in desc_lower or "web" in desc_lower:
        tool_name = "web_search"
    else:
        tool_name = "memory_search"

    config = _TOOL_ADJUSTMENTS[tool_name]

    rationale = (
        f"Adjusted {tool_name} parameters: "
        + ", ".join(f"{k}={v}" for k, v in config.items())
        + f". Based on: {description[:80]}"
    )

    return {
        "content": json.dumps({"tool": tool_name, **config}, indent=2),
        "target_path": "evolution/tool_config.json",
        "rationale": rationale,
    }


def generate_memory_config(
    description: str,
) -> dict[str, Any]:
    """Generate memory retrieval strategy adjustments.

    Args:
        description: The opportunity description from analysis.

    Returns:
        Dict with content, target_path, and rationale.
    """
    desc_lower = description.lower()

    if "precision" in desc_lower or "relevant" in desc_lower or "noise" in desc_lower:
        strategy = "precision_focused"
    elif "recall" in desc_lower or "miss" in desc_lower or "more context" in desc_lower:
        strategy = "recall_focused"
    else:
        strategy = "balanced"

    config = _MEMORY_STRATEGIES[strategy]

    rationale = (
        f"Applied '{strategy}' memory retrieval strategy: "
        f"min_fitness={config['min_fitness']}, max_results={config['max_results']}. "
        f"Based on: {description[:80]}"
    )

    return {
        "content": json.dumps({"strategy": strategy, **config}, indent=2),
        "target_path": "evolution/memory_config.json",
        "rationale": rationale,
    }


def generate_code_improvement(
    description: str,
    current_content: str | None = None,
) -> dict[str, Any]:
    """Generate a code improvement suggestion.

    For heuristic mode this produces a structured analysis rather than
    actual code changes (LLM generation is preferred for CODE mutations).

    Args:
        description: The opportunity description.
        current_content: Current code to improve.

    Returns:
        Dict with content, target_path, and rationale.
    """
    _ = current_content  # noqa: ARG001 — kept for interface consistency with LLM generation
    content = {
        "analysis": description,
        "suggestion": (
            "LLM generation recommended for code mutations. "
            "Heuristic mode provides analysis only."
        ),
        "current_lines": len(current_content.splitlines()) if current_content else 0,
        "generation_source": "heuristic",
    }

    rationale = (
        f"Code improvement analysis for: {description[:80]}. "
        "Full code generation requires LLM — heuristic mode provides structural analysis."
    )

    return {
        "content": json.dumps(content, indent=2),
        "target_path": "evolution/code_analysis.json",
        "rationale": rationale,
    }


def generate_config_tuning(
    description: str,
) -> dict[str, Any]:
    """Generate configuration parameter tuning.

    Args:
        description: The opportunity description.

    Returns:
        Dict with content, target_path, and rationale.
    """
    content = {
        "tuning_target": description[:100],
        "adjustments": {
            "temperature": 0.4,
            "max_tokens_factor": 0.9,
            "cache_enabled": True,
        },
        "generation_source": "heuristic",
    }

    rationale = f"Configuration tuning based on: {description[:80]}"

    return {
        "content": json.dumps(content, indent=2),
        "target_path": "evolution/config_tuning.json",
        "rationale": rationale,
    }
