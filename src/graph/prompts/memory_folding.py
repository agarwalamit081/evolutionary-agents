"""Prompt templates for memory folding.

Adapted from DeepAgent's episode/working/tool memory generation.
Each prompt compresses the full conversation history into a structured
summary that replaces the original messages, dramatically reducing
token consumption on long-running tasks.
"""

from __future__ import annotations


def episode_memory_prompt(
    goal: str,
    history: str,
    available_tools: str = "",
) -> str:
    """Prompt for generating episode memory — key events and decisions.

    Args:
        goal: The original task goal.
        history: Serialized conversation history (messages → text).
        available_tools: Comma-separated list of available tool names.

    Returns:
        Prompt string for the LLM.
    """
    tools_section = f"\nAvailable tools:\n{available_tools}\n" if available_tools else ""
    return (
        "You are a memory compression assistant. Summarize the key events "
        "and decisions in the agent's reasoning process into structured episode memory.\n"
        f"\nTask:\n{goal}\n"
        f"{tools_section}"
        f"\nFull reasoning history:\n{history}\n"
        "\nInstructions:\n"
        "1. Identify major milestones, subgoal completions, and strategic decisions\n"
        "2. Extract only the most critical events that provide experience for long-term goals\n"
        "3. Output in this JSON format:\n"
        "```json\n"
        "{\n"
        '  "task_description": "Summary of what the agent has been doing.",\n'
        '  "key_events": [\n'
        "    {\n"
        '      "step": "step number",\n'
        '      "description": "What happened and why",\n'
        '      "outcome": "Result or observation"\n'
        "    }\n"
        "  ],\n"
        '  "current_progress": "What is done and what remains."\n'
        "}\n"
        "```\n"
        f"\nGenerate the episode memory for: {goal}\n"
        "Output only the JSON."
    )


def working_memory_prompt(
    goal: str,
    history: str,
    available_tools: str = "",
) -> str:
    """Prompt for generating working memory — current goals and next actions.

    Args:
        goal: The original task goal.
        history: Serialized conversation history.
        available_tools: Comma-separated list of available tool names.

    Returns:
        Prompt string for the LLM.
    """
    tools_section = f"\nAvailable tools:\n{available_tools}\n" if available_tools else ""
    return (
        "You are a working memory manager. Create a concise snapshot of the "
        "agent's CURRENT working state.\n"
        f"\nTask:\n{goal}\n"
        f"{tools_section}"
        f"\nFull reasoning history:\n{history}\n"
        "\nInstructions:\n"
        "1. Extract ONLY immediate goals, current challenges, and next steps\n"
        "2. Ignore completed/historical information\n"
        "3. Output in this JSON format:\n"
        "```json\n"
        "{\n"
        '  "immediate_goal": "Current subgoal being worked on.",\n'
        '  "current_challenges": "Main obstacles right now.",\n'
        '  "next_actions": [\n'
        "    {\n"
        '      "type": "tool_call/planning/decision",\n'
        '      "description": "Next concrete action to take."\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "```\n"
        f"\nGenerate the working memory for: {goal}\n"
        "Output only the JSON."
    )


def tool_memory_prompt(
    goal: str,
    history: str,
    tool_history: str,
    available_tools: str = "",
) -> str:
    """Prompt for generating tool memory — usage patterns and derived rules.

    Args:
        goal: The original task goal.
        history: Serialized conversation history.
        tool_history: Chronological list of tool calls and results.
        available_tools: Comma-separated list of available tool names.

    Returns:
        Prompt string for the LLM.
    """
    tools_section = f"\nAvailable tools:\n{available_tools}\n" if available_tools else ""
    return (
        "You are a tool experience recorder. Synthesize tool usage patterns "
        "into structured knowledge.\n"
        f"\nTask:\n{goal}\n"
        f"{tools_section}"
        f"\nFull reasoning history:\n{history}\n"
        f"\nTool Call History:\n{tool_history}\n"
        "\nInstructions:\n"
        "1. Analyze successful/unsuccessful tool patterns\n"
        "2. Extract metadata about each tool's effective parameters, "
        "common errors, and response patterns\n"
        "3. Output in this JSON format:\n"
        "```json\n"
        "{\n"
        '  "tools_used": [\n'
        "    {\n"
        '      "tool_name": "string",\n'
        '      "success_rate": 0.0,\n'
        '      "effective_parameters": ["param1"],\n'
        '      "common_errors": ["error_type"],\n'
        '      "experience": "What was learned."\n'
        "    }\n"
        "  ],\n"
        '  "derived_rules": [\n'
        '    "When X occurs, prefer tool Y",\n'
        '    "Tool Z works best with parameter A=B"\n'
        "  ]\n"
        "}\n"
        "```\n"
        f"\nGenerate the tool memory for: {goal}\n"
        "Output only the JSON."
    )
