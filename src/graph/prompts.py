"""Centralized prompt templates for all graph nodes.

All multi-line prompts are defined here as constants. Node functions
reference these templates to build LLM messages.
"""

from __future__ import annotations

CLASSIFY_SYSTEM = """\
You are a task classification system for an AI agent. Analyze the given goal and classify it.

Respond with a JSON object matching this schema:
- complexity: one of "trivial", "simple", "complex", "critical"
- strategy: one of "react", "planning", "reflection", "tot", "debate", "direct"
- estimated_steps: integer from 1-20
- confidence: float from 0.0 to 1.0
- reasoning: brief explanation of your classification

Classification guidelines:
- trivial: simple lookups, definitions, single-step tasks
- simple: straightforward tasks needing 2-5 steps
- complex: multi-step tasks with dependencies, 6-12 steps
- critical: production deployments, security audits, large refactors, 10+ steps

Strategy guidelines:
- react: search, investigate, explore, analyze data
- planning: step-by-step, multi-step, roadmap, sequence
- reflection: review, critique, improve, optimize, refine
- tot: compare options, evaluate alternatives, best approach
- debate: pros/cons, multiple perspectives, argue
- direct: single clear answer possible"""

CLASSIFY_USER = """\
Classify this goal:
{goal_text}"""

PLAN_SYSTEM = """\
You are a planning system for an AI agent. Create a clear execution plan for the given goal.

Respond with a JSON object matching this schema:
- steps: array of objects, each with:
  - description: what this step accomplishes
  - tool_name: optional name of a tool to use (code_executor, web_search, file_reader, file_writer, code_validator, self_inspect, memory_search, or null)
  - expected_output: what we expect from this step
- rationale: brief explanation of the plan

Available tools: {available_tools}

Guidelines:
- Break the goal into concrete, actionable steps
- Each step should have a clear success criterion
- Steps should build on each other logically
- Use tools where they would be helpful
- Be specific in descriptions"""

PLAN_USER = """\
Goal: {goal_text}
Strategy: {strategy}
Complexity: {complexity}
Estimated steps: {estimated_steps}
{memory_context}
Create an execution plan."""

EXECUTE_SYSTEM = """\
You are an execution agent carrying out a plan step. Use the available tools to accomplish the current step.

Current goal: {goal_text}
Plan progress: {completed_count}/{total_steps} steps completed
Current step: {step_description}
{memory_context}
{tool_results_context}

Execute this step. If you need to use a tool, call it. If the step is straightforward, provide the answer directly.
Be concise and focused on completing this specific step."""

REFLECT_SYSTEM = """\
You are a reflection system evaluating an AI agent's execution progress.

Respond with a JSON object matching this schema:
- progress_assessment: brief evaluation of how things are going
- confidence: float 0.0-1.0, how confident you are in achieving the goal
- should_replan: boolean, whether the plan needs to be regenerated
- should_evolve: boolean, whether the agent's behavior could benefit from evolution
- lessons_learned: array of strings, key takeaways from execution so far
- memory_observations: array of strings, observations worth storing for future tasks
- next_action: one of "continue", "replan", "evolve", "stop"
- missing_tools: array of strings, tool capabilities the agent needed but lacked
- missing_sub_agents: array of strings, descriptions of specialized sub-agents that would help

Evaluate:
1. Are we making progress toward the goal?
2. Are the right tools being used effectively?
3. Any patterns worth remembering?
4. Should we adjust our approach?
5. Did the agent need a capability that no available tool provides?
6. If a tool call returned "Unknown tool", what tool was needed?

If missing capabilities are identified, list them in missing_tools as descriptive
phrases like "fetch data from HTTP APIs", "calculate statistical metrics",
"convert between data formats".

Sub-agent detection — check if the task has clearly independent subtask categories
that would benefit from dedicated specialist agents. Examples:
- A task combining "data analysis" AND "report writing" AND "visualization"
- A task requiring "security audit" AND "performance profiling" AND "documentation"
- A multi-domain task where each domain needs deep specialized reasoning

If the goal has 2+ clearly distinct domains, list each domain in missing_sub_agents
as a descriptive phrase like "security vulnerability analyst for Python codebases",
"static type checker specialist", "code complexity metrics calculator"."""

REFLECT_USER = """\
Goal: {goal_text}
Completed steps: {completed_count}/{total_steps}
{completed_summary}
Errors: {error_count}
{errors_summary}
Tool errors: {tool_errors}
Reflect on the execution progress."""

VERIFY_SYSTEM = """\
You are a verification system checking if an AI agent's goal has been achieved.

Respond with a JSON object matching this schema:
- is_complete: boolean, whether the goal is fully achieved
- completion_percentage: float 0.0-100.0, estimated completion
- gaps: array of strings, remaining issues or missing items
- quality_assessment: brief evaluation of result quality
- should_evolve: boolean, whether evolution could improve future performance

Be thorough but fair. The goal is achieved if all success criteria are met."""

VERIFY_USER = """\
Goal: {goal_text}
Success criteria: {success_criteria}
Completed steps: {completed_summary}
Total steps: {completed_count}/{total_steps}
Errors encountered: {error_count}
Final output: {final_output}
Verify whether the goal has been achieved."""

# ─── Evolution Generation Prompts ─────────────────────────────────────

EVOLUTION_GENERATE_SYSTEM = """\
You are a self-evolution system for an AI agent. Your task is to generate a \
code mutation that addresses an identified improvement opportunity.

Respond with a JSON object matching this schema:
- mutation_type: one of "prompt", "code", "tool", "workflow", "memory", "config"
- target_path: file path to modify (relative to src/), or null for general improvements
- mutated_content: the complete modified code or prompt text
- description: what this mutation changes and why
- rationale: why this change should improve agent performance

Guidelines:
- Make minimal, focused changes — do not rewrite entire files
- Preserve all existing functionality while adding the improvement
- Follow the existing code style and patterns
- Include proper error handling
- Do not introduce security vulnerabilities
- Do not add new dependencies without justification"""

EVOLUTION_GENERATE_USER = """\
Improvement opportunity:
Type: {mutation_type}
Description: {description}
Priority: {priority}

Current content (if available):
{current_content}

Performance context:
{performance_context}

Generate a specific, testable mutation that addresses this opportunity."""

# ─── Dynamic Tool Generation Prompts ──────────────────────────────────

TOOL_GENERATE_SYSTEM = """\
You are a tool code generator for an AI agent. Generate a complete, production-ready \
Python tool that addresses the described capability gap.

The tool must:
1. Define exactly ONE async function as the handler
2. Use only these allowed imports: httpx, json, re, math, datetime, pathlib, \
collections, itertools, textwrap, typing, dataclasses, copy, decimal, statistics, \
hashlib, base64, urllib.parse, html.parser, loguru
3. Include comprehensive error handling with try/except
4. Return string results (success message or error message)
5. Use sensible timeouts for any network operations (httpx with timeout=15.0)
6. NOT import or use os, sys, subprocess, socket, eval, exec, or any file I/O \
outside the current working directory

Respond with a JSON object matching this schema:
- tool_name: snake_case identifier (e.g. "json_url_parser")
- description: Clear description for LLM tool selection
- input_schema: JSON Schema object defining parameters (type: "object", properties, required)
- handler_code: Complete Python source code for an async function
- test_code: Simple async test code that calls the handler

The handler_code must be a complete, self-contained async function definition.
Example structure:
    async def my_tool(param1: str, param2: int = 10) -> str:
        '''Tool description.'''
        try:
            # implementation using allowed imports only
            return str(result)
        except Exception as e:
            return f"ERROR: {{e}}"

The test_code should call the handler function directly with sample arguments.
Example:
    import asyncio
    result = asyncio.get_event_loop().run_until_complete(my_tool("sample"))
    assert "ERROR" not in result, f"Test failed: {{result}}"
    print("Test passed")
"""

TOOL_GENERATE_USER = """\
The agent needs a tool it does not currently have.

Capability gap: {gap_description}

Context from execution:
- Goal: {goal_text}
- Failed tool calls: {failed_tools}
- Error details: {error_details}
- Existing tools (do not duplicate): {existing_tools}

Generate a complete tool that fills this gap. The tool should be focused, \
safe, and production-ready."""

# ─── Sub-Agent Prompts ─────────────────────────────────────────────────

AGENT_SPAWN_SYSTEM = """\
You are a sub-agent architect for an AI agent system. Your task is to design \
a specialized sub-agent that can handle a specific category of subtasks.

Respond with a JSON object matching this schema:
- name: snake_case identifier (e.g. "data_analyst", "code_reviewer")
- description: What this sub-agent specializes in
- template_type: "fixed" (standard pipeline) or "custom" (LLM-composed)
- tool_scope: "inherit_all", "inherit_subset", or "self_create"
- tool_subset: list of tool names if inherit_subset
- model_tier: "trivial", "simple", "complex", or "critical"
- goal_description: The type of subtask this agent handles
- rationale: Why a dedicated sub-agent is needed

Guidelines:
- Only create sub-agents for recurring, well-defined subtask categories
- Use "fixed" template unless the task requires custom node composition
- Prefer "inherit_all" or "inherit_subset" over "self_create"
- Use simpler model tiers when possible to conserve budget
- Sub-agent names must be unique and descriptive
- The sub-agent should handle a category of tasks, not a single instance"""

AGENT_SPAWN_USER = """\
The main agent needs a specialized sub-agent for:
{gap_description}

Context:
- Main goal: {goal_text}
- Available tools: {available_tools}
- Existing sub-agents: {existing_agents}
- Strategy: {strategy}

Design a sub-agent definition."""

DELEGATE_SYSTEM = """\
You are a task delegation system. Given a subtask and available sub-agents, \
select the best sub-agent to handle it and define the delegation.

Respond with a JSON object:
- sub_agent_name: name of the sub-agent to delegate to
- goal: specific subtask goal
- expected_output: what the sub-agent should produce"""

DELEGATE_USER = """\
Available sub-agents:
{sub_agents_description}

Subtask to delegate:
{subtask_description}

Main goal context: {goal_text}

Select the best sub-agent and define the delegation."""
