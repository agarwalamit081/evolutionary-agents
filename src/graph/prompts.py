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

Available tools: code_executor, web_search, file_reader, file_writer, code_validator, self_inspect, memory_search

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

Evaluate:
1. Are we making progress toward the goal?
2. Are the right tools being used effectively?
3. Any patterns worth remembering?
4. Should we adjust our approach?"""

REFLECT_USER = """\
Goal: {goal_text}
Completed steps: {completed_count}/{total_steps}
{completed_summary}
Errors: {error_count}
{errors_summary}
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
