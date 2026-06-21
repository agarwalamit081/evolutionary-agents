"""Predefined benchmark goals for agent performance evaluation."""

from __future__ import annotations

from src.eval.models import BenchmarkGoal

# ── Simple goals: 1-2 steps, no tool use expected ──────────────────
SIMPLE_GOALS: list[BenchmarkGoal] = [
    BenchmarkGoal(
        name="explain_concept",
        description="Explain a technical concept in a few sentences",
        goal_text="Explain what a REST API is in 2-3 sentences",
        category="simple",
        max_iterations=5,
        expected_min_steps=1,
    ),
    BenchmarkGoal(
        name="calculate_fibonacci",
        description="Compute a mathematical sequence",
        goal_text="Calculate the Fibonacci sequence up to the 10th term and list the values",
        category="simple",
        max_iterations=5,
        expected_min_steps=1,
    ),
]

# ── Complex goals: 3+ steps, tool use expected ─────────────────────
COMPLEX_GOALS: list[BenchmarkGoal] = [
    BenchmarkGoal(
        name="url_analysis",
        description="Analyze URL components with multiple steps",
        goal_text=(
            "Analyze the structure of a URL and explain each component: "
            "scheme, host, port, path, query parameters, and fragment. "
            "Provide an example for each component."
        ),
        category="complex",
        max_iterations=8,
        expected_min_steps=3,
    ),
    BenchmarkGoal(
        name="error_handling_guide",
        description="Research and synthesize error handling patterns",
        goal_text=(
            "Summarize the top 5 error handling patterns in Python "
            "with brief code examples for each"
        ),
        category="complex",
        max_iterations=8,
        expected_min_steps=2,
    ),
]

# ── Multi-agent goals: trigger sub-agent spawning ──────────────────
MULTI_AGENT_GOALS: list[BenchmarkGoal] = [
    BenchmarkGoal(
        name="comprehensive_analysis",
        description="Multi-faceted analysis that may benefit from sub-agents",
        goal_text=(
            "Perform a comprehensive analysis of error handling patterns "
            "in Python and generate a best practices guide"
        ),
        category="multi_agent",
        max_iterations=10,
    ),
]

# ── Tool creation goals: force tool gap detection ──────────────────
TOOL_CREATION_GOALS: list[BenchmarkGoal] = [
    BenchmarkGoal(
        name="fibonacci_no_tools",
        description="Compute with empty tool registry to force tool creation",
        goal_text="Calculate the Fibonacci sequence up to the 10th term",
        category="tool_creation",
        max_iterations=5,
    ),
]

ALL_GOALS: list[BenchmarkGoal] = SIMPLE_GOALS + COMPLEX_GOALS + MULTI_AGENT_GOALS + TOOL_CREATION_GOALS
