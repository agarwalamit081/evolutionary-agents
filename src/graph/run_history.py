"""Run history generator — produces a markdown summary of each agent run.

Writes a timestamped markdown file to the workspace directory containing:
goal, classification, strategy, plan steps, tools used, sub-agents spawned,
new tools/agents created, iteration count, token usage, cost, and errors.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger


class RunHistoryGenerator:
    """Generates a markdown run history from final agent state."""

    def __init__(self, workspace_root: str = ".turing/workspace") -> None:
        self._workspace_root = Path(workspace_root)

    async def generate(self, state: dict[str, Any]) -> Path:
        """Generate and write a markdown run history file.

        Args:
            state: Final agent state dict from graph invocation.

        Returns:
            Path to the written file.
        """
        self._workspace_root.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(tz=timezone.utc)
        filename = f"run_history_{timestamp.strftime('%Y%m%d_%H%M%S')}.md"
        filepath = self._workspace_root / filename

        content = self._build_markdown(state, timestamp)
        filepath.write_text(content, encoding="utf-8")

        logger.info(f"Run history written to: {filepath}")
        return filepath

    def _build_markdown(self, state: dict[str, Any], timestamp: datetime) -> str:
        """Build the markdown content from agent state."""
        lines: list[str] = []

        # Header
        lines.append("# Agent Run History")
        lines.append("")
        lines.append(f"**Timestamp**: {timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append(f"**Thread ID**: {state.get('thread_id', 'unknown')}")
        lines.append("")

        # Goal
        goal = state.get("current_goal")
        goal_text = goal.text if goal and hasattr(goal, "text") else state.get("goal_text", "unknown")
        lines.append("## Goal")
        lines.append("")
        lines.append(goal_text)
        lines.append("")

        # Classification
        lines.append("## Classification")
        lines.append("")
        strategy = state.get("strategy", "unknown")
        confidence = state.get("confidence", "unknown")
        if hasattr(confidence, "value"):
            confidence = confidence.value
        lines.append(f"- **Strategy**: {strategy}")
        lines.append(f"- **Confidence**: {confidence}")
        lines.append(f"- **Generation**: {state.get('generation', 1)}")
        lines.append("")

        # Plan Steps
        plan_steps = state.get("plan_steps", [])
        completed_steps = state.get("completed_steps", [])
        lines.append("## Plan Execution")
        lines.append("")
        lines.append(f"**Total steps**: {len(plan_steps)} | **Completed**: {len(completed_steps)}")
        lines.append("")
        if plan_steps:
            lines.append("| # | Step | Status |")
            lines.append("|---|------|--------|")
            completed_descs = {
                getattr(s, "description", str(s))
                for s in completed_steps
            }
            for i, step in enumerate(plan_steps, 1):
                desc = getattr(step, "description", str(step))
                is_done = desc in completed_descs
                status = "Done" if is_done else "Pending"
                lines.append(f"| {i} | {desc[:80]} | {status} |")
            lines.append("")

        # Tools
        tool_results = state.get("tool_results", [])
        tools_called = state.get("tools_called", [])
        tools_created = state.get("tools_created", [])
        existing_tool_count = len(tools_called) - len(tools_created)

        lines.append("## Tools")
        lines.append("")
        lines.append(f"- **Existing tools used**: {max(existing_tool_count, 0)}")
        lines.append(f"- **New tools created**: {len(tools_created)}")
        lines.append(f"- **Total tool calls**: {len(tool_results)}")
        lines.append("")

        # Tool usage breakdown
        if tool_results:
            tool_counts: dict[str, int] = {}
            for tr in tool_results:
                name = getattr(tr, "tool_name", "unknown")
                tool_counts[name] = tool_counts.get(name, 0) + 1
            lines.append("### Tool Usage Breakdown")
            lines.append("")
            lines.append("| Tool | Calls |")
            lines.append("|------|-------|")
            for name, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
                lines.append(f"| {name} | {count} |")
            lines.append("")

        # New tools created
        if tools_created:
            lines.append("### New Tools Created")
            lines.append("")
            for tool in tools_created:
                name = tool.get("name", "unknown")
                desc = tool.get("description", "")
                lines.append(f"- **{name}**: {desc[:100]}")
            lines.append("")

        # Sub-Agents
        sub_agents_spawned = state.get("sub_agents_spawned", [])
        delegation_results = state.get("delegation_results", [])

        lines.append("## Sub-Agents")
        lines.append("")
        lines.append(f"- **Spawned**: {len(sub_agents_spawned)}")
        lines.append(f"- **Delegations**: {len(delegation_results)}")
        lines.append("")

        if sub_agents_spawned:
            lines.append("### Spawned Sub-Agents")
            lines.append("")
            for agent in sub_agents_spawned:
                name = agent.get("name", "unknown")
                desc = agent.get("description", "")
                tool_scope = agent.get("tool_scope", "unknown")
                lines.append(f"- **{name}** ({tool_scope}): {desc[:100]}")
            lines.append("")

        if delegation_results:
            lines.append("### Delegation Results")
            lines.append("")
            lines.append("| Sub-Agent | Success | Summary |")
            lines.append("|-----------|---------|---------|")
            for result in delegation_results:
                name = result.get("sub_agent_name", "unknown")
                success = "Yes" if result.get("success", False) else "No"
                summary = str(result.get("result_summary", ""))[:60]
                lines.append(f"| {name} | {success} | {summary} |")
            lines.append("")

        # Metrics
        lines.append("## Metrics")
        lines.append("")
        lines.append(f"- **Iterations**: {state.get('iteration_count', 0)} / {state.get('max_iterations', 25)}")
        lines.append(f"- **Tokens used**: {state.get('total_tokens_used', 0)}")

        cost_records = state.get("cost_records", [])
        total_cost = sum(
            getattr(cr, "cost_usd", 0) if hasattr(cr, "cost_usd") else cr.get("cost_usd", 0)
            for cr in cost_records
        )
        lines.append(f"- **Total cost**: ${total_cost:.4f}")
        lines.append(f"- **Cost records**: {len(cost_records)}")
        lines.append("")

        # Models Used breakdown — aggregate cost_records by model
        if cost_records:
            lines.append("## Models Used")
            lines.append("")

            model_usage: dict[str, dict[str, int | float]] = {}
            for cr in cost_records:
                model = getattr(cr, "model", cr.get("model", "unknown") if isinstance(cr, dict) else "unknown")
                provider = getattr(cr, "provider", cr.get("provider", "") if isinstance(cr, dict) else "")
                inp = int(getattr(cr, "input_tokens", cr.get("input_tokens", 0) if isinstance(cr, dict) else 0))
                out = int(getattr(cr, "output_tokens", cr.get("output_tokens", 0) if isinstance(cr, dict) else 0))
                cost_val = float(getattr(cr, "cost_usd", cr.get("cost_usd", 0) if isinstance(cr, dict) else 0))

                key = f"{model} ({provider})" if provider else model
                if key not in model_usage:
                    model_usage[key] = {"input_tokens": 0, "output_tokens": 0, "total_cost": 0.0}
                model_usage[key]["input_tokens"] += inp
                model_usage[key]["output_tokens"] += out
                model_usage[key]["total_cost"] += cost_val

            lines.append("| Model (Provider) | Input Tokens | Output Tokens | Cost |")
            lines.append("|-----------------|-------------|--------------|------|")
            for model_key, usage in sorted(model_usage.items(), key=lambda x: -x[1]["total_cost"]):
                lines.append(
                    f"| {model_key} | {usage['input_tokens']:,} | "
                    f"{usage['output_tokens']:,} | ${usage['total_cost']:.4f} |"
                )
            lines.append("")

        # Errors
        errors = state.get("errors", [])
        lines.append("## Errors")
        lines.append("")
        if errors:
            for error in errors[-10:]:  # Last 10 errors
                lines.append(f"- {str(error)[:200]}")
        else:
            lines.append("None")
        lines.append("")

        # Output
        final_output = state.get("final_output", "")
        is_complete = state.get("is_complete", False)
        lines.append("## Result")
        lines.append("")
        lines.append(f"**Complete**: {'Yes' if is_complete else 'No'}")
        lines.append("")
        if final_output:
            lines.append("### Final Output")
            lines.append("")
            lines.append(final_output[:2000])
            lines.append("")

        return "\n".join(lines)
