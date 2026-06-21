"""Tests for RunHistoryGenerator."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.graph.enums import Confidence, Strategy
from src.graph.models import Goal, GoalStatus, PlanStep
from src.graph.run_history import RunHistoryGenerator


def _sample_state() -> dict[str, Any]:
    """Create a populated sample state for testing."""
    return {
        "thread_id": "test-thread-001",
        "generation": 1,
        "current_goal": Goal(text="Analyze quarterly revenue data and generate a report", status=GoalStatus.ACTIVE),
        "strategy": Strategy.PLANNING,
        "confidence": Confidence.HIGH,
        "plan_steps": [
            PlanStep(description="Fetch revenue data from database"),
            PlanStep(description="Calculate quarterly totals"),
            PlanStep(description="Generate summary report"),
        ],
        "completed_steps": [
            PlanStep(description="Fetch revenue data from database"),
            PlanStep(description="Calculate quarterly totals"),
        ],
        "tool_results": [
            _mock_tool_result("code_executor", True),
            _mock_tool_result("code_executor", True),
            _mock_tool_result("file_writer", True),
        ],
        "tools_called": [
            {"name": "code_executor", "args": {}},
            {"name": "code_executor", "args": {}},
            {"name": "file_writer", "args": {}},
        ],
        "tools_created": [
            {"name": "revenue_calculator", "description": "Calculates revenue metrics"},
        ],
        "sub_agents_spawned": [
            {"name": "data_analyst", "description": "Analyzes numerical data", "tool_scope": "inherit_all"},
        ],
        "delegation_results": [
            {"sub_agent_name": "data_analyst", "success": True, "result_summary": "Analyzed revenue trends"},
        ],
        "iteration_count": 8,
        "max_iterations": 25,
        "total_tokens_used": 15420,
        "cost_records": [],
        "errors": [],
        "final_output": "Quarterly revenue increased 12% YoY",
        "is_complete": True,
        "pending_tool_gaps": [],
        "pending_agent_gaps": [],
    }


def _mock_tool_result(name: str, success: bool) -> MagicMock:
    """Create a mock tool result."""
    result = MagicMock()
    result.tool_name = name
    result.success = success
    return result


class TestRunHistoryGenerator:
    """Tests for RunHistoryGenerator."""

    @pytest.mark.asyncio
    async def test_generates_markdown_file(self, tmp_path: Path) -> None:
        """Should generate a markdown file in the workspace directory."""
        gen = RunHistoryGenerator(workspace_root=str(tmp_path / "workspace"))
        state = _sample_state()

        filepath = await gen.generate(state)

        assert filepath.exists()
        assert filepath.suffix == ".md"
        assert filepath.name.startswith("run_history_")
        content = filepath.read_text(encoding="utf-8")
        assert "# Agent Run History" in content

    @pytest.mark.asyncio
    async def test_includes_goal(self, tmp_path: Path) -> None:
        """Should include the goal text."""
        gen = RunHistoryGenerator(workspace_root=str(tmp_path))
        state = _sample_state()

        filepath = await gen.generate(state)
        content = filepath.read_text(encoding="utf-8")

        assert "Analyze quarterly revenue data" in content

    @pytest.mark.asyncio
    async def test_includes_classification(self, tmp_path: Path) -> None:
        """Should include strategy and confidence."""
        gen = RunHistoryGenerator(workspace_root=str(tmp_path))
        state = _sample_state()

        filepath = await gen.generate(state)
        content = filepath.read_text(encoding="utf-8")

        assert "Strategy" in content
        assert "Confidence" in content

    @pytest.mark.asyncio
    async def test_includes_plan_steps(self, tmp_path: Path) -> None:
        """Should include plan steps with completion status."""
        gen = RunHistoryGenerator(workspace_root=str(tmp_path))
        state = _sample_state()

        filepath = await gen.generate(state)
        content = filepath.read_text(encoding="utf-8")

        assert "Fetch revenue data" in content
        assert "Calculate quarterly totals" in content
        assert "Done" in content
        assert "Pending" in content

    @pytest.mark.asyncio
    async def test_includes_tool_usage(self, tmp_path: Path) -> None:
        """Should include tool usage breakdown."""
        gen = RunHistoryGenerator(workspace_root=str(tmp_path))
        state = _sample_state()

        filepath = await gen.generate(state)
        content = filepath.read_text(encoding="utf-8")

        assert "code_executor" in content
        assert "file_writer" in content
        assert "New Tools Created" in content
        assert "revenue_calculator" in content

    @pytest.mark.asyncio
    async def test_includes_sub_agents(self, tmp_path: Path) -> None:
        """Should include sub-agent information."""
        gen = RunHistoryGenerator(workspace_root=str(tmp_path))
        state = _sample_state()

        filepath = await gen.generate(state)
        content = filepath.read_text(encoding="utf-8")

        assert "data_analyst" in content
        assert "inherit_all" in content
        assert "Delegation Results" in content

    @pytest.mark.asyncio
    async def test_includes_metrics(self, tmp_path: Path) -> None:
        """Should include iteration count and token usage."""
        gen = RunHistoryGenerator(workspace_root=str(tmp_path))
        state = _sample_state()

        filepath = await gen.generate(state)
        content = filepath.read_text(encoding="utf-8")

        assert "Iterations" in content
        assert "15420" in content
        assert "Complete" in content

    @pytest.mark.asyncio
    async def test_includes_final_output(self, tmp_path: Path) -> None:
        """Should include the final output."""
        gen = RunHistoryGenerator(workspace_root=str(tmp_path))
        state = _sample_state()

        filepath = await gen.generate(state)
        content = filepath.read_text(encoding="utf-8")

        assert "Quarterly revenue increased 12%" in content

    @pytest.mark.asyncio
    async def test_handles_empty_state(self, tmp_path: Path) -> None:
        """Should handle minimal state without errors."""
        gen = RunHistoryGenerator(workspace_root=str(tmp_path))
        state = {"thread_id": "minimal"}

        filepath = await gen.generate(state)
        content = filepath.read_text(encoding="utf-8")

        assert "# Agent Run History" in content
        assert "None" in content  # Errors section says None

    @pytest.mark.asyncio
    async def test_creates_workspace_directory(self, tmp_path: Path) -> None:
        """Should create workspace directory if it does not exist."""
        workspace = tmp_path / "nested" / "workspace"
        gen = RunHistoryGenerator(workspace_root=str(workspace))

        filepath = await gen.generate({"thread_id": "test"})

        assert workspace.exists()
        assert filepath.exists()
