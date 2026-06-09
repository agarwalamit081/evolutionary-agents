"""Tests for SubAgentRunner from src.agents.runner."""

from __future__ import annotations


import pytest
from unittest.mock import AsyncMock, MagicMock

from src.agents.runner import SubAgentRunner
from src.graph.models import SubAgentSpec


@pytest.fixture
def sample_spec() -> SubAgentSpec:
    """Create a sample SubAgentSpec for testing."""
    return SubAgentSpec(
        name="test_runner_agent",
        description="Agent for runner tests",
        goal="test runner goal",
        parent_thread_id="thread-001",
        tool_scope="inherit_all",
    )


@pytest.fixture
def mock_gateway() -> MagicMock:
    """Create a mock LLMGateway."""
    gateway = MagicMock()
    gateway.acompletion = AsyncMock()
    return gateway


@pytest.fixture
def mock_tools() -> MagicMock:
    """Create a mock ToolRegistry."""
    tools = MagicMock()
    tools.list_names = MagicMock(return_value=["tool1", "tool2"])
    tools.get = MagicMock(return_value=None)
    return tools


@pytest.fixture
def runner(sample_spec: SubAgentSpec, mock_gateway: MagicMock, mock_tools: MagicMock) -> SubAgentRunner:
    """Create a SubAgentRunner instance for testing."""
    return SubAgentRunner(
        definition=sample_spec,
        gateway=mock_gateway,
        tools=mock_tools,
    )


class TestSubAgentRunner:
    """Tests for SubAgentRunner class."""

    def test_runner_initialization(self, sample_spec: SubAgentSpec, mock_gateway: MagicMock, mock_tools: MagicMock) -> None:
        """SubAgentRunner initializes with definition, gateway, and tools."""
        runner = SubAgentRunner(
            definition=sample_spec,
            gateway=mock_gateway,
            tools=mock_tools,
        )

        assert runner.definition.name == "test_runner_agent"
        assert runner.definition.description == "Agent for runner tests"

    def test_runner_definition_property(self, runner: SubAgentRunner, sample_spec: SubAgentSpec) -> None:
        """definition property returns the SubAgentSpec."""
        assert runner.definition is sample_spec
        assert runner.definition.name == "test_runner_agent"


class TestRunnerDepthLimit:
    """Tests for depth limit enforcement in run()."""

    @pytest.mark.asyncio
    async def test_runner_depth_limit_enforced(self, runner: SubAgentRunner) -> None:
        """run() returns error when depth >= depth_limit."""
        # Modify spec to have depth_limit of 2
        runner._definition.depth_limit = 2

        result = await runner.run(
            goal="test goal",
            parent_thread_id="thread-001",
            depth=2,  # Equal to depth_limit
        )

        assert result["success"] is False
        assert "Depth limit" in result["errors"][0]
        assert result["sub_agent_name"] == "test_runner_agent"
        assert result["tokens_used"] == 0
        assert result["cost_usd"] == 0.0

    @pytest.mark.asyncio
    async def test_runner_depth_limit_zero(self, runner: SubAgentRunner) -> None:
        """run() allows execution when depth_limit=0 (no limit)."""
        runner._definition.depth_limit = 0

        # Mock the graph compilation and execution
        from unittest.mock import patch

        mock_result_state = {
            "final_output": "test output",
            "is_complete": True,
            "errors": [],
            "cost_records": [],
            "total_tokens_used": 100,
            "iteration_count": 1,
        }

        with patch("src.agents.runner.build_subgraph") as mock_build:
            mock_graph = MagicMock()
            mock_compiled = MagicMock()
            mock_compiled.ainvoke = AsyncMock(return_value=mock_result_state)
            mock_graph.compile = MagicMock(return_value=mock_compiled)
            mock_build.return_value = mock_graph

            result = await runner.run(
                goal="test goal",
                parent_thread_id="thread-001",
                depth=5,  # Depth would normally be > limit
            )

            # Should not return depth limit error
            assert "Depth limit" not in result.get("errors", [])

    @pytest.mark.asyncio
    async def test_runner_depth_below_limit(self, runner: SubAgentRunner) -> None:
        """run() allows execution when depth < depth_limit."""
        runner._definition.depth_limit = 5

        from unittest.mock import patch

        mock_result_state = {
            "final_output": "test output",
            "is_complete": True,
            "errors": [],
            "cost_records": [],
            "total_tokens_used": 50,
            "iteration_count": 1,
        }

        with patch("src.agents.runner.build_subgraph") as mock_build:
            mock_graph = MagicMock()
            mock_compiled = MagicMock()
            mock_compiled.ainvoke = AsyncMock(return_value=mock_result_state)
            mock_graph.compile = MagicMock(return_value=mock_compiled)
            mock_build.return_value = mock_graph

            result = await runner.run(
                goal="test goal",
                parent_thread_id="thread-001",
                depth=3,  # Below limit
            )

            assert "Depth limit" not in result.get("errors", [])


class TestRunWithMemory:
    """Tests for run() with memory parameter."""

    @pytest.mark.asyncio
    async def test_runner_with_memory(self, runner: SubAgentRunner) -> None:
        """run() accepts memory parameter."""
        from unittest.mock import patch, MagicMock

        memory = MagicMock()
        runner._memory = memory

        mock_result_state = {
            "final_output": "test output",
            "is_complete": True,
            "errors": [],
            "cost_records": [],
            "total_tokens_used": 100,
            "iteration_count": 1,
        }

        with patch("src.agents.runner.build_subgraph") as mock_build:
            mock_graph = MagicMock()
            mock_compiled = MagicMock()
            mock_compiled.ainvoke = AsyncMock(return_value=mock_result_state)
            mock_graph.compile = MagicMock(return_value=mock_compiled)
            mock_build.return_value = mock_graph

            result = await runner.run(
                goal="test goal",
                parent_thread_id="thread-001",
            )

            assert result["success"] is True
            assert result["result"] == "test output"


class TestRunErrorHandling:
    """Tests for error handling in run()."""

    @pytest.mark.asyncio
    async def test_run_handles_build_failure(self, runner: SubAgentRunner) -> None:
        """run() returns error result when graph build fails."""
        from unittest.mock import patch

        with patch("src.agents.runner.build_subgraph", side_effect=Exception("Build failed")):
            result = await runner.run(
                goal="test goal",
                parent_thread_id="thread-001",
            )

            assert result["success"] is False
            assert "Sub-agent execution error" in result["errors"][0]
            assert result["tokens_used"] == 0
            assert result["cost_usd"] == 0.0

    @pytest.mark.asyncio
    async def test_run_handles_execution_failure(self, runner: SubAgentRunner) -> None:
        """run() returns error result when graph execution fails."""
        from unittest.mock import patch

        with patch("src.agents.runner.build_subgraph") as mock_build:
            mock_graph = MagicMock()
            mock_compiled = MagicMock()
            mock_compiled.ainvoke = AsyncMock(side_effect=Exception("Execution failed"))
            mock_graph.compile = MagicMock(return_value=mock_compiled)
            mock_build.return_value = mock_graph

            result = await runner.run(
                goal="test goal",
                parent_thread_id="thread-001",
            )

            assert result["success"] is False
            assert "Sub-agent execution error" in result["errors"][0]


class TestResultExtraction:
    """Tests for _extract_results() helper function."""

    def test_extract_results_success(self) -> None:
        """_extract_results extracts successful result."""
        from src.agents.runner import _extract_results
        from src.graph.models import ReflectionResult

        result_state = {
            "final_output": "Completed successfully",
            "is_complete": True,
            "errors": [],
            "cost_records": [],
            "total_tokens_used": 150,
            "iteration_count": 3,
            "reflection": ReflectionResult(
                summary="Good",
                cost_efficiency=0.9,
            ),
        }

        result = _extract_results(
            result_state=result_state,
            latency_ms=100,
            goal="test goal",
            spec=SubAgentSpec(
                name="test_agent",
                description="Test",
                goal="test",
                parent_thread_id="thread-001",
            ),
        )

        assert result["success"] is True
        assert result["result"] == "Completed successfully"
        assert result["tokens_used"] == 150
        assert result["latency_ms"] == 100
        assert result["iterations"] == 3
        assert result["goal"] == "test goal"
        assert result["sub_agent_name"] == "test_agent"

    def test_extract_results_with_errors(self) -> None:
        """_extract_results handles errors."""
        from src.agents.runner import _extract_results

        result_state = {
            "final_output": "Partial result",
            "is_complete": True,
            "errors": ["Some error occurred"],
            "cost_records": [],
            "total_tokens_used": 100,
            "iteration_count": 2,
        }

        result = _extract_results(
            result_state=result_state,
            latency_ms=50,
            goal="test goal",
            spec=SubAgentSpec(
                name="test_agent",
                description="Test",
                goal="test",
                parent_thread_id="thread-001",
            ),
        )

        assert result["success"] is False  # Errors present
        assert result["errors"] == ["Some error occurred"]

    def test_extract_results_calculates_cost(self) -> None:
        """_extract_results sums cost from cost_records."""
        from src.agents.runner import _extract_results
        from src.graph.models import CostRecord

        result_state = {
            "final_output": "Done",
            "is_complete": True,
            "errors": [],
            "cost_records": [
                CostRecord(
                    provider="openai",
                    model="gpt-4o-mini",
                    input_tokens=100,
                    output_tokens=50,
                    cost_usd=0.0001,
                    latency_ms=100,
                ),
                CostRecord(
                    provider="openai",
                    model="gpt-4o-mini",
                    input_tokens=200,
                    output_tokens=100,
                    cost_usd=0.0002,
                    latency_ms=200,
                ),
            ],
            "total_tokens_used": 450,
            "iteration_count": 1,
        }

        result = _extract_results(
            result_state=result_state,
            latency_ms=300,
            goal="test goal",
            spec=SubAgentSpec(
                name="test_agent",
                description="Test",
                goal="test",
                parent_thread_id="thread-001",
            ),
        )

        assert result["cost_usd"] == pytest.approx(0.0003)  # Sum of both records

    def test_extract_results_missing_reflection(self) -> None:
        """_extract_results handles missing reflection."""
        from src.agents.runner import _extract_results

        result_state = {
            "final_output": "Done",
            "is_complete": True,
            "errors": [],
            "cost_records": [],
            "total_tokens_used": 100,
            "iteration_count": 1,
            "reflection": None,
        }

        result = _extract_results(
            result_state=result_state,
            latency_ms=100,
            goal="test goal",
            spec=SubAgentSpec(
                name="test_agent",
                description="Test",
                goal="test",
                parent_thread_id="thread-001",
            ),
        )

        assert result["success"] is True
        assert result.get("quality_rating") is None


class TestRunParallel:
    """Tests for run_parallel() function."""

    @pytest.mark.asyncio
    async def test_run_parallel_empty_list(self) -> None:
        """run_parallel returns empty list for empty input."""
        from src.agents.runner import run_parallel

        result = await run_parallel([])
        assert result == []

    @pytest.mark.asyncio
    async def test_run_parallel_executes_all(self, sample_spec: SubAgentSpec, mock_gateway: MagicMock, mock_tools: MagicMock) -> None:
        """run_parallel executes all runners concurrently."""
        from src.agents.runner import run_parallel, SubAgentRunner
        from unittest.mock import patch

        # Create multiple runners
        runners = []
        for i in range(3):
            spec = SubAgentSpec(
                name=f"agent_{i}",
                description=f"Agent {i}",
                goal=f"goal {i}",
                parent_thread_id=f"thread-{i}",
            )
            runner = SubAgentRunner(
                definition=spec,
                gateway=mock_gateway,
                tools=mock_tools,
            )
            runners.append((runner, f"goal {i}", f"thread-{i}", None, 0))

        # Mock the run method
        async def mock_run(goal, parent_thread_id, budget_remaining, depth):
            return {
                "success": True,
                "result": f"Completed {goal}",
                "tokens_used": 100,
                "cost_usd": 0.01,
                "latency_ms": 100,
                "iterations": 1,
                "errors": [],
                "goal": goal,
                "sub_agent_name": "test",
                "sub_agent_id": "test-id",
            }

        with patch.object(SubAgentRunner, "run", side_effect=mock_run):
            results = await run_parallel(runners)

            assert len(results) == 3
            for result in results:
                assert result["success"] is True

    @pytest.mark.asyncio
    async def test_run_parallel_handles_exceptions(self, sample_spec: SubAgentSpec, mock_gateway: MagicMock, mock_tools: MagicMock) -> None:
        """run_parallel converts exceptions to error results."""
        from src.agents.runner import run_parallel, SubAgentRunner
        from unittest.mock import patch

        runners = []

        # One successful runner
        spec1 = SubAgentSpec(
            name="success_agent",
            description="Success",
            goal="success",
            parent_thread_id="thread-001",
        )
        runner1 = SubAgentRunner(
            definition=spec1,
            gateway=mock_gateway,
            tools=mock_tools,
        )
        runners.append((runner1, "goal1", "thread-001", None, 0))

        # One failing runner
        spec2 = SubAgentSpec(
            name="fail_agent",
            description="Fail",
            goal="fail",
            parent_thread_id="thread-002",
        )
        runner2 = SubAgentRunner(
            definition=spec2,
            gateway=mock_gateway,
            tools=mock_tools,
        )
        runners.append((runner2, "fail", "thread-002", None, 0))

        # Mock run to succeed for first, fail for second
        async def mock_run(goal, parent_thread_id, budget_remaining, depth):
            if "fail" in goal:
                raise Exception("Simulated failure")
            return {
                "success": True,
                "result": "Success",
                "tokens_used": 100,
                "cost_usd": 0.01,
                "latency_ms": 100,
                "iterations": 1,
                "errors": [],
                "goal": goal,
                "sub_agent_name": "success_agent",
                "sub_agent_id": "id1",
            }

        with patch.object(SubAgentRunner, "run", side_effect=mock_run):
            results = await run_parallel(runners)

            assert len(results) == 2
            assert results[0]["success"] is True
            assert results[1]["success"] is False
            assert "Parallel execution error" in results[1]["errors"][0]


class TestBudgetMode:
    """Tests for budget mode handling."""

    @pytest.mark.asyncio
    async def test_runner_shared_budget_mode(self, runner: SubAgentRunner) -> None:
        """run() uses shared budget when budget_mode is 'shared'."""
        from unittest.mock import patch

        runner._definition.budget_mode = "shared"
        runner._definition.budget_limit = 0.0  # Ignored in shared mode

        mock_result_state = {
            "final_output": "Done",
            "is_complete": True,
            "errors": [],
            "cost_records": [],
            "total_tokens_used": 100,
            "iteration_count": 1,
        }

        with patch("src.agents.runner.build_subgraph") as mock_build:
            mock_graph = MagicMock()
            mock_compiled = MagicMock()
            mock_compiled.ainvoke = AsyncMock(return_value=mock_result_state)
            mock_graph.compile = MagicMock(return_value=mock_compiled)
            mock_build.return_value = mock_graph

            # Pass shared budget
            result = await runner.run(
                goal="test goal",
                parent_thread_id="thread-001",
                budget_remaining=5.0,
            )

            # Should pass the shared budget to the graph
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_runner_separate_budget_mode(self, runner: SubAgentRunner) -> None:
        """run() uses agent's budget_limit when budget_mode is 'separate'."""
        from unittest.mock import patch

        runner._definition.budget_mode = "separate"
        runner._definition.budget_limit = 2.5

        mock_result_state = {
            "final_output": "Done",
            "is_complete": True,
            "errors": [],
            "cost_records": [],
            "total_tokens_used": 100,
            "iteration_count": 1,
        }

        with patch("src.agents.runner.build_subgraph") as mock_build:
            mock_graph = MagicMock()
            mock_compiled = MagicMock()
            mock_compiled.ainvoke = AsyncMock(return_value=mock_result_state)
            mock_graph.compile = MagicMock(return_value=mock_compiled)
            mock_build.return_value = mock_graph

            # Shared budget should be ignored
            result = await runner.run(
                goal="test goal",
                parent_thread_id="thread-001",
                budget_remaining=5.0,  # Should be ignored
            )

            assert result["success"] is True
