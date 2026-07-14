"""Tests for src.safety.pipeline.SafetyPipeline."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.safety.pipeline import SafetyPipeline
from src.sandbox.executor import SandboxResult


# ─── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def pipeline() -> SafetyPipeline:
    """Return a fresh SafetyPipeline instance."""
    return SafetyPipeline()


# ─── SafetyPipeline Tests ────────────────────────────────────────────


class TestSafetyPipeline:
    """Unit tests for SafetyPipeline.validate."""

    @pytest.mark.asyncio
    async def test_validate_clean_code_passes(self, pipeline: SafetyPipeline) -> None:
        """Simple valid Python passes all safety checks."""
        code = (
            "def add(a: int, b: int) -> int:\n"
            "    return a + b\n"
        )
        result = await pipeline.validate(code)

        assert result["passed"] is True
        assert result["issues"] == []

    @pytest.mark.asyncio
    async def test_validate_syntax_error_fails(self, pipeline: SafetyPipeline) -> None:
        """Code with syntax errors fails validation."""
        code = "def broken(\n"
        result = await pipeline.validate(code)

        assert result["passed"] is False
        assert any("Syntax error" in issue for issue in result["issues"])

    @pytest.mark.asyncio
    async def test_validate_dangerous_import_fails(
        self, pipeline: SafetyPipeline
    ) -> None:
        """Code importing os/system fails the security and import checks."""
        code = (
            "import os\n"
            "import sys\n"
            "os.system('rm -rf /')\n"
        )
        result = await pipeline.validate(code)

        assert result["passed"] is False
        # Should catch via both forbidden patterns (security) and import validation
        assert len(result["issues"]) > 0

    @pytest.mark.asyncio
    async def test_validate_infinite_loop_fails(self, pipeline: SafetyPipeline) -> None:
        """Code with while True and no break fails behavioral check."""
        code = (
            "def run_forever() -> None:\n"
            "    x = 1\n"
            "    while True:\n"
            "        x += 1\n"
        )
        result = await pipeline.validate(code)

        assert result["passed"] is False
        assert any("infinite loop" in issue.lower() for issue in result["issues"])

    @pytest.mark.asyncio
    async def test_validate_over_complex_function_warns(self, pipeline: SafetyPipeline) -> None:
        """Code with high cyclomatic complexity fails static analysis."""
        # Build a function with many branches
        branches = "\n".join(f"    if x == {i}: pass" for i in range(25))
        code = f"def complex_fn(x: int) -> None:\n{branches}\n"
        result = await pipeline.validate(code)

        assert result["passed"] is False

    @pytest.mark.asyncio
    async def test_validate_file_write_outside_sandbox(self, pipeline: SafetyPipeline) -> None:
        """Code with open(..., 'w') outside sandbox fails behavioral check."""
        code = (
            "with open('/etc/malicious.txt', 'w') as f:\n"
            "    f.write('bad')\n"
        )
        result = await pipeline.validate(code)

        assert result["passed"] is False


# ─── Layer 6 — Sandbox Execution Tests ────────────────────────────────


class TestLayer6Sandbox:
    """Tests for Layer 6 (sandbox execution) of the safety pipeline."""

    @pytest.mark.asyncio
    async def test_no_sandbox_executor_returns_passed(
        self, pipeline: SafetyPipeline
    ) -> None:
        """When no sandbox_executor is provided, sandbox layer passes with a note."""
        code = "def add(a: int, b: int) -> int:\n    return a + b\n"
        result = await pipeline.validate(code, sandbox_executor=None)

        assert result["layers"]["sandbox"]["passed"] is True
        assert "note" in result["layers"]["sandbox"]
        assert "sandbox skipped" in result["layers"]["sandbox"]["note"]
        assert result["passed"] is True

    @pytest.mark.asyncio
    async def test_mock_sandbox_success_passes(
        self, pipeline: SafetyPipeline
    ) -> None:
        """Mock sandbox that returns success makes sandbox layer pass."""
        mock_sandbox = MagicMock()
        mock_sandbox.execute_code = AsyncMock(return_value=SandboxResult(
            success=True, exit_code=0, stdout="ok", stderr="",
            duration_seconds=0.1, memory_mb=None, timed_out=False,
        ))

        code = "def add(a: int, b: int) -> int:\n    return a + b\n"
        result = await pipeline.validate(code, sandbox_executor=mock_sandbox)

        assert result["layers"]["sandbox"]["passed"] is True
        assert result["layers"]["sandbox"]["issues"] == []
        mock_sandbox.execute_code.assert_awaited_once_with(code)

    @pytest.mark.asyncio
    async def test_mock_sandbox_failure_fails(
        self, pipeline: SafetyPipeline
    ) -> None:
        """Mock sandbox that returns failure makes sandbox layer fail."""
        mock_sandbox = MagicMock()
        mock_sandbox.execute_code = AsyncMock(return_value=SandboxResult(
            success=False, exit_code=1, stdout="", stderr="ImportError: no module",
            duration_seconds=0.2, memory_mb=None, timed_out=False,
        ))

        code = "def add(a: int, b: int) -> int:\n    return a + b\n"
        result = await pipeline.validate(code, sandbox_executor=mock_sandbox)

        assert result["layers"]["sandbox"]["passed"] is False
        assert any("Sandbox execution failed" in issue for issue in result["layers"]["sandbox"]["issues"])

    @pytest.mark.asyncio
    async def test_mock_sandbox_timeout_fails(
        self, pipeline: SafetyPipeline
    ) -> None:
        """Mock sandbox that returns timed_out=True makes sandbox layer fail."""
        mock_sandbox = MagicMock()
        mock_sandbox.execute_code = AsyncMock(return_value=SandboxResult(
            success=False, exit_code=None, stdout="", stderr="Timeout",
            duration_seconds=30.0, memory_mb=None, timed_out=True,
        ))

        code = "def add(a: int, b: int) -> int:\n    return a + b\n"
        result = await pipeline.validate(code, sandbox_executor=mock_sandbox)

        assert result["layers"]["sandbox"]["passed"] is False
        assert any("timed out" in issue.lower() for issue in result["layers"]["sandbox"]["issues"])

    @pytest.mark.asyncio
    async def test_mock_sandbox_exception_fails(
        self, pipeline: SafetyPipeline
    ) -> None:
        """Mock sandbox that raises an exception makes sandbox layer fail."""
        mock_sandbox = MagicMock()
        mock_sandbox.execute_code = AsyncMock(side_effect=RuntimeError("connection refused"))

        code = "def add(a: int, b: int) -> int:\n    return a + b\n"
        result = await pipeline.validate(code, sandbox_executor=mock_sandbox)

        assert result["layers"]["sandbox"]["passed"] is False
        assert any("Sandbox execution error" in issue for issue in result["layers"]["sandbox"]["issues"])


# ─── Layer 7 — Semantic Check Tests ───────────────────────────────────


class TestLayer7Semantic:
    """Tests for Layer 7 (semantic check) of the safety pipeline."""

    @pytest.mark.asyncio
    async def test_clean_code_with_function_passes(
        self, pipeline: SafetyPipeline
    ) -> None:
        """Well-formed code with a function definition passes semantic check."""
        code = (
            "def add(a: int, b: int) -> int:\n"
            "    result = a + b\n"
            "    return result\n"
        )
        result = await pipeline.validate(code)

        assert result["layers"]["semantic"]["passed"] is True
        assert result["layers"]["semantic"]["issues"] == []

    @pytest.mark.asyncio
    async def test_sys_exit_call_fails(
        self, pipeline: SafetyPipeline
    ) -> None:
        """Code with sys.exit() fails semantic check."""
        code = (
            "import sys\n"
            "def shutdown() -> None:\n"
            "    sys.exit(1)\n"
        )
        result = await pipeline.validate(code)

        # Note: import sys is blocked by layer 4, so this test focuses on
        # semantic layer specifically catching sys.exit()
        semantic_layer = result["layers"]["semantic"]
        assert semantic_layer["passed"] is False
        assert any("sys.exit()" in issue for issue in semantic_layer["issues"])

    @pytest.mark.asyncio
    async def test_empty_function_body_pass_only_fails(
        self, pipeline: SafetyPipeline
    ) -> None:
        """Function with only 'pass' in body fails semantic check."""
        code = "def foo() -> None:\n    pass\n"
        result = await pipeline.validate(code)

        semantic_layer = result["layers"]["semantic"]
        assert semantic_layer["passed"] is False
        assert any("empty body" in issue and "foo" in issue for issue in semantic_layer["issues"])

    @pytest.mark.asyncio
    async def test_docstring_only_function_fails(
        self, pipeline: SafetyPipeline
    ) -> None:
        """Function with only a docstring in body fails semantic check."""
        code = 'def foo() -> None:\n    """This is a docstring."""\n'
        result = await pipeline.validate(code)

        semantic_layer = result["layers"]["semantic"]
        assert semantic_layer["passed"] is False
        assert any("docstring-only" in issue and "foo" in issue for issue in semantic_layer["issues"])

    @pytest.mark.asyncio
    async def test_no_definitions_over_5_lines_fails(
        self, pipeline: SafetyPipeline
    ) -> None:
        """Code with more than 5 lines but no function/class definitions fails."""
        code = (
            "x = 1\n"
            "y = 2\n"
            "z = 3\n"
            "w = 4\n"
            "v = 5\n"
            "u = 6\n"
        )
        result = await pipeline.validate(code)

        semantic_layer = result["layers"]["semantic"]
        assert semantic_layer["passed"] is False
        assert any("no function or class definitions" in issue for issue in semantic_layer["issues"])

    @pytest.mark.asyncio
    async def test_short_code_without_definitions_passes(
        self, pipeline: SafetyPipeline
    ) -> None:
        """Short code (<6 lines) passes even without function/class definitions."""
        code = "x = 1\ny = 2\n"
        result = await pipeline.validate(code)

        semantic_layer = result["layers"]["semantic"]
        assert semantic_layer["passed"] is True
        assert semantic_layer["issues"] == []

    @pytest.mark.asyncio
    async def test_syntax_error_in_semantic_check_fails(
        self, pipeline: SafetyPipeline
    ) -> None:
        """Code with syntax errors fails semantic check with parse error."""
        code = "def foo(:\n    pass\n"
        result = await pipeline.validate(code)

        semantic_layer = result["layers"]["semantic"]
        assert semantic_layer["passed"] is False
        assert any("syntax error" in issue.lower() for issue in semantic_layer["issues"])

    @pytest.mark.asyncio
    async def test_class_definition_passes_semantic(
        self, pipeline: SafetyPipeline
    ) -> None:
        """Code with a class definition passes the definitions check."""
        code = (
            "class Calculator:\n"
            "    def add(self, a: int, b: int) -> int:\n"
            "        return a + b\n"
        )
        result = await pipeline.validate(code)

        semantic_layer = result["layers"]["semantic"]
        assert semantic_layer["passed"] is True


# ─── Full Pipeline Integration Tests ──────────────────────────────────


class TestFullPipelineIntegration:
    """Tests that all 8 layers (7 behavioral + safety-preservation) work together."""

    @pytest.mark.asyncio
    async def test_all_seven_layers_pass_with_mock_sandbox(
        self, pipeline: SafetyPipeline
    ) -> None:
        """All 8 layers pass with clean code and a mock sandbox executor."""
        mock_sandbox = MagicMock()
        mock_sandbox.execute_code = AsyncMock(return_value=SandboxResult(
            success=True, exit_code=0, stdout="ok", stderr="",
            duration_seconds=0.1, memory_mb=None, timed_out=False,
        ))

        code = (
            "def add(a: int, b: int) -> int:\n"
            "    result = a + b\n"
            "    return result\n"
        )
        result = await pipeline.validate(code, sandbox_executor=mock_sandbox)

        assert result["passed"] is True
        assert result["issues"] == []
        # Verify all 8 layers are present and passed (7 behavioral + preservation)
        assert len(result["layers"]) == 8
        for layer_name, layer_result in result["layers"].items():
            assert layer_result["passed"] is True, f"Layer {layer_name} did not pass"

    @pytest.mark.asyncio
    async def test_sandbox_failure_overrides_otherwise_clean_code(
        self, pipeline: SafetyPipeline
    ) -> None:
        """Clean code fails overall when sandbox execution fails."""
        mock_sandbox = MagicMock()
        mock_sandbox.execute_code = AsyncMock(return_value=SandboxResult(
            success=False, exit_code=1, stdout="", stderr="RuntimeError",
            duration_seconds=0.5, memory_mb=None, timed_out=False,
        ))

        code = (
            "def add(a: int, b: int) -> int:\n"
            "    result = a + b\n"
            "    return result\n"
        )
        result = await pipeline.validate(code, sandbox_executor=mock_sandbox)

        # Layers 1-5 and 7 should pass, but layer 6 (sandbox) fails
        assert result["layers"]["sandbox"]["passed"] is False
        assert result["passed"] is False
        assert len(result["issues"]) > 0
