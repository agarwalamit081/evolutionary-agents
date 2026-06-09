"""Tests for src.safety.pipeline.SafetyPipeline."""

from __future__ import annotations

import pytest

from src.safety.pipeline import SafetyPipeline


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
