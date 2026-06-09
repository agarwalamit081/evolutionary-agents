"""Tests for src.tools.builtin — built-in tool handlers and definitions."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.tools.builtin import ALL_TOOL_DEFINITIONS
from src.tools.builtin.code_executor import code_executor
from src.tools.builtin.code_validator import code_validator
from src.tools.builtin.file_reader import file_reader
from src.tools.builtin.file_writer import file_writer
from src.tools.builtin.self_inspect import self_inspect


class TestCodeExecutor:
    """Tests for the code_executor tool."""

    @pytest.mark.asyncio
    async def test_simple_code(self) -> None:
        """Simple print statement executes and returns output."""
        result = await code_executor('print("hello from test")')
        assert "hello from test" in result

    @pytest.mark.asyncio
    async def test_syntax_error(self) -> None:
        """Invalid Python returns error information."""
        result = await code_executor("def incomplete(")
        assert "error" in result.lower() or "syntax" in result.lower()

    @pytest.mark.asyncio
    async def test_timeout(self) -> None:
        """Infinite loop with timeout=1 returns timeout message."""
        result = await code_executor("while True: pass", timeout=1)
        assert "timeout" in result.lower() or "timed out" in result.lower()


class TestFileReader:
    """Tests for the file_reader tool."""

    @pytest.mark.asyncio
    async def test_reads_existing_file(self, tmp_path: Path) -> None:
        """Reading an existing file returns its content."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello file content")
        result = await file_reader(str(test_file), sandbox_root=str(tmp_path))
        assert "hello file content" in result

    @pytest.mark.asyncio
    async def test_nonexistent_file(self) -> None:
        """Reading a missing file returns error message."""
        result = await file_reader("/nonexistent/path/file.txt")
        assert "error" in result.lower() or "not found" in result.lower()


class TestCodeValidator:
    """Tests for the code_validator tool."""

    @pytest.mark.asyncio
    async def test_valid_code(self) -> None:
        """Valid Python code passes validation."""
        result = await code_validator("x = 1 + 2\nprint(x)")
        assert "valid" in result.lower() or "pass" in result.lower()

    @pytest.mark.asyncio
    async def test_invalid_code(self) -> None:
        """Invalid Python code fails validation."""
        result = await code_validator("def incomplete(")
        assert "error" in result.lower() or "invalid" in result.lower()


class TestFileWriter:
    """Tests for the file_writer tool."""

    @pytest.mark.asyncio
    async def test_writes_file(self, tmp_path: Path) -> None:
        """Writing to a file creates it with correct content."""
        test_file = tmp_path / "output.txt"
        result = await file_writer(str(test_file), "written content", sandbox_root=str(tmp_path))
        assert "success" in result.lower() or "written" in result.lower()
        assert test_file.read_text() == "written content"


class TestSelfInspect:
    """Tests for the self_inspect tool."""

    @pytest.mark.asyncio
    async def test_returns_info(self) -> None:
        """self_inspect returns non-empty string."""
        result = await self_inspect()
        assert len(result) > 0


class TestToolDefinitions:
    """Tests for ALL_TOOL_DEFINITIONS schema."""

    def test_all_definitions_have_required_fields(self) -> None:
        """Every tool definition has name, handler, description, parameters."""
        for tool_def in ALL_TOOL_DEFINITIONS:
            assert "name" in tool_def, f"Missing 'name' in {tool_def}"
            assert "handler" in tool_def, f"Missing 'handler' in {tool_def['name']}"
            assert "description" in tool_def, f"Missing 'description' in {tool_def['name']}"
            assert "parameters" in tool_def, f"Missing 'parameters' in {tool_def['name']}"

    def test_all_handlers_are_callable(self) -> None:
        """Every tool handler is callable."""
        for tool_def in ALL_TOOL_DEFINITIONS:
            assert callable(tool_def["handler"]), f"Handler not callable for {tool_def['name']}"

    def test_seven_tools_registered(self) -> None:
        """Exactly 7 built-in tools are registered."""
        assert len(ALL_TOOL_DEFINITIONS) == 7
