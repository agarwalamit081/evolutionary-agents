"""Tests for src.tools.builtin — built-in tool handlers and definitions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

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


class TestCodeExecutorCWD:
    """Tests for code_executor setting CWD to results directory."""

    @pytest.mark.asyncio
    async def test_file_created_in_results_dir(self, tmp_path: Path) -> None:
        """Files created with relative paths land in the results directory."""
        from src.config.settings import AgentSettings

        mock_settings = AgentSettings(results_root=str(tmp_path))
        with patch("src.tools.builtin.code_executor.get_settings", return_value=type("S", (), {"agent": mock_settings})):
            await code_executor(
                "import pathlib; pathlib.Path('test_output.txt').write_text('hello')"
            )
        assert (tmp_path / "test_output.txt").exists()
        assert (tmp_path / "test_output.txt").read_text() == "hello"

    @pytest.mark.asyncio
    async def test_results_dir_auto_created(self, tmp_path: Path) -> None:
        """Results directory is created automatically if missing."""
        nested = tmp_path / "nested" / "results"
        from src.config.settings import AgentSettings

        mock_settings = AgentSettings(results_root=str(nested))
        with patch("src.tools.builtin.code_executor.get_settings", return_value=type("S", (), {"agent": mock_settings})):
            await code_executor("print('ok')")
        assert nested.exists()


class TestFileWriterResultsDir:
    """Tests for file_writer defaulting to results_root."""

    @pytest.mark.asyncio
    async def test_default_sandbox_is_results_root(self, tmp_path: Path) -> None:
        """file_writer without explicit sandbox_root uses results_root."""
        from src.config.settings import AgentSettings

        mock_settings = AgentSettings(results_root=str(tmp_path))
        with patch("src.tools.builtin.file_writer.get_settings", return_value=type("S", (), {"agent": mock_settings})):
            result = await file_writer("test.txt", "results content", create_dirs=True)

        assert "success" in result.lower() or "wrote" in result.lower()
        assert (tmp_path / "test.txt").read_text() == "results content"

    @pytest.mark.asyncio
    async def test_strips_results_prefix_no_double_nesting(self, tmp_path: Path) -> None:
        """A goal-style 'results/<file>' path resolves under the root, not nested."""
        from src.config.settings import AgentSettings

        mock_settings = AgentSettings(results_root=str(tmp_path))
        with patch(
            "src.tools.builtin.file_writer.get_settings",
            return_value=type("S", (), {"agent": mock_settings}),
        ):
            result = await file_writer("results/report.html", "<html/>", create_dirs=True)

        assert "success" in result.lower() or "wrote" in result.lower()
        # De-nested: lands directly under the workspace root, not under results/.
        assert (tmp_path / "report.html").exists()
        assert (tmp_path / "report.html").read_text() == "<html/>"
        assert not (tmp_path / "results" / "report.html").exists()

    @pytest.mark.asyncio
    async def test_strips_results_prefix_preserves_subfolder(self, tmp_path: Path) -> None:
        """'results/<sub>/<file>' de-nests while preserving the subfolder."""
        from src.config.settings import AgentSettings

        mock_settings = AgentSettings(results_root=str(tmp_path))
        with patch(
            "src.tools.builtin.file_writer.get_settings",
            return_value=type("S", (), {"agent": mock_settings}),
        ):
            result = await file_writer(
                "results/design_patterns/singleton.md", "# Singleton", create_dirs=True
            )

        assert "success" in result.lower() or "wrote" in result.lower()
        assert (tmp_path / "design_patterns" / "singleton.md").exists()
        assert not (tmp_path / "results" / "design_patterns").exists()
