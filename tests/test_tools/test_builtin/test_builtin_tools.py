"""Tests for src.tools.builtin — built-in tool handlers and definitions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.tools.builtin import ALL_TOOL_DEFINITIONS
from src.tools.builtin.code_executor import code_executor
from src.tools.builtin.code_validator import code_validator
from src.tools.builtin.document_parser import document_parser
from src.tools.builtin.environment_inspect import environment_inspect
from src.tools.builtin.file_reader import file_reader
from src.tools.builtin.file_writer import file_writer
from src.tools.builtin.get_current_time import get_current_time
from src.tools.builtin.http_request import http_request
from src.tools.builtin.list_directory import list_directory
from src.tools.builtin.self_inspect import self_inspect
from src.tools.builtin.terminal_command import terminal_command
from src.tools.builtin.web_scraper import web_scraper
from src.tools.builtin.web_search import web_search


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

    def test_fourteen_tools_registered(self) -> None:
        """Exactly 14 built-in tools are registered (7 original + 7 new)."""
        assert len(ALL_TOOL_DEFINITIONS) == 14


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


# ─── New-tool tests (WS1 ddgs rewrite + WS2/WS3/WS4 additions) ─────────


class TestWebSearch:
    """Tests for the ddgs-based web_search tool (mocked ddgs client)."""

    @pytest.mark.asyncio
    async def test_returns_formatted_results(self) -> None:
        """Results are formatted as title / snippet / URL."""
        fake = [{"title": "T", "href": "http://example.com/x", "body": "B"}]
        with patch("src.tools.builtin.web_search._ddgs_text", return_value=fake):
            result = await web_search("test query")
        assert "T" in result and "http://example.com/x" in result and "B" in result

    @pytest.mark.asyncio
    async def test_no_results_message(self) -> None:
        """Empty results yield a clear 'no results' message."""
        with patch("src.tools.builtin.web_search._ddgs_text", return_value=[]):
            result = await web_search("obscure query")
        assert "No results" in result

    @pytest.mark.asyncio
    async def test_backend_failure_returns_error(self) -> None:
        """When all backends fail, an ERROR string is returned (not raised)."""
        from ddgs.exceptions import DDGSException

        with patch(
            "src.tools.builtin.web_search._ddgs_text", side_effect=DDGSException("boom")
        ):
            result = await web_search("fail query")
        assert "ERROR" in result


class TestGetCurrentTime:
    """Tests for the get_current_time tool."""

    @pytest.mark.asyncio
    async def test_utc_default(self) -> None:
        """Default call returns a UTC ISO timestamp."""
        result = await get_current_time()
        assert "UTC" in result and "T" in result

    @pytest.mark.asyncio
    async def test_named_timezone(self) -> None:
        """A valid IANA zone is honored."""
        result = await get_current_time("America/New_York")
        assert "America/New_York" in result

    @pytest.mark.asyncio
    async def test_invalid_timezone_falls_back(self) -> None:
        """An unknown zone falls back to UTC with a note."""
        result = await get_current_time("Bogus/Zone")
        assert "UTC" in result and "unknown timezone" in result


class TestEnvironmentInspect:
    """Tests for the environment_inspect tool."""

    @pytest.mark.asyncio
    async def test_summary(self) -> None:
        """Summary includes OS, Python, CPU, and disk info."""
        result = await environment_inspect("summary")
        assert "Python" in result and "CPU" in result and "Disk" in result

    @pytest.mark.asyncio
    async def test_packages(self) -> None:
        """Packages mode lists installed distributions."""
        result = await environment_inspect("packages")
        assert "package" in result.lower()


class TestListDirectory:
    """Tests for the list_directory tool."""

    @pytest.mark.asyncio
    async def test_lists_entries(self, tmp_path: Path) -> None:
        """Files and subdirectories are listed."""
        (tmp_path / "a.txt").write_text("x")
        (tmp_path / "subdir").mkdir()
        result = await list_directory(".", sandbox_root=str(tmp_path))
        assert "a.txt" in result and "subdir" in result

    @pytest.mark.asyncio
    async def test_traversal_blocked(self, tmp_path: Path) -> None:
        """Escaping the sandbox root is rejected."""
        result = await list_directory("../../etc", sandbox_root=str(tmp_path))
        assert "traversal" in result.lower() or "error" in result.lower()

    @pytest.mark.asyncio
    async def test_entry_cap(self, tmp_path: Path) -> None:
        """Listing is capped at max_entries."""
        for i in range(5):
            (tmp_path / f"f{i}.txt").write_text("x")
        result = await list_directory(".", max_entries=2, sandbox_root=str(tmp_path))
        assert "showing first 2" in result


class TestWebScraper:
    """Tests for the web_scraper tool (SSRF guard + content pass-through)."""

    @pytest.mark.asyncio
    async def test_ssrf_blocks_loopback(self) -> None:
        """A loopback URL is rejected before any network call."""
        result = await web_scraper("http://127.0.0.1:8080/")
        assert "ERROR" in result and "Blocked" in result

    @pytest.mark.asyncio
    async def test_ssrf_blocks_file_scheme(self) -> None:
        """A file:// URL is rejected (only http(s) allowed)."""
        result = await web_scraper("file:///etc/passwd")
        assert "ERROR" in result and "http" in result.lower()

    @pytest.mark.asyncio
    async def test_content_pass_through(self) -> None:
        """Extracted markdown is returned by the wrapper."""
        with patch(
            "src.tools.builtin.web_scraper._extract", return_value="# Title\nbody text"
        ):
            result = await web_scraper("https://example.com")
        assert "Title" in result and "body text" in result


class TestDocumentParser:
    """Tests for the document_parser tool (per-extension dispatch)."""

    @pytest.mark.asyncio
    async def test_parses_txt(self, tmp_path: Path) -> None:
        """Plain text is extracted verbatim."""
        (tmp_path / "note.txt").write_text("hello world")
        result = await document_parser("note.txt", sandbox_root=str(tmp_path))
        assert "hello world" in result

    @pytest.mark.asyncio
    async def test_parses_csv(self, tmp_path: Path) -> None:
        """CSV rows are extracted."""
        (tmp_path / "data.csv").write_text("a,b\n1,2\n")
        result = await document_parser("data.csv", sandbox_root=str(tmp_path))
        assert "1,2" in result

    @pytest.mark.asyncio
    async def test_parses_xlsx(self, tmp_path: Path) -> None:
        """Excel cell values are extracted."""
        from openpyxl import Workbook

        wb = Workbook()
        ws = wb.active
        assert ws is not None  # openpyxl stubs type active as Optional
        ws["A1"] = "alpha"
        wb.save(str(tmp_path / "sheet.xlsx"))
        result = await document_parser("sheet.xlsx", sandbox_root=str(tmp_path))
        assert "alpha" in result

    @pytest.mark.asyncio
    async def test_parses_docx(self, tmp_path: Path) -> None:
        """Word paragraph text is extracted."""
        from docx import Document

        doc = Document()
        doc.add_paragraph("docx body")
        doc.save(str(tmp_path / "doc.docx"))
        result = await document_parser("doc.docx", sandbox_root=str(tmp_path))
        assert "docx body" in result

    @pytest.mark.asyncio
    async def test_unsupported_extension(self, tmp_path: Path) -> None:
        """An unsupported type yields a clear error."""
        (tmp_path / "x.unknownext").write_text("nope")
        result = await document_parser("x.unknownext", sandbox_root=str(tmp_path))
        assert "ERROR" in result and "Unsupported" in result

    @pytest.mark.asyncio
    async def test_traversal_blocked(self, tmp_path: Path) -> None:
        """Escaping the sandbox root is rejected."""
        result = await document_parser("../../etc/passwd", sandbox_root=str(tmp_path))
        assert "traversal" in result.lower() or "error" in result.lower()

    @pytest.mark.asyncio
    async def test_truncation(self, tmp_path: Path) -> None:
        """Output is truncated to max_chars."""
        (tmp_path / "big.txt").write_text("y" * 500)
        result = await document_parser("big.txt", max_chars=50, sandbox_root=str(tmp_path))
        assert "truncated" in result.lower()


class TestHttpRequest:
    """Tests for the http_request tool (method allowlist + SSRF guard)."""

    @pytest.mark.asyncio
    async def test_method_not_allowed(self) -> None:
        """A non-allowlisted method is rejected."""
        result = await http_request("https://example.com", method="TRACE")
        assert "ERROR" in result and "not allowed" in result

    @pytest.mark.asyncio
    async def test_ssrf_blocks_loopback(self) -> None:
        """A loopback URL is rejected before any network call."""
        result = await http_request("http://127.0.0.1/admin")
        assert "ERROR" in result and "Blocked" in result

    @pytest.mark.asyncio
    async def test_get_success(self) -> None:
        """A successful GET returns status line + body (httpx mocked)."""

        class _FakeResp:
            status_code = 200
            text = '{"ok": true}'
            headers = {"content-type": "application/json"}

        class _FakeClient:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                pass

            async def __aenter__(self) -> "_FakeClient":
                return self

            async def __aexit__(self, *_args: object) -> bool:
                return False

            async def request(self, *_args: object, **_kwargs: object) -> "_FakeResp":
                return _FakeResp()

        with patch(
            "src.tools.builtin.http_request.assert_public_host", return_value=None
        ), patch("src.tools.builtin.http_request.httpx.AsyncClient", _FakeClient):
            result = await http_request("https://example.com/api")
        assert "HTTP 200" in result and "ok" in result

    @pytest.mark.asyncio
    async def test_body_too_large(self) -> None:
        """An oversized request body is rejected."""
        big = "x" * 2_000_000
        with patch(
            "src.tools.builtin.http_request.assert_public_host", return_value=None
        ):
            result = await http_request("https://example.com", method="POST", body=big)
        assert "ERROR" in result and "body" in result.lower()


class TestTerminalCommand:
    """Tests for the terminal_command tool (4-layer security defense)."""

    def _settings(self, tmp_path: Path) -> object:
        from src.config.settings import AgentSettings

        return type("S", (), {"agent": AgentSettings(workspace_root=str(tmp_path))})

    @pytest.mark.asyncio
    async def test_disallowed_command_rejected(self) -> None:
        """rm is not in the allowlist."""
        result = await terminal_command("rm", ["-rf", "/"])
        assert "ERROR" in result and "not allowed" in result

    @pytest.mark.asyncio
    async def test_git_mutating_subcommand_blocked(self) -> None:
        """git push is rejected (read-only sub-commands only)."""
        result = await terminal_command("git", ["push"])
        assert "ERROR" in result and "not allowed" in result

    @pytest.mark.asyncio
    async def test_find_exec_predicate_blocked(self) -> None:
        """find -exec is rejected (no command execution via find)."""
        result = await terminal_command("find", [".", "-exec", "rm", "{}", ";"])
        assert "ERROR" in result and "-exec" in result

    @pytest.mark.asyncio
    async def test_curl_post_flag_blocked(self) -> None:
        """curl -X POST is rejected (GET-only)."""
        result = await terminal_command("curl", ["-X", "POST", "http://example.com"])
        assert "ERROR" in result and "blocked" in result.lower()

    @pytest.mark.asyncio
    async def test_cwd_traversal_blocked(self, tmp_path: Path) -> None:
        """cwd outside the allowed roots is rejected."""
        with patch(
            "src.tools.builtin.terminal_command.get_settings",
            return_value=self._settings(tmp_path),
        ):
            result = await terminal_command("ls", cwd="../../etc")
        assert "ERROR" in result and "outside" in result.lower()

    @pytest.mark.asyncio
    async def test_shell_metacharacters_literal(self, tmp_path: Path) -> None:
        """Shell metacharacters are passed literally — no shell injection."""
        with patch(
            "src.tools.builtin.terminal_command.get_settings",
            return_value=self._settings(tmp_path),
        ):
            result = await terminal_command("echo", ["hello; rm -rf /", "$(whoami)"])
        assert "hello; rm -rf / $(whoami)" in result

    @pytest.mark.asyncio
    async def test_real_ls(self, tmp_path: Path) -> None:
        """A real ls lists a file (list-form subprocess, no shell)."""
        (tmp_path / "marker.txt").write_text("x")
        with patch(
            "src.tools.builtin.terminal_command.get_settings",
            return_value=self._settings(tmp_path),
        ):
            result = await terminal_command("ls", cwd=".")
        assert "marker.txt" in result
