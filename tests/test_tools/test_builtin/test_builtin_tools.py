"""Tests for src.tools.builtin — built-in tool handlers and definitions."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import httpx
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
from src.tools.builtin.web_scraper import (
    ExtractedPage,
    chunk_text,
    compute_content_hash,
    extract_page,
    web_scraper,
)
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

    @pytest.mark.asyncio
    async def test_falls_back_to_results_root_for_outputs(self, tmp_path: Path) -> None:
        """F13: file_reader reads a file written to results_root via file_writer.

        The default sandbox is workspace_root, but file_writer writes to
        results_root. A file that exists only under results_root must still be
        readable under the default sandbox so the agent can confirm its own
        output (and avoid false-negative verify verdicts).
        """
        from src.config.settings import AgentSettings

        workspace = tmp_path / "workspace"
        results = tmp_path / "results"
        workspace.mkdir()
        results.mkdir()
        (results / "out.txt").write_text("agent output")

        mock_settings = AgentSettings(
            workspace_root=str(workspace), results_root=str(results)
        )
        with patch(
            "src.config.settings.get_settings",
            return_value=type("S", (), {"agent": mock_settings}),
        ):
            result = await file_reader("out.txt")  # default sandbox
        assert "agent output" in result


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

    @pytest.mark.asyncio
    async def test_creates_nested_parents_by_default(self, tmp_path: Path) -> None:
        """create_dirs defaults to True — nested writes land without a prior mkdir.

        Regression for battery-04 q3: with create_dirs defaulting to False, a
        write to a nested run subfolder that didn't exist (e.g.
        results/q03/retention.csv right after --clean) hit the parent-exists
        guard and silently returned ``ERROR: Parent directory does not exist``
        — no log, so the agent believed nothing was wrong and never converged.
        The default is now True so deliverables under per-run subfolders land.
        """
        nested = tmp_path / "q03" / "retention.csv"
        # Omit create_dirs entirely — the default must create the parent.
        result = await file_writer(str(nested), "cohort,week0,week1\n", sandbox_root=str(tmp_path))
        assert "success" in result.lower() or "written" in result.lower(), result
        assert nested.exists()
        assert nested.read_text() == "cohort,week0,week1\n"
        # Parent directory was auto-created.
        assert nested.parent.exists()

    @pytest.mark.asyncio
    async def test_explicit_create_dirs_false_still_blocks_missing_parent(self, tmp_path: Path) -> None:
        """create_dirs=False keeps the original guard behavior."""
        nested = tmp_path / "missing" / "x.txt"
        result = await file_writer(str(nested), "x", sandbox_root=str(tmp_path), create_dirs=False)
        assert "error" in result.lower()
        assert not nested.exists()


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

    def test_twenty_three_tools_registered(self) -> None:
        """Exactly 23 built-in tools are registered (14 audited + arxiv_search + 2 corpus tools + ocr_parser + image_generator + create_scheduled_task + git_clone + code_search + lean4_runner)."""
        assert len(ALL_TOOL_DEFINITIONS) == 23


class TestCodeExecutorCWD:
    """Tests for code_executor running at the shared project root."""

    @pytest.mark.asyncio
    async def test_file_created_in_results_dir(self, tmp_path: Path) -> None:
        """A script writing ``results/<file>`` lands it under results_root.

        cwd is the project root (parent of results_root), so a deliverable must be
        written explicitly under ``results/``. With results_root=tmp_path/results,
        project_root==tmp_path and ``results/<file>`` resolves into results_root —
        the parity that previously broke (double-nest → empty glob → fabrication).
        """
        from src.config.settings import AgentSettings

        mock_settings = AgentSettings(results_root=str(tmp_path / "results"))
        with patch("src.config.settings.get_settings", return_value=type("S", (), {"agent": mock_settings})):
            await code_executor(
                "with open('results/test_output.txt', 'w') as f:\n"
                "    f.write('hello')"
            )
        assert (tmp_path / "results" / "test_output.txt").exists()
        assert (tmp_path / "results" / "test_output.txt").read_text() == "hello"

    @pytest.mark.asyncio
    async def test_project_root_auto_created(self, tmp_path: Path) -> None:
        """code_executor creates its cwd (project_root) if missing.

        project_root = parent of results_root; code_executor mkdir's it so the
        subprocess always has a valid working directory.
        """
        nested = tmp_path / "nested" / "results"
        from src.config.settings import AgentSettings

        mock_settings = AgentSettings(results_root=str(nested))
        with patch("src.config.settings.get_settings", return_value=type("S", (), {"agent": mock_settings})):
            await code_executor("print('ok')")
        # cwd = project_root = nested.parent, auto-created.
        assert (tmp_path / "nested").exists()

    @pytest.mark.asyncio
    async def test_subdir_writes_auto_create_parents(self, tmp_path: Path) -> None:
        """F8: ``open('results/subdir/file')`` auto-creates the subdir.

        The ``_write_bootstrap`` shim patches ``builtins.open`` so a generator
        script that writes to a relative nested path under results/ (e.g.
        ``results/design_patterns/x.md``) succeeds instead of failing on a missing
        parent directory — the gap that left ``results/design_patterns/`` empty.
        """
        from src.config.settings import AgentSettings

        mock_settings = AgentSettings(results_root=str(tmp_path / "results"))
        with patch("src.config.settings.get_settings", return_value=type("S", (), {"agent": mock_settings})):
            await code_executor(
                "with open('results/design_patterns/singleton.md', 'w') as f:\n"
                "    f.write('pattern')"
            )
        target = tmp_path / "results" / "design_patterns" / "singleton.md"
        assert target.exists()
        assert target.read_text() == "pattern"

    @pytest.mark.asyncio
    async def test_bare_write_relocates_into_results_not_project_root(
        self, tmp_path: Path
    ) -> None:
        """#314: a BARE relative write (no ``results/`` prefix) must NOT pollute
        the project root.

        ``code_executor`` runs the subprocess with ``cwd = project_root()``
        (the results dir's parent). Without relocation, an LLM script doing
        ``open("vector_db_comparator.py", "w")`` wrote the file into the repo
        root — polluting ``ruff check .`` and shadowing real modules on
        ``sys.path``. The bootstrap relocates bare relative writes into
        ``results/`` so the deliverable lands there instead.
        """
        from src.config.settings import AgentSettings

        mock_settings = AgentSettings(results_root=str(tmp_path / "results"))
        with patch("src.config.settings.get_settings", return_value=type("S", (), {"agent": mock_settings})):
            await code_executor(
                "with open('vector_db_comparator.py', 'w') as f:\n"
                "    f.write('handler source')\n"
            )

        # Relocated into results/ …
        relocated = tmp_path / "results" / "vector_db_comparator.py"
        assert relocated.exists()
        assert relocated.read_text() == "handler source"
        # … and NOT leaked into the project root (the cwd / repo root).
        assert not (tmp_path / "vector_db_comparator.py").exists()

    @pytest.mark.asyncio
    async def test_results_prefixed_write_not_relocated_or_double_nested(
        self, tmp_path: Path
    ) -> None:
        """Non-regression for #314: a write already under ``results/`` is left
        exactly where the ``results/<file>`` contract puts it — not relocated
        away, and not double-nested to ``results/results/<file>``."""
        from src.config.settings import AgentSettings

        mock_settings = AgentSettings(results_root=str(tmp_path / "results"))
        with patch("src.config.settings.get_settings", return_value=type("S", (), {"agent": mock_settings})):
            await code_executor(
                "with open('results/report.md', 'w') as f:\n"
                "    f.write('body')\n"
            )

        assert (tmp_path / "results" / "report.md").exists()
        assert (tmp_path / "results" / "report.md").read_text() == "body"
        # No double-nesting, no project-root leak.
        assert not (tmp_path / "results" / "results" / "report.md").exists()
        assert not (tmp_path / "report.md").exists()

    @pytest.mark.asyncio
    async def test_bootstrap_defaults_mplbackend_agg(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """D6: the ``_write_bootstrap`` shim defaults ``MPLBACKEND=Agg`` so a
        generated tool that imports matplotlib and calls ``savefig`` works
        headless — there is no display server in host subprocess / runner mode.
        ``setdefault`` honors a caller that already set the backend, so clear the
        host value to assert the injection engages deterministically.
        """
        from src.config.settings import AgentSettings

        monkeypatch.delenv("MPLBACKEND", raising=False)
        mock_settings = AgentSettings(results_root=str(tmp_path / "results"))
        with patch("src.config.settings.get_settings", return_value=type("S", (), {"agent": mock_settings})):
            out = await code_executor(
                "import os\n"
                "print(os.environ.get('MPLBACKEND'))\n"
            )
        assert "Agg" in out, out


class TestCodeExecutorRunSubdir:
    """Per-run subfoldering: code_executor deliverables isolate under
    ``results/<run_id>/`` (no flat leakage) and reads round-trip / fall back."""

    @pytest.mark.asyncio
    async def test_write_isolates_under_run_subdir(self, tmp_path: Path) -> None:
        """``results/<file>`` lands under ``results/<run_id>/<file>``, not flat."""
        from src.config.settings import AgentSettings
        from src.tools._paths import set_active_run_id

        mock_settings = AgentSettings(results_root=str(tmp_path / "results"))
        set_active_run_id("q09")
        try:
            with patch(
                "src.config.settings.get_settings",
                return_value=type("S", (), {"agent": mock_settings}),
            ):
                await code_executor(
                    "with open('results/transactions.csv', 'w') as f:\n"
                    "    f.write('DATA')\n"
                )
        finally:
            set_active_run_id(None)

        assert (tmp_path / "results" / "q09" / "transactions.csv").exists()
        assert (tmp_path / "results" / "q09" / "transactions.csv").read_text() == "DATA"
        # No flat leakage into the shared results root.
        assert not (tmp_path / "results" / "transactions.csv").exists()

    @pytest.mark.asyncio
    async def test_bare_write_lands_under_run_subdir(self, tmp_path: Path) -> None:
        """A BARE relative write (no ``results/`` prefix) lands under
        ``results/<run_id>/`` — not in the project root (the cwd)."""
        from src.config.settings import AgentSettings
        from src.tools._paths import set_active_run_id

        mock_settings = AgentSettings(results_root=str(tmp_path / "results"))
        set_active_run_id("q09")
        try:
            with patch(
                "src.config.settings.get_settings",
                return_value=type("S", (), {"agent": mock_settings}),
            ):
                await code_executor(
                    "with open('compute.py', 'w') as f:\n"
                    "    f.write('CODE')\n"
                )
        finally:
            set_active_run_id(None)

        assert (tmp_path / "results" / "q09" / "compute.py").exists()
        assert (tmp_path / "results" / "q09" / "compute.py").read_text() == "CODE"
        # cwd = parent of results_root = tmp_path; the bare write must not land there.
        assert not (tmp_path / "compute.py").exists()
        assert not (tmp_path / "results" / "compute.py").exists()

    @pytest.mark.asyncio
    async def test_write_not_double_nested_in_run_subdir(self, tmp_path: Path) -> None:
        """A goal already naming ``results/<run_id>/<file>`` is NOT double-nested
        to ``results/<run_id>/<run_id>/<file>`` (the strip-set includes run_id)."""
        from src.config.settings import AgentSettings
        from src.tools._paths import set_active_run_id

        mock_settings = AgentSettings(results_root=str(tmp_path / "results"))
        set_active_run_id("q09")
        try:
            with patch(
                "src.config.settings.get_settings",
                return_value=type("S", (), {"agent": mock_settings}),
            ):
                await code_executor(
                    "with open('results/q09/report.md', 'w') as f:\n"
                    "    f.write('body')\n"
                )
        finally:
            set_active_run_id(None)

        assert (tmp_path / "results" / "q09" / "report.md").exists()
        assert (tmp_path / "results" / "q09" / "report.md").read_text() == "body"
        assert not (tmp_path / "results" / "q09" / "q09" / "report.md").exists()

    @pytest.mark.asyncio
    async def test_read_round_trips_after_subdir_write(self, tmp_path: Path) -> None:
        """A write then a relative read of the same path finds the subdir file."""
        from src.config.settings import AgentSettings
        from src.tools._paths import set_active_run_id

        mock_settings = AgentSettings(results_root=str(tmp_path / "results"))
        set_active_run_id("q09")
        try:
            with patch(
                "src.config.settings.get_settings",
                return_value=type("S", (), {"agent": mock_settings}),
            ):
                result = await code_executor(
                    "with open('results/foo.csv', 'w') as f:\n"
                    "    f.write('ROUNDTRIP')\n"
                    "print(open('results/foo.csv').read())\n"
                )
        finally:
            set_active_run_id(None)

        assert "ROUNDTRIP" in result
        assert (tmp_path / "results" / "q09" / "foo.csv").exists()

    @pytest.mark.asyncio
    async def test_read_falls_back_to_flat_for_legacy(self, tmp_path: Path) -> None:
        """A flat-only deliverable (prior run, no subdir copy) still recalls via
        the read flat-fallback — mirroring ``resolve_existing``."""
        from src.config.settings import AgentSettings
        from src.tools._paths import set_active_run_id

        results = tmp_path / "results"
        results.mkdir(parents=True, exist_ok=True)
        # Legacy FLAT deliverable from a prior run (NOT under q09/).
        (results / "legacy.csv").write_text("LEGACY")
        mock_settings = AgentSettings(results_root=str(results))
        set_active_run_id("q09")
        try:
            with patch(
                "src.config.settings.get_settings",
                return_value=type("S", (), {"agent": mock_settings}),
            ):
                result = await code_executor("print(open('legacy.csv').read())\n")
        finally:
            set_active_run_id(None)

        assert "LEGACY" in result
        # The read did not relocate/create a subdir copy.
        assert not (results / "q09" / "legacy.csv").exists()

    @pytest.mark.asyncio
    async def test_traversal_write_not_relocated_outside_run_subdir(
        self, tmp_path: Path
    ) -> None:
        """A traversal write (``../../escape``) is NOT relocated into the run cell
        — the traversal guard holds and the shim falls back to the original
        CWD-relative path rather than ever writing outside the cell via relocation."""
        from src.config.settings import AgentSettings
        from src.tools._paths import set_active_run_id

        # results_root two levels deep so the guard-failed CWD-relative fall-back
        # ("../../escape.txt" from cwd=project_root) stays inside tmp_path.
        results = tmp_path / "deep" / "results"
        mock_settings = AgentSettings(results_root=str(results))
        set_active_run_id("q09")
        try:
            with patch(
                "src.config.settings.get_settings",
                return_value=type("S", (), {"agent": mock_settings}),
            ):
                await code_executor(
                    "with open('../../escape.txt', 'w') as f:\n"
                    "    f.write('PWN')\n"
                )
        finally:
            set_active_run_id(None)

        # The traversal was NOT relocated into the run cell (or the results root).
        assert not (results / "q09" / "escape.txt").exists()
        assert not (results / "escape.txt").exists()


class TestCodeExecutorHostPathGuard:
    """D8 opt-in host path guard: when ``code_executor_host_path_guard`` is on,
    the bootstrap-injected ``open`` rejects any path resolving OUTSIDE the
    results/workspace tree — checked AFTER relocation, so a script still reads
    and writes its own deliverables/inputs but cannot reach the repo root
    (``open(".env")``) or an absolute escape (``open("/etc/hosts")``). OFF by
    default (the dispatch point aliases the real builtin, byte-identical)."""

    @pytest.mark.asyncio
    async def test_guard_on_blocks_absolute_escape(self, tmp_path: Path) -> None:
        """An absolute path OUTSIDE the guard roots is refused with a guard error."""
        from src.config.settings import AgentSettings

        secret = tmp_path / "outside_root.txt"  # sits at project_root, NOT under results/workspace
        secret.write_text("ROOT_SECRET")
        mock_settings = AgentSettings(
            results_root=str(tmp_path / "results"),
            workspace_root=str(tmp_path / "workspace"),
            code_executor_host_path_guard=True,
        )
        with patch(
            "src.config.settings.get_settings",
            return_value=type("S", (), {"agent": mock_settings}),
        ):
            result = await code_executor(f"print(open({str(secret)!r}).read())\n")
        # The read was blocked — the secret never printed, and the guard surfaced.
        assert "ROOT_SECRET" not in result
        assert "host path guard" in result.lower(), result

    @pytest.mark.asyncio
    async def test_guard_on_blocks_repo_root_env(self, tmp_path: Path) -> None:
        """A relative ``open('.env')`` resolves to the cwd (project_root = repo
        root in prod) which is NOT a guard root — so it is blocked, not read."""
        from src.config.settings import AgentSettings

        (tmp_path / ".env").write_text("API_KEY=sk-leak")  # cwd = tmp_path (project_root)
        mock_settings = AgentSettings(
            results_root=str(tmp_path / "results"),
            workspace_root=str(tmp_path / "workspace"),
            code_executor_host_path_guard=True,
        )
        with patch(
            "src.config.settings.get_settings",
            return_value=type("S", (), {"agent": mock_settings}),
        ):
            result = await code_executor("print(open('.env').read())\n")
        assert "sk-leak" not in result
        assert "host path guard" in result.lower(), result

    @pytest.mark.asyncio
    async def test_guard_on_allows_results_write_and_read(self, tmp_path: Path) -> None:
        """A legitimate deliverable under results/ (and an input under
        workspace/) passes the guard — confinement, not a blanket block."""
        from src.config.settings import AgentSettings

        (tmp_path / "workspace").mkdir(parents=True, exist_ok=True)
        (tmp_path / "workspace" / "fixture.txt").write_text("FIXTURE")
        mock_settings = AgentSettings(
            results_root=str(tmp_path / "results"),
            workspace_root=str(tmp_path / "workspace"),
            code_executor_host_path_guard=True,
        )
        with patch(
            "src.config.settings.get_settings",
            return_value=type("S", (), {"agent": mock_settings}),
        ):
            result = await code_executor(
                "with open('results/out.md', 'w') as f:\n"
                "    f.write('OK')\n"
                "print(open('results/out.md').read())\n"
                "print(open('workspace/fixture.txt').read())\n"
            )
        assert "OK" in result
        assert "FIXTURE" in result
        assert (tmp_path / "results" / "out.md").read_text() == "OK"

    @pytest.mark.asyncio
    async def test_guard_off_is_unchanged(self, tmp_path: Path) -> None:
        """OFF (default): the dispatch aliases the real builtin, so a path that
        the guard WOULD block reads normally — proves the off path is inert."""
        from src.config.settings import AgentSettings

        secret = tmp_path / "outside_root.txt"
        secret.write_text("ROOT_SECRET")
        mock_settings = AgentSettings(
            results_root=str(tmp_path / "results"),
            workspace_root=str(tmp_path / "workspace"),
            code_executor_host_path_guard=False,  # default
        )
        with patch(
            "src.config.settings.get_settings",
            return_value=type("S", (), {"agent": mock_settings}),
        ):
            result = await code_executor(f"print(open({str(secret)!r}).read())\n")
        assert "ROOT_SECRET" in result  # read succeeded — no guard engaged


class TestCodeExecutorHostCwd:
    """D8 cwd knob: ``code_executor_host_cwd`` (default ``project_root``) opts
    the host subprocess into ``cwd = results/<run_id>/`` (or ``results_root``
    when no run_id) for tighter disk isolation. The default MUST stay
    ``project_root`` so a ``results/<file>`` write + ``glob('results/*.md')``
    resolve uniformly across every file-touching tool."""

    @pytest.mark.asyncio
    async def test_results_subdir_cwd_is_run_cell(self, tmp_path: Path) -> None:
        """``results_subdir`` sets the subprocess cwd to results/<run_id>/, and a
        bare write still isolates there (isolation holds under the tighter cwd)."""
        from src.config.settings import AgentSettings
        from src.tools._paths import set_active_run_id

        mock_settings = AgentSettings(
            results_root=str(tmp_path / "results"),
            workspace_root=str(tmp_path / "workspace"),
            code_executor_host_cwd="results_subdir",
            results_per_run_subdir=True,
        )
        set_active_run_id("q09")
        try:
            with patch(
                "src.config.settings.get_settings",
                return_value=type("S", (), {"agent": mock_settings}),
            ):
                result = await code_executor(
                    "import os\n"
                    "print('CWD=' + os.getcwd())\n"
                    "with open('bare.txt', 'w') as f:\n"
                    "    f.write('X')\n"
                )
        finally:
            set_active_run_id(None)

        cwd_line = next(  # pragma: no branch
            line for line in result.splitlines() if line.startswith("CWD=")
        )
        subprocess_cwd = Path(cwd_line[len("CWD="):])
        assert subprocess_cwd.name == "q09"  # cwd is the run subdir, not project root
        assert subprocess_cwd.parent == (tmp_path / "results").resolve()
        # Bare write landed in the run cell — no flat/project-root leakage.
        assert (tmp_path / "results" / "q09" / "bare.txt").read_text() == "X"
        assert not (tmp_path / "results" / "bare.txt").exists()

    @pytest.mark.asyncio
    async def test_default_cwd_is_project_root(self, tmp_path: Path) -> None:
        """Default ``project_root`` cwd is unchanged — the documented contract
        that ``results/<file>`` resolves the same as every other tool."""
        from src.config.settings import AgentSettings

        mock_settings = AgentSettings(
            results_root=str(tmp_path / "results"),
            workspace_root=str(tmp_path / "workspace"),
            code_executor_host_cwd="project_root",  # default, explicit
        )
        with patch(
            "src.config.settings.get_settings",
            return_value=type("S", (), {"agent": mock_settings}),
        ):
            result = await code_executor("import os\nprint('CWD=' + os.getcwd())\n")
        cwd_line = next(  # pragma: no branch
            line for line in result.splitlines() if line.startswith("CWD=")
        )
        subprocess_cwd = Path(cwd_line[len("CWD="):])
        assert subprocess_cwd == tmp_path  # project_root = parent of results_root
        assert subprocess_cwd == (tmp_path / "results").resolve().parent


class TestFileWriterResultsDir:
    """Tests for file_writer defaulting to results_root."""

    @pytest.mark.asyncio
    async def test_default_sandbox_is_results_root(self, tmp_path: Path) -> None:
        """file_writer without explicit sandbox_root uses results_root."""
        from src.config.settings import AgentSettings

        mock_settings = AgentSettings(results_root=str(tmp_path))
        with patch("src.config.settings.get_settings", return_value=type("S", (), {"agent": mock_settings})):
            result = await file_writer("test.txt", "results content", create_dirs=True)

        assert "success" in result.lower() or "wrote" in result.lower()
        assert (tmp_path / "test.txt").read_text() == "results content"

    @pytest.mark.asyncio
    async def test_strips_results_prefix_no_double_nesting(self, tmp_path: Path) -> None:
        """A goal-style 'results/<file>' path resolves under the root, not nested."""
        from src.config.settings import AgentSettings

        mock_settings = AgentSettings(results_root=str(tmp_path))
        with patch(
            "src.config.settings.get_settings",
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
            "src.config.settings.get_settings",
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
    """Tests for the SearXNG-primary web_search tool (mocked fetch seam)."""

    @pytest.mark.asyncio
    async def test_returns_formatted_results(self) -> None:
        """Results are formatted as title / snippet / URL."""
        fake = [{"title": "T", "href": "http://example.com/x", "body": "B"}]
        with patch("src.tools.builtin.web_search._fetch_results", return_value=fake):
            result = await web_search("test query")
        assert "T" in result and "http://example.com/x" in result and "B" in result

    @pytest.mark.asyncio
    async def test_no_results_message(self) -> None:
        """Empty results yield a clear 'no results' message."""
        with patch("src.tools.builtin.web_search._fetch_results", return_value=[]):
            result = await web_search("obscure query")
        assert "No results" in result

    @pytest.mark.asyncio
    async def test_fetch_failure_returns_error(self) -> None:
        """When the fetch layer raises, an ERROR string is returned (not raised)."""
        with patch(
            "src.tools.builtin.web_search._fetch_results",
            side_effect=RuntimeError("boom"),
        ):
            result = await web_search("fail query")
        assert "ERROR" in result

    @pytest.mark.asyncio
    async def test_filters_spam_domains(self) -> None:
        """Blocked spam domains are removed from results."""
        fake = [
            {"title": "Spam", "href": "https://www.pinterest.com/pin/123", "body": "x"},
            {"title": "Good", "href": "https://example.com/article", "body": "y"},
        ]
        with patch("src.tools.builtin.web_search._fetch_results", return_value=fake):
            result = await web_search("query")
        assert "Good" in result
        assert "pinterest.com" not in result

    @pytest.mark.asyncio
    async def test_dedups_by_canonical_url(self) -> None:
        """Same URL with differing tracking params collapses to one result."""
        fake = [
            {"title": "A", "href": "https://example.com/a?utm_source=x", "body": "b"},
            {"title": "A2", "href": "https://example.com/a", "body": "b"},
        ]
        with patch("src.tools.builtin.web_search._fetch_results", return_value=fake):
            result = await web_search("query")
        assert result.count("URL:") == 1  # deduped to a single result
        assert "utm_source" not in result  # tracking param stripped from output

    @pytest.mark.asyncio
    async def test_unwraps_ddg_redirect(self) -> None:
        """DDG redirect wrapper is unwrapped to the real URL in output."""
        redirect = (
            "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Freal&rut=abc"
        )
        fake = [{"title": "Real", "href": redirect, "body": "content"}]
        with patch("src.tools.builtin.web_search._fetch_results", return_value=fake):
            result = await web_search("query")
        assert "https://example.com/real" in result
        assert "duckduckgo.com/l/" not in result

    @pytest.mark.asyncio
    async def test_empty_query_returns_error(self) -> None:
        """A whitespace-only query is rejected before any network call."""
        from unittest.mock import MagicMock

        mock = MagicMock(return_value=[])
        with patch("src.tools.builtin.web_search._fetch_results", mock):
            result = await web_search("   ")
        assert "ERROR" in result
        assert mock.call_count == 0  # never reached the fetcher

    @pytest.mark.asyncio
    async def test_batch_mode_returns_one_block_per_query(self) -> None:
        """queries=[...] fans out and yields one Query-prefixed block each."""
        per_query = [
            [{"title": "T1", "href": "http://a.com/1", "body": "B1"}],
            [],  # second query has no results
        ]
        with patch(
            "src.tools.builtin.web_search._fetch_batch", return_value=per_query
        ):
            result = await web_search(queries=["alpha", "beta"])
        assert 'Query: "alpha"' in result
        assert "T1" in result
        # second query's no-results block is present
        assert 'Query: "beta"' in result
        assert "No results" in result


class TestSearXNGFetcher:
    """Unit tests for the SearXNG primary fetcher (mocked httpx transport)."""

    @pytest.mark.asyncio
    async def test_searxng_call_normalizes_results(self) -> None:
        """SearXNG JSON is normalized to title/href/body + params carry region/time."""
        from src.tools.builtin.web_search import _searxng_call

        seen: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(request.url.params)
            return httpx.Response(
                200,
                json={"results": [
                    {"title": "T", "url": "https://x.com/a", "content": "snippet"},
                ]},
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            rows = await _searxng_call(client, "q", 5, "us-en", "w")

        assert rows == [{"title": "T", "href": "https://x.com/a", "body": "snippet"}]
        # region us-en -> SearXNG language en-US; timelimit w -> time_range week.
        assert seen["format"] == "json"
        assert seen["language"] == "en-US"
        assert seen["time_range"] == "week"
        assert seen["safesearch"] == "2"

    @pytest.mark.asyncio
    async def test_transient_5xx_is_retried_then_raises(self) -> None:
        """A persistent 5xx is retried (transient) then surfaces as an error."""
        from src.tools.builtin.web_search import (
            _searxng_fetch,
            TransientSearchError,
        )

        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(503, text="upstream down")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(TransientSearchError):
                await _searxng_fetch(client, "q", 5, "us-en", "")

        # Retried up to WEB_SEARCH_MAX_ATTEMPTS (default 3) before giving up.
        assert calls["n"] == 3


class TestFallbackChain:
    """Tests for the SearXNG → paid-provider fallback orchestrator."""

    @pytest.mark.asyncio
    async def test_searxng_success_short_circuits_paid(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When SearXNG returns results, no paid provider is called."""
        from src.tools.builtin import web_search as ws

        monkeypatch.setenv("TAVILY_API_KEY", "k")
        tavily_calls = {"n": 0}

        async def fake_tavily(client, key, query, max_results):
            tavily_calls["n"] += 1
            return []

        monkeypatch.setitem(ws.PROVIDER_ADAPTERS, "tavily", fake_tavily)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"results": [
                {"title": "S", "url": "https://s.com", "content": "c"}]})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            rows = await ws._search_with_fallback(client, "q", 5, "us-en", "", False)

        assert rows and rows[0]["href"] == "https://s.com"
        assert tavily_calls["n"] == 0  # paid fallback never engaged

    @pytest.mark.asyncio
    async def test_searxng_down_falls_to_paid_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SearXNG 400 (non-transient) → keyed Tavily provider is tried next."""
        from src.tools.builtin import web_search as ws

        monkeypatch.setenv("TAVILY_API_KEY", "k")

        def handler(request: httpx.Request) -> httpx.Response:
            if "api.tavily.com" in str(request.url):
                return httpx.Response(200, json={"results": [
                    {"title": "TV", "url": "https://t.com", "content": "tc"}]})
            # SearXNG primary: non-transient 400 → provider unavailable, no retry.
            return httpx.Response(400, text="bad")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            rows = await ws._search_with_fallback(client, "q", 5, "us-en", "", False)

        assert rows and rows[0]["href"] == "https://t.com"

    @pytest.mark.asyncio
    async def test_no_keyed_providers_returns_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With SearXNG down and no provider keys, [] is returned (never raised)."""
        from src.tools.builtin import web_search as ws

        for var in (
            "TAVILY_API_KEY", "SERPER_API_KEY", "BRAVE_SEARCH_API_KEY",
            "SERPAPI_API_KEY", "SERPSTACK_API_KEY", "LLMLAYER_API_KEY",
        ):
            monkeypatch.delenv(var, raising=False)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, text="bad")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            rows = await ws._search_with_fallback(client, "q", 5, "us-en", "", False)

        assert rows == []


class TestBatchConcurrency:
    """The batch semaphore caps concurrent fetches at SEARCH_BATCH_CONCURRENCY."""

    @pytest.mark.asyncio
    async def test_concurrency_capped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from types import SimpleNamespace

        from src.tools.builtin import web_search as ws

        # get_settings() is a cached singleton, so setenv can't change the value
        # already read at construction — patch the accessor directly.
        monkeypatch.setattr(
            ws, "_search_settings",
            lambda: SimpleNamespace(search_batch_concurrency=2),
        )
        live = {"now": 0, "peak": 0}

        async def slow_fetch(query, max_results, region, timelimit, deep_crawl):
            live["now"] += 1
            live["peak"] = max(live["peak"], live["now"])
            await asyncio.sleep(0.02)
            live["now"] -= 1
            return [{"title": query, "href": f"http://x/{query}", "body": "b"}]

        monkeypatch.setattr(ws, "_fetch_results", slow_fetch)
        # 6 queries, concurrency capped at 2 → peak in-flight never exceeds 2.
        await ws._fetch_batch(["q1", "q2", "q3", "q4", "q5", "q6"], 5, "us-en", "", False)
        assert live["peak"] <= 2


class TestWebSearchCleaning:
    """Unit tests for web_search result-cleaning helpers (§6)."""

    def test_build_query_normalizes_whitespace(self) -> None:
        from src.tools.builtin.web_search import _build_query

        assert _build_query("  multi   word  query ") == "multi word query"

    def test_unwrap_redirect_extracts_inner_url(self) -> None:
        from src.tools.builtin.web_search import _unwrap_redirect

        redirect = "https://duckduckgo.com/l/?uddg=https%3A%2F%2Freal.com%2Fpage&rut=z"
        assert _unwrap_redirect(redirect) == "https://real.com/page"
        # Non-redirect URLs pass through unchanged.
        assert _unwrap_redirect("https://example.com/x") == "https://example.com/x"

    def test_canonicalize_url_strips_tracking_and_fragment(self) -> None:
        from src.tools.builtin.web_search import _canonicalize_url

        url = "HTTPS://Example.COM/A/?utm_source=x&fbclid=y#frag"
        assert _canonicalize_url(url) == "https://example.com/A"

    def test_is_spam_url_matches_domains_and_subdomains(self) -> None:
        from src.tools.builtin.web_search import _is_spam_url

        assert _is_spam_url("https://pinterest.com/x") is True
        assert _is_spam_url("https://www.pinterest.com/x") is True
        assert _is_spam_url("https://m.facebook.com/x") is True
        assert _is_spam_url("https://example.com/x") is False
        # Not fooled by substring lookalikes.
        assert _is_spam_url("https://not-pinterest.com/x") is False

    def test_clean_results_filters_spam_and_dedups(self) -> None:
        from src.tools.builtin.web_search import _clean_results

        results = [
            {"title": "Spam", "href": "https://pinterest.com/a", "body": "b1"},
            {"title": "Dup1", "href": "https://example.com/a?utm_source=x", "body": "b2"},
            {"title": "Dup2", "href": "https://example.com/a", "body": "b3"},
            {"title": "Keep", "href": "https://example.com/b", "body": "b4"},
        ]
        cleaned = _clean_results(results)
        hrefs = [r["href"] for r in cleaned]
        assert len(cleaned) == 2  # spam dropped, utm-duplicate collapsed
        assert "https://example.com/a" in hrefs
        assert "https://example.com/b" in hrefs
        assert all("pinterest.com" not in h for h in hrefs)


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


class TestWebScraperExtraction:
    """Tests for the AI-format extraction layer (Phase 1): hash, chunking, metadata."""

    _HTML = (
        "<!DOCTYPE html><html><head>"
        "<title>Example Article Title</title>"
        '<meta name="description" content="A short description of the page.">'
        '<meta name="author" content="Jane Doe">'
        "</head><body><article><h1>Example Article Title</h1>"
        "<p>This is the main body content of the article with enough words for "
        "trafilatura to treat it as the main readable text on the page being "
        "scraped for the purpose of testing the extraction pipeline end to end.</p>"
        "<p>Another paragraph adds more content so that the extraction returns a "
        "non-trivial body for the markdown and text fields used by downstream "
        "chunking and corpus indexing of the agent's gathered web research.</p>"
        "<p>A third paragraph of filler content further increases the signal so "
        "the main-content extractor reliably fires rather than skipping the page "
        "as too sparse to yield any usable article text at all.</p>"
        "</article></body></html>"
    )

    def test_compute_content_hash_stable(self) -> None:
        """Same normalized content → same hash; whitespace-only diffs collapse."""
        a = compute_content_hash("hello   world\t\nfoo")
        b = compute_content_hash("hello world foo")
        assert a == b
        assert len(a) == 32
        assert all(c in "0123456789abcdef" for c in a)
        assert compute_content_hash("different content") != a

    def test_chunk_text_empty(self) -> None:
        """Empty/blank input yields no chunks."""
        assert chunk_text("") == []
        assert chunk_text("   \n\t  ") == []

    def test_chunk_text_char_mode_small_is_single(self) -> None:
        """Text shorter than chunk_size returns exactly one chunk."""
        out = chunk_text("short text", chunk_size=1200, chunk_overlap=150, mode="char")
        assert out == ["short text"]

    def test_chunk_text_char_mode_overlap(self) -> None:
        """Char mode with overlap produces >=2 overlapping chunks of bounded size."""
        text = "x" * 3000
        out = chunk_text(text, chunk_size=1000, chunk_overlap=200, mode="char")
        assert len(out) >= 2
        # Every chunk within the size cap.
        assert all(len(c) <= 1000 for c in out)
        # Overlap: adjacent chunks share the `overlap`-sized tail/head.
        assert out[0][-200:] == out[1][:200]
        # Full coverage of the source (last chunk reaches the end).
        assert out[-1].endswith("x")

    def test_chunk_text_token_mode_returns_chunks(self) -> None:
        """Token mode produces bounded chunks (or degrades to char mode offline)."""
        text = "word " * 4000
        out = chunk_text(text, chunk_size=500, chunk_overlap=50, mode="token")
        assert out  # non-empty
        # Either path is acceptable; token mode must still bound piece size in chars.
        assert all(len(c) <= 3000 for c in out)

    def test_chunk_text_overlap_clamped_to_size(self) -> None:
        """An overlap >= chunk_size is clamped so step stays >= 1 (no infinite loop)."""
        out = chunk_text("a" * 50, chunk_size=10, chunk_overlap=100, mode="char")
        assert out  # terminates and returns something
        assert all(len(c) <= 10 for c in out)

    def test_extract_page_metadata(self) -> None:
        """extract_page pulls title/description + computes a hash from raw HTML."""
        with patch(
            "src.tools.builtin.web_scraper._fetch_html", return_value=self._HTML
        ):
            page = extract_page("https://example.com/article", timeout=10.0, max_bytes=2_000_000)
        assert page.url == "https://example.com/article"
        assert "Example Article Title" in page.title
        assert page.content_hash and len(page.content_hash) == 32
        # Body extracted (markdown or plain text) and hash derived from it.
        assert (page.markdown or page.text)
        assert page.content_hash == compute_content_hash(page.markdown or page.text)
        # Metadata surfaces the title (best-effort; other fields optional).
        assert page.metadata.get("title") == page.title

    def test_extract_page_fetch_error_surfaces(self) -> None:
        """A fetch failure raises _FetchError (extract_page does not swallow it)."""
        from src.tools.builtin.web_scraper import _FetchError

        with patch(
            "src.tools.builtin.web_scraper._fetch_html",
            side_effect=_FetchError("ERROR: Only http/https URLs allowed"),
        ):
            with pytest.raises(_FetchError):
                extract_page("file:///etc/passwd", timeout=10.0, max_bytes=2_000_000)

    @pytest.mark.asyncio
    async def test_chunk_handler_returns_blocks(self) -> None:
        """web_scraper(chunk=True) returns joined [Chunk N/M] blocks."""
        from types import SimpleNamespace

        long_page = ExtractedPage(
            url="https://example.com/long",
            title="Long",
            description="",
            markdown="w" * 3000,
            text="",
            content_hash="0" * 32,
            metadata={},
        )
        with (
            patch("src.tools.builtin.web_scraper.extract_page", return_value=long_page),
            patch(
                "src.tools.builtin.web_scraper._search_settings",
                return_value=SimpleNamespace(chunk_size=1000, chunk_overlap=0),
            ),
        ):
            result = await web_scraper("https://example.com/long", chunk=True)
        # 3000 chars / step 1000 → 3 chunks.
        assert result.startswith("[Chunk 1/3]")
        assert "[Chunk 3/3]" in result
        assert result.count("[Chunk ") == 3


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

        # terminal_command cwd = project_root = parent of results_root. Set
        # results_root under tmp_path so project_root resolves to tmp_path and the
        # cwd-sandboxed commands (ls/echo) run there.
        return type("S", (), {"agent": AgentSettings(results_root=str(tmp_path / "results"))})

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
            "src.config.settings.get_settings",
            return_value=self._settings(tmp_path),
        ):
            result = await terminal_command("ls", cwd="../../etc")
        assert "ERROR" in result and "outside" in result.lower()

    @pytest.mark.asyncio
    async def test_shell_metacharacters_literal(self, tmp_path: Path) -> None:
        """Shell metacharacters are passed literally — no shell injection."""
        with patch(
            "src.config.settings.get_settings",
            return_value=self._settings(tmp_path),
        ):
            result = await terminal_command("echo", ["hello; rm -rf /", "$(whoami)"])
        assert "hello; rm -rf / $(whoami)" in result

    @pytest.mark.asyncio
    async def test_real_ls(self, tmp_path: Path) -> None:
        """A real ls lists a file (list-form subprocess, no shell)."""
        (tmp_path / "marker.txt").write_text("x")
        with patch(
            "src.config.settings.get_settings",
            return_value=self._settings(tmp_path),
        ):
            result = await terminal_command("ls", cwd=".")
        assert "marker.txt" in result


class TestCrossToolPathParity:
    """Headline regression for battery-02 B1 (fabrication).

    The original bug: each file tool resolved its own cwd (results_root vs
    workspace_root) with no shared resolver, so a ``code_executor`` script
    globbing ``results/*.md`` from inside ``results/`` found
    ``results/results/*.md`` (nothing) and the LLM fabricated output. With a
    single shared resolver + cwd aligned to ``project_root``, a path like
    ``results/<file>`` resolves identically whether written, executed, or cat'd.
    """

    @staticmethod
    def _settings(tmp_path: Path) -> object:
        from src.config.settings import AgentSettings

        # results_root under tmp_path → project_root == tmp_path.
        return type(
            "S", (), {"agent": AgentSettings(results_root=str(tmp_path / "results"))}
        )

    @pytest.mark.asyncio
    async def test_write_glob_and_cat_agree(self, tmp_path: Path) -> None:
        """file_writer, code_executor (glob), and terminal_command (cat) agree."""
        settings = self._settings(tmp_path)
        with patch("src.config.settings.get_settings", return_value=settings):
            # 1. Write a deliverable under results/ via file_writer.
            write_result = await file_writer(
                "results/report.md", "FOUND IT", create_dirs=True
            )
            assert "success" in write_result.lower() or "wrote" in write_result.lower()

            # 2. A code_executor script globbing results/*.md from project root
            #    MUST find the file (the empty-glob → fabrication path).
            glob_code = (
                "import glob\n"
                "files = sorted(glob.glob('results/*.md'))\n"
                "print('GLOB_COUNT=' + str(len(files)))\n"
                "print('GLOB_FILES=' + ','.join(f.split('/')[-1] for f in files))"
            )
            exec_out = await code_executor(glob_code)
            assert "GLOB_COUNT=1" in exec_out, exec_out
            assert "GLOB_FILES=report.md" in exec_out, exec_out

            # 3. terminal_command `cat results/report.md` reads the same bytes.
            cat_out = await terminal_command("cat", ["results/report.md"])
            assert "FOUND IT" in cat_out, cat_out
