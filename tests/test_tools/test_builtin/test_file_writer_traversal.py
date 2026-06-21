"""Tests for src.tools.builtin.file_writer — sandbox + path-traversal safety.

Verifies file_writer resolves writes under its sandbox root, blocks path
traversal, de-nests a leading workspace component, and enforces the 1MB cap.
Uses pytest's tmp_path as an isolated sandbox_root so no repo path is touched.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.tools.builtin.file_writer import file_writer


class TestFileWriterSandbox:
    @pytest.mark.asyncio
    async def test_writes_under_sandbox_and_creates_dirs(self, tmp_path: Path) -> None:
        msg = await file_writer(
            "sub/out.md",
            "hello world",
            sandbox_root=str(tmp_path),
            create_dirs=True,
        )
        assert msg.startswith("Successfully wrote")
        written = (tmp_path / "sub" / "out.md").read_text(encoding="utf-8")
        assert written == "hello world"

    @pytest.mark.asyncio
    async def test_blocks_path_traversal(self, tmp_path: Path) -> None:
        msg = await file_writer(
            "../../evil_target",
            "pwned",
            sandbox_root=str(tmp_path),
        )
        assert "Path traversal blocked" in msg
        # nothing escaped the sandbox
        assert not (tmp_path.parent.parent / "evil_target").exists()

    @pytest.mark.asyncio
    async def test_denests_leading_workspace_component(self, tmp_path: Path) -> None:
        """A 'results/' prefix in the path is stripped so it lands at the root."""
        msg = await file_writer(
            "results/n1.md",
            "x",
            sandbox_root=str(tmp_path),
        )
        assert msg.startswith("Successfully wrote")
        assert (tmp_path / "n1.md").exists()          # de-nested to root
        assert not (tmp_path / "results" / "n1.md").exists()  # not double-nested

    @pytest.mark.asyncio
    async def test_rejects_oversized_content(self, tmp_path: Path) -> None:
        msg = await file_writer(
            "big.txt",
            "x" * 1_000_001,
            sandbox_root=str(tmp_path),
        )
        assert "Content too large" in msg
        assert not (tmp_path / "big.txt").exists()
