"""Tests for src.evolution.git_tracker — GitTracker shadow repository management."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.evolution.git_tracker import GitTracker


class TestGitTracker:
    """Tests for the GitTracker class."""

    def _create_source_dir(self) -> Path:
        """Create a temporary source directory with sample files."""
        source = Path(tempfile.mkdtemp(prefix="turing_test_src_"))
        (source / "main.py").write_text("print('hello')", encoding="utf-8")
        (source / "utils.py").write_text("def helper(): pass", encoding="utf-8")
        sub = source / "subdir"
        sub.mkdir()
        (sub / "nested.py").write_text("x = 1", encoding="utf-8")
        return source

    def _create_repo_dir(self) -> Path:
        """Create a temporary repo directory path (does not create the dir)."""
        return Path(tempfile.mkdtemp(prefix="turing_test_repo_"))

    # ── Initialize tests ─────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_initialize_creates_repo(self) -> None:
        """Initialize creates .git directory and has initial commit."""
        source_dir = self._create_source_dir()
        repo_dir = self._create_repo_dir()
        try:
            tracker = GitTracker(source_dir, repo_dir)
            await tracker.initialize()

            # .git should exist
            assert (repo_dir / ".git").exists()

            # Source files should be copied
            assert (repo_dir / "main.py").exists()
            assert (repo_dir / "utils.py").exists()
            assert (repo_dir / "subdir" / "nested.py").exists()

            # HEAD should point to a valid commit (initial commit)
            head_hash = await tracker.get_current_hash()
            assert len(head_hash) == 40  # Full SHA-1 hex string
        finally:
            import shutil
            shutil.rmtree(source_dir, ignore_errors=True)
            shutil.rmtree(repo_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self) -> None:
        """Calling initialize twice is a no-op (does not error or reset)."""
        source_dir = self._create_source_dir()
        repo_dir = self._create_repo_dir()
        try:
            tracker = GitTracker(source_dir, repo_dir)
            await tracker.initialize()
            first_hash = await tracker.get_current_hash()

            # Initialize again
            await tracker.initialize()
            second_hash = await tracker.get_current_hash()

            # Hash should be the same — no new commit created
            assert first_hash == second_hash
        finally:
            import shutil
            shutil.rmtree(source_dir, ignore_errors=True)
            shutil.rmtree(repo_dir, ignore_errors=True)

    # ── Snapshot tests ───────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_snapshot_returns_hash(self) -> None:
        """Snapshot returns a non-empty hash string."""
        source_dir = self._create_source_dir()
        repo_dir = self._create_repo_dir()
        try:
            tracker = GitTracker(source_dir, repo_dir)
            await tracker.initialize()

            commit_hash = await tracker.snapshot("test snapshot")
            assert isinstance(commit_hash, str)
            assert len(commit_hash) == 40
        finally:
            import shutil
            shutil.rmtree(source_dir, ignore_errors=True)
            shutil.rmtree(repo_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_snapshot_creates_new_commit(self) -> None:
        """Snapshot creates a new commit distinct from the previous one."""
        source_dir = self._create_source_dir()
        repo_dir = self._create_repo_dir()
        try:
            tracker = GitTracker(source_dir, repo_dir)
            await tracker.initialize()
            first_hash = await tracker.get_current_hash()

            # Make a change and snapshot
            await tracker.apply_mutation("new_file.py", "x = 42")
            second_hash = await tracker.snapshot("added new file")

            assert second_hash != first_hash
        finally:
            import shutil
            shutil.rmtree(source_dir, ignore_errors=True)
            shutil.rmtree(repo_dir, ignore_errors=True)

    # ── Apply mutation tests ─────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_apply_mutation_writes_file(self) -> None:
        """Apply mutation writes file content to the shadow repo."""
        source_dir = self._create_source_dir()
        repo_dir = self._create_repo_dir()
        try:
            tracker = GitTracker(source_dir, repo_dir)
            await tracker.initialize()

            await tracker.apply_mutation("graph/prompts.py", "# mutated prompt\npass")

            target_file = repo_dir / "graph" / "prompts.py"
            assert target_file.exists()
            assert target_file.read_text(encoding="utf-8") == "# mutated prompt\npass"
        finally:
            import shutil
            shutil.rmtree(source_dir, ignore_errors=True)
            shutil.rmtree(repo_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_apply_mutation_overwrites_existing(self) -> None:
        """Apply mutation overwrites an existing file in the repo."""
        source_dir = self._create_source_dir()
        repo_dir = self._create_repo_dir()
        try:
            tracker = GitTracker(source_dir, repo_dir)
            await tracker.initialize()

            await tracker.apply_mutation("main.py", "print('mutated')")
            content = (repo_dir / "main.py").read_text(encoding="utf-8")
            assert content == "print('mutated')"
        finally:
            import shutil
            shutil.rmtree(source_dir, ignore_errors=True)
            shutil.rmtree(repo_dir, ignore_errors=True)

    # ── Get diff tests ───────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_get_diff_after_mutation(self) -> None:
        """Get diff after a mutation returns a non-empty diff string."""
        source_dir = self._create_source_dir()
        repo_dir = self._create_repo_dir()
        try:
            tracker = GitTracker(source_dir, repo_dir)
            await tracker.initialize()

            # Apply a mutation and snapshot it
            await tracker.apply_mutation("main.py", "print('changed')")
            await tracker.snapshot("mutation: change main.py")

            diff = await tracker.get_diff()
            assert isinstance(diff, str)
            assert len(diff) > 0
            assert "changed" in diff
        finally:
            import shutil
            shutil.rmtree(source_dir, ignore_errors=True)
            shutil.rmtree(repo_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_get_diff_with_since_hash(self) -> None:
        """Get diff since a specific commit returns changes only after that commit."""
        source_dir = self._create_source_dir()
        repo_dir = self._create_repo_dir()
        try:
            tracker = GitTracker(source_dir, repo_dir)
            await tracker.initialize()
            initial_hash = await tracker.get_current_hash()

            # First mutation
            await tracker.apply_mutation("main.py", "print('first')")
            await tracker.snapshot("first mutation")

            # Second mutation
            await tracker.apply_mutation("utils.py", "def new(): pass")
            await tracker.snapshot("second mutation")

            # Diff since initial should include both changes
            diff = await tracker.get_diff(since_hash=initial_hash)
            assert "first" in diff
            assert "new" in diff
        finally:
            import shutil
            shutil.rmtree(source_dir, ignore_errors=True)
            shutil.rmtree(repo_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_get_diff_no_changes(self) -> None:
        """Get diff with no changes returns empty string."""
        source_dir = self._create_source_dir()
        repo_dir = self._create_repo_dir()
        try:
            tracker = GitTracker(source_dir, repo_dir)
            await tracker.initialize()
            first_hash = await tracker.get_current_hash()

            # No changes committed
            diff = await tracker.get_diff(since_hash=first_hash)
            assert diff == ""
        finally:
            import shutil
            shutil.rmtree(source_dir, ignore_errors=True)
            shutil.rmtree(repo_dir, ignore_errors=True)

    # ── Get current hash tests ───────────────────────────────────────

    @pytest.mark.asyncio
    async def test_get_current_hash(self) -> None:
        """get_current_hash returns a valid 40-character hex hash after init."""
        source_dir = self._create_source_dir()
        repo_dir = self._create_repo_dir()
        try:
            tracker = GitTracker(source_dir, repo_dir)
            await tracker.initialize()

            current_hash = await tracker.get_current_hash()
            assert isinstance(current_hash, str)
            assert len(current_hash) == 40
            # Should be a valid hex string
            int(current_hash, 16)  # Will raise ValueError if not hex
        finally:
            import shutil
            shutil.rmtree(source_dir, ignore_errors=True)
            shutil.rmtree(repo_dir, ignore_errors=True)

    # ── Get log tests ────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_get_log(self) -> None:
        """get_log returns list of dicts with hash/message/date keys."""
        source_dir = self._create_source_dir()
        repo_dir = self._create_repo_dir()
        try:
            tracker = GitTracker(source_dir, repo_dir)
            await tracker.initialize()

            # Create additional commits
            await tracker.apply_mutation("file1.py", "x = 1")
            await tracker.snapshot("commit: file1")
            await tracker.apply_mutation("file2.py", "y = 2")
            await tracker.snapshot("commit: file2")

            log = await tracker.get_log(max_entries=10)
            assert isinstance(log, list)
            assert len(log) == 3  # Initial + 2 snapshots

            # Each entry has the expected keys
            for entry in log:
                assert "hash" in entry
                assert "message" in entry
                assert "date" in entry
                assert len(entry["hash"]) == 40

            # Most recent commit should be first
            assert "file2" in log[0]["message"]
        finally:
            import shutil
            shutil.rmtree(source_dir, ignore_errors=True)
            shutil.rmtree(repo_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_get_log_with_max_entries(self) -> None:
        """get_log respects max_entries parameter."""
        source_dir = self._create_source_dir()
        repo_dir = self._create_repo_dir()
        try:
            tracker = GitTracker(source_dir, repo_dir)
            await tracker.initialize()

            # Create 3 additional commits
            for i in range(3):
                await tracker.apply_mutation(f"file{i}.py", f"x = {i}")
                await tracker.snapshot(f"commit {i}")

            log = await tracker.get_log(max_entries=2)
            assert len(log) == 2
        finally:
            import shutil
            shutil.rmtree(source_dir, ignore_errors=True)
            shutil.rmtree(repo_dir, ignore_errors=True)

    # ── Rollback tests ───────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_rollback(self) -> None:
        """Rollback reverts file content to a previous commit."""
        source_dir = self._create_source_dir()
        repo_dir = self._create_repo_dir()
        try:
            tracker = GitTracker(source_dir, repo_dir)
            await tracker.initialize()
            initial_hash = await tracker.get_current_hash()

            # Modify main.py and commit
            await tracker.apply_mutation("main.py", "print('mutated')")
            await tracker.snapshot("mutation applied")

            # Verify mutation is in place
            assert (repo_dir / "main.py").read_text(encoding="utf-8") == "print('mutated')"

            # Rollback to initial commit
            await tracker.rollback(initial_hash)

            # File should be reverted
            content = (repo_dir / "main.py").read_text(encoding="utf-8")
            assert content == "print('hello')"
        finally:
            import shutil
            shutil.rmtree(source_dir, ignore_errors=True)
            shutil.rmtree(repo_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_rollback_multiple_commits(self) -> None:
        """Rollback to the first commit reverts all subsequent changes."""
        source_dir = self._create_source_dir()
        repo_dir = self._create_repo_dir()
        try:
            tracker = GitTracker(source_dir, repo_dir)
            await tracker.initialize()
            initial_hash = await tracker.get_current_hash()

            # First mutation
            await tracker.apply_mutation("main.py", "print('v1')")
            await tracker.snapshot("v1")

            # Second mutation
            await tracker.apply_mutation("main.py", "print('v2')")
            await tracker.snapshot("v2")

            # Third mutation
            await tracker.apply_mutation("utils.py", "def changed(): return True")
            await tracker.snapshot("v3")

            # Rollback all the way to initial
            await tracker.rollback(initial_hash)

            assert (repo_dir / "main.py").read_text(encoding="utf-8") == "print('hello')"
            assert (repo_dir / "utils.py").read_text(encoding="utf-8") == "def helper(): pass"
        finally:
            import shutil
            shutil.rmtree(source_dir, ignore_errors=True)
            shutil.rmtree(repo_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_rollback_removes_untracked_files(self) -> None:
        """reset --hard + clean -fd removes files added since the target commit.

        This is the M4 fix: the old ``git checkout <hash> -- .`` left untracked
        files (written by apply_mutation but never snapshotted) in place, so the
        working tree never actually matched the rolled-back commit.
        """
        source_dir = self._create_source_dir()
        repo_dir = self._create_repo_dir()
        try:
            tracker = GitTracker(source_dir, repo_dir)
            await tracker.initialize()
            initial_hash = await tracker.get_current_hash()

            # Write a file WITHOUT snapshotting it → untracked on disk.
            await tracker.apply_mutation("untracked.py", "x = 999")
            assert (repo_dir / "untracked.py").exists()

            ok = await tracker.rollback(initial_hash)

            assert ok is True
            assert not (repo_dir / "untracked.py").exists()  # clean -fd removed it
        finally:
            import shutil
            shutil.rmtree(source_dir, ignore_errors=True)
            shutil.rmtree(repo_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_rollback_returns_bool(self) -> None:
        """rollback returns True on success, False on an invalid commit hash."""
        source_dir = self._create_source_dir()
        repo_dir = self._create_repo_dir()
        try:
            tracker = GitTracker(source_dir, repo_dir)
            await tracker.initialize()
            initial_hash = await tracker.get_current_hash()

            assert await tracker.rollback(initial_hash) is True
            assert await tracker.rollback("not_a_valid_commit_hash_xyz") is False
        finally:
            import shutil
            shutil.rmtree(source_dir, ignore_errors=True)
            shutil.rmtree(repo_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_rollback_restores_tracked_and_removes_untracked(self) -> None:
        """A full rollback both restores modified files and drops added ones."""
        source_dir = self._create_source_dir()
        repo_dir = self._create_repo_dir()
        try:
            tracker = GitTracker(source_dir, repo_dir)
            await tracker.initialize()
            initial_hash = await tracker.get_current_hash()

            # Commit a modification to a tracked file.
            await tracker.apply_mutation("main.py", "print('mutated')")
            await tracker.snapshot("mutate main")
            # Add an untracked file (no snapshot).
            await tracker.apply_mutation("stray.py", "x = 1")
            assert (repo_dir / "stray.py").exists()

            ok = await tracker.rollback(initial_hash)

            assert ok is True
            assert (repo_dir / "main.py").read_text(encoding="utf-8") == "print('hello')"
            assert not (repo_dir / "stray.py").exists()
        finally:
            import shutil
            shutil.rmtree(source_dir, ignore_errors=True)
            shutil.rmtree(repo_dir, ignore_errors=True)


class TestCommitPaths:
    """Phase 5 G2 — explicit-path commit primitive for the MAIN repo (never ``-A``).

    ``commit_paths`` stages ONLY the named paths (unlike ``snapshot``'s ``git add
    -A``), so an automated promotion commit never sweeps machine-specific local
    overrides (pyrightconfig / alembic.ini / scratch scripts) into git history.
    """

    def _create_source_dir(self) -> Path:
        """A minimal source dir (commit_paths needs an initialized repo, not
        rich content)."""
        source = Path(tempfile.mkdtemp(prefix="turing_test_src_"))
        (source / "main.py").write_text("print('hello')", encoding="utf-8")
        return source

    def _create_repo_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="turing_test_repo_"))

    def _tracked(self, repo_dir: Path, path: str) -> str:
        """``git ls-files`` for one path (empty string ⇒ not tracked)."""
        import subprocess

        return subprocess.run(
            ["git", "-C", str(repo_dir), "ls-files", "--", path],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    @pytest.mark.asyncio
    async def test_commits_named_file_and_leaves_others_untracked(self) -> None:
        """The named artifact is committed; a co-resident scratch file is NOT."""
        import shutil

        source_dir = self._create_source_dir()
        repo_dir = self._create_repo_dir()
        try:
            tracker = GitTracker(source_dir, repo_dir)
            await tracker.initialize()
            before = await tracker.get_current_hash()

            # The named tracked artifact + an unrelated machine-local scratch file.
            (repo_dir / "prompts_evolved").mkdir(parents=True, exist_ok=True)
            artifact = repo_dir / "prompts_evolved" / "execute.deadbeef.json"
            artifact.write_text('{"node": "execute"}', encoding="utf-8")
            (repo_dir / "scratch_local.txt").write_text("machine-specific", encoding="utf-8")

            sha = await tracker.commit_paths(
                ["prompts_evolved/execute.deadbeef.json"],
                "chore(evolution): promote execute prompt",
            )

            assert isinstance(sha, str)
            assert len(sha) == 40
            # A new commit was created.
            assert await tracker.get_current_hash() != before
            # The named artifact is tracked...
            assert self._tracked(repo_dir, "prompts_evolved/execute.deadbeef.json") != ""
            # ...but the scratch file was NOT swept in (explicit path, never -A).
            assert self._tracked(repo_dir, "scratch_local.txt") == ""
        finally:
            shutil.rmtree(source_dir, ignore_errors=True)
            shutil.rmtree(repo_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_empty_paths_returns_empty(self) -> None:
        """An empty path list is a benign no-op (returns ``""``, no git call)."""
        import shutil

        source_dir = self._create_source_dir()
        repo_dir = self._create_repo_dir()
        try:
            tracker = GitTracker(source_dir, repo_dir)
            await tracker.initialize()

            assert await tracker.commit_paths([], "nothing") == ""
        finally:
            shutil.rmtree(source_dir, ignore_errors=True)
            shutil.rmtree(repo_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_nothing_to_commit_returns_empty(self) -> None:
        """Re-committing an unchanged file is a benign no-op (rc 1 → ``""``)."""
        import shutil

        source_dir = self._create_source_dir()
        repo_dir = self._create_repo_dir()
        try:
            tracker = GitTracker(source_dir, repo_dir)
            await tracker.initialize()
            artifact = repo_dir / "execute.abc123.json"
            artifact.write_text('{"node": "execute"}', encoding="utf-8")

            first = await tracker.commit_paths(["execute.abc123.json"], "first")
            assert first != ""
            # Second commit with no change → nothing to commit → "".
            assert await tracker.commit_paths(["execute.abc123.json"], "second") == ""
        finally:
            shutil.rmtree(source_dir, ignore_errors=True)
            shutil.rmtree(repo_dir, ignore_errors=True)
