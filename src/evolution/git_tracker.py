"""Git-based change tracking for the evolution engine.

Manages a shadow git repository that mirrors src/ and tracks
evolution mutations with full diff and rollback capability.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import aiofiles
from loguru import logger


class GitTracker:
    """Tracks agent mutations in a separate shadow git repository.

    The shadow repo is a copy of the source directory that the evolution
    engine uses to safely apply and test mutations without affecting the
    running agent. Each mutation is committed, enabling diff generation
    and rollback.
    """

    def __init__(self, source_dir: Path, repo_dir: Path) -> None:
        """Initialize the git tracker.

        Args:
            source_dir: Path to the source directory to mirror (e.g., Path("src")).
            repo_dir: Path where the shadow repo will live (e.g., Path(".turing/evolution-repo")).
        """
        self._source_dir = source_dir.resolve()
        self._repo_dir = repo_dir.resolve()

    @property
    def repo_dir(self) -> Path:
        """The shadow-repo root the engine applies mutations to (resolved).

        Exposed so stage-1 graph-invariant checks (``evolution.invariants``)
        can inspect the deployed snapshot in place — they read
        ``repo_dir/src/graph/{state,routers,task_graph}.py`` after a CODE
        mutation is applied, before the post-deploy sandbox smoke. Read-only
        consumers only; mutations go through ``apply_mutation`` / ``rollback``.
        """
        return self._repo_dir

    async def _git(self, *args: str) -> tuple[int, str, str]:
        """Run a git command in the shadow repo directory.

        Args:
            *args: Git command arguments (e.g., "add", "-A").

        Returns:
            Tuple of (returncode, stdout, stderr).
        """
        cmd = ["git", *args]
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self._repo_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await process.communicate()
            stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
            stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
            return (process.returncode or 0, stdout, stderr)
        except Exception as e:
            logger.error(f"Git command failed: {' '.join(cmd)} — {e}")
            return (1, "", str(e))

    async def initialize(self) -> None:
        """Initialize the shadow repository.

        Creates the repo directory if needed, copies source files, and
        performs git init + initial commit. If repo already exists with
        commits, this is a no-op.
        """
        try:
            # Already initialized if .git exists with commits
            git_dir = self._repo_dir / ".git"
            if git_dir.exists():
                returncode, stdout, _ = await self._git("rev-parse", "HEAD")
                if returncode == 0:
                    logger.debug(f"Shadow repo already initialized at {self._repo_dir}")
                    return

            # Create repo directory and copy source files
            self._repo_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(
                src=str(self._source_dir),
                dst=str(self._repo_dir),
                dirs_exist_ok=True,
            )

            # Initialize git repository
            await self._git("init")

            # Configure local user identity (never global)
            await self._git("config", "user.name", "Turing Agent")
            await self._git("config", "user.email", "agent@turing.local")

            # Create initial commit
            await self._git("add", "-A")
            returncode, _, stderr = await self._git(
                "commit", "-m", "Initial snapshot"
            )

            if returncode == 0:
                logger.info(f"Shadow repo initialized at {self._repo_dir}")
            else:
                logger.warning(
                    f"Initial commit may have failed (rc={returncode}): {stderr}"
                )
        except Exception as e:
            logger.error(f"Failed to initialize shadow repo: {e}")

    async def snapshot(self, message: str) -> str:
        """Create a snapshot (git add + commit) of current state.

        Args:
            message: Commit message describing the change.

        Returns:
            The commit hash of the new snapshot. Empty string on failure.
        """
        try:
            await self._git("add", "-A")
            await self._git("commit", "--allow-empty-message", "-m", message)

            returncode, stdout, _ = await self._git("rev-parse", "HEAD")
            if returncode == 0 and stdout:
                logger.debug(f"Snapshot created: {stdout[:12]}")
                return stdout
            logger.warning("Failed to get commit hash after snapshot")
            return ""
        except Exception as e:
            logger.error(f"Failed to create snapshot: {e}")
            return ""

    async def commit_paths(self, paths: list[str], message: str) -> str:
        """Stage EXPLICIT repo-relative paths and commit; return the HEAD hash.

        Unlike ``snapshot`` (which ``git add -A``s the whole shadow repo), this
        stages ONLY the named paths — the main-project-repo discipline (Phase 5
        G2), where ``-A`` would sweep machine-specific paths (pyrightconfig /
        alembic.ini local overrides) and scratch scripts into an automated
        commit. ``paths`` are repo-relative (or absolute) file paths under
        ``_repo_dir``. Used by the promotion gate to land a promoted prompt
        artifact in the main repo's VCS trail (not the shadow repo).

        Args:
            paths: Repo-relative file paths to stage explicitly (never ``-A``).
            message: Commit message.

        Returns:
            The new HEAD commit hash, or ``""`` on failure / nothing-to-commit.
            Never raises — the caller treats ``""`` as a benign no-op (the live
            pointer is the source of truth; a VCS commit is best-effort).
        """
        if not paths:
            return ""
        try:
            rc, _, stderr = await self._git("add", "--", *paths)
            if rc != 0:
                logger.warning(f"commit_paths: git add failed (rc={rc}): {stderr}")
                return ""
            rc, _, stderr = await self._git("commit", "-m", message)
            if rc != 0:
                # rc 1 "nothing to commit" is the benign no-op case (already committed).
                logger.debug(f"commit_paths: nothing committed (rc={rc}): {stderr}")
                return ""
            rc, stdout, _ = await self._git("rev-parse", "HEAD")
            if rc == 0 and stdout:
                logger.debug(f"commit_paths: committed {paths} → {stdout[:12]}")
                return stdout
            return ""
        except Exception as e:
            logger.error(f"commit_paths failed: {e}")
            return ""

    async def apply_mutation(self, file_path: str, content: str) -> None:
        """Apply a mutation by writing content to a file in the shadow repo.

        Args:
            file_path: Relative file path within the repo (e.g., "graph/prompts.py").
            content: The new file content.
        """
        try:
            target = self._repo_dir / file_path
            target.parent.mkdir(parents=True, exist_ok=True)

            async with aiofiles.open(target, mode="w", encoding="utf-8") as f:
                await f.write(content)

            logger.debug(f"Applied mutation to {file_path}")
        except Exception as e:
            logger.error(f"Failed to apply mutation to {file_path}: {e}")

    async def get_diff(self, since_hash: str | None = None) -> str:
        """Get the diff since a given commit.

        Args:
            since_hash: Commit hash to diff from. If None, diffs against HEAD~1.

        Returns:
            Git diff output as a string. Empty string on failure.
        """
        try:
            ref = since_hash if since_hash is not None else "HEAD~1"
            returncode, stdout, _ = await self._git("diff", ref)
            if returncode == 0:
                return stdout
            logger.warning(f"Git diff failed (rc={returncode})")
            return ""
        except Exception as e:
            logger.error(f"Failed to get diff: {e}")
            return ""

    async def rollback(self, commit_hash: str) -> bool:
        """Roll back the shadow repo to exactly match a specific commit.

        ``git reset --hard <hash>`` restores tracked files to that commit and
        moves HEAD onto it; ``git clean -fd`` then removes untracked files and
        directories added since. Together they make the working tree identical
        to the target commit — unlike ``git checkout <hash> -- .`` which neither
        moves HEAD nor removes files added after that commit, leaving the repo
        in a half-reverted state. Confined to the disposable shadow repo
        (``_repo_dir``); the running agent's ``src/`` is never touched.

        Args:
            commit_hash: The commit to roll back to.

        Returns:
            True if both reset and clean succeeded, False otherwise.
        """
        try:
            rc_reset, _, stderr_reset = await self._git(
                "reset", "--hard", commit_hash
            )
            if rc_reset != 0:
                logger.warning(
                    f"Rollback reset failed (rc={rc_reset}): {stderr_reset}"
                )
                return False
            rc_clean, _, stderr_clean = await self._git("clean", "-fd")
            if rc_clean != 0:
                logger.warning(
                    f"Rollback clean failed (rc={rc_clean}): {stderr_clean}"
                )
                return False
            logger.info(f"Rolled back shadow repo to {commit_hash[:12]}")
            return True
        except Exception as e:
            logger.error(f"Failed to rollback to {commit_hash}: {e}")
            return False

    async def get_current_hash(self) -> str:
        """Get the current HEAD commit hash.

        Returns:
            The full commit hash string. Empty string on failure.
        """
        try:
            returncode, stdout, _ = await self._git("rev-parse", "HEAD")
            if returncode == 0:
                return stdout
            logger.warning("Failed to get current HEAD hash")
            return ""
        except Exception as e:
            logger.error(f"Failed to get current hash: {e}")
            return ""

    async def get_log(self, max_entries: int = 10) -> list[dict[str, str]]:
        """Get recent commit log entries.

        Args:
            max_entries: Maximum number of entries to return.

        Returns:
            List of dicts with 'hash', 'message', 'date' keys.
        """
        try:
            returncode, stdout, _ = await self._git(
                "log", f"--max-count={max_entries}", "--format=%H|%s|%ai"
            )
            if returncode != 0 or not stdout:
                return []

            entries: list[dict[str, str]] = []
            for line in stdout.splitlines():
                parts = line.split("|", maxsplit=2)
                if len(parts) == 3:
                    entries.append({
                        "hash": parts[0],
                        "message": parts[1],
                        "date": parts[2],
                    })
            return entries
        except Exception as e:
            logger.error(f"Failed to get git log: {e}")
            return []
