"""Allowlisted, shell-free terminal command tool.

Gives the agent narrow, controlled access to read-oriented CLI utilities
(``ls``, ``cat``, ``grep``, ``jq``, ``git`` read-only, ``curl`` GET, …) that
the locked ``code_executor`` sandbox cannot provide. Defense in depth:

1. **Command allowlist** — exact-match; no ``rm``/``mv``/``cp``/``chmod``/``ssh``/``wget``.
2. **No shell** — invoked via ``create_subprocess_exec`` in *list form*
   (``shell=False``). Shell metacharacters (``|``, ``;``, ``$()``) become
   literal arguments, so shell injection is structurally impossible.
3. **Sub-command allowlists** for multi-mode tools: ``git`` read-only only;
   ``curl`` GET only; ``find`` dangerous predicates (``-exec``/``-delete``/…)
   blocked — so ``find . -exec rm {} \\;`` cannot escape the allowlist.
4. **Sandboxed cwd** — resolved under the shared project root (parent of
   ``results_root``); path traversal rejected. Plus timeout + output cap.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from loguru import logger

from src.config.settings import get_settings
from src.tools._paths import project_root

# Layer 1: commands the agent may invoke. All read-oriented.
_ALLOWED_COMMANDS = frozenset(
    {
        "ls", "cat", "head", "tail", "wc", "grep", "rg", "find", "jq", "tree",
        "curl", "git", "pwd", "echo", "sort", "uniq", "diff", "stat",
        "file", "du", "df",
    }
)

# Layer 3a: git sub-commands — read-only only (no commit/push/checkout/reset).
_GIT_READONLY_SUBCOMMANDS = frozenset(
    {"status", "diff", "log", "show", "branch", "ls-files"}
)
# git branch can mutate with these flags — block them.
_GIT_BRANCH_BLOCKED = frozenset(
    {"-d", "-D", "-m", "-M", "--delete", "--move", "--copy", "--set-upstream"}
)

# Layer 3b: curl flags that set a method or upload/send a body — block them,
# keeping curl GET-only.
_CURL_BLOCKED = frozenset(
    {
        "-X", "--request", "-d", "--data", "--data-raw", "--data-binary",
        "--data-ascii", "--data-urlencode", "-F", "--form", "-T",
        "--upload-file", "-o", "--output", "--post301", "--post302",
        "--post303", "-K", "--config",
    }
)
_CURL_BLOCKED_PREFIXES = ("-X", "-d", "-T", "-F", "-o")

# Layer 3c: find predicates that can execute or write/delete.
_FIND_BLOCKED = frozenset(
    {"-exec", "-execdir", "-ok", "-okdir", "-delete", "-fls", "-fprint",
     "-fprintf"}
)

# Output cap and timeout are operator-configurable via ToolLimitsSettings
# (TERMINAL_MAX_OUTPUT_BYTES / TERMINAL_COMMAND_TIMEOUT). The schema default
# below mirrors the settings default so the LLM-facing description is stable;
# actual enforcement reads settings at call-time via _tool_limits().
_SCHEMA_DEFAULT_TIMEOUT = 30.0  # mirrors ToolLimitsSettings.terminal_command_timeout


def _tool_limits():
    """Call-time accessor — never capture get_settings() at module import."""
    return get_settings().tools


def _resolve_cwd(cwd: Optional[str]) -> tuple[Optional[Path], str]:
    """Resolve cwd under the shared project root. Returns (path, error_or_empty).

    The project root (parent of ``results_root``) is the single cwd every
    file-touching tool runs from; both ``results/`` and the workspace sit beneath
    it, so resolving candidates relative to it accepts them uniformly instead of
    a per-root disjunction. Traversal outside the root is rejected.
    """
    root = project_root()
    if cwd is None:
        return root, ""
    # Path joining: if `cwd` is absolute it replaces; if relative it appends.
    candidate = (root / cwd).resolve()
    if candidate.is_relative_to(root):
        return candidate, ""
    return None, f"ERROR: cwd outside allowed roots: {cwd}"


def _validate_args(command: str, args: list[str]) -> Optional[str]:
    """Layer 3: sub-command allowlists for multi-mode tools. Returns err or None."""
    if command == "git":
        if not args:
            return "ERROR: git requires a sub-command (e.g. status, diff, log)."
        sub = args[0]
        if sub not in _GIT_READONLY_SUBCOMMANDS:
            return (
                f"ERROR: git sub-command '{sub}' not allowed. Read-only only: "
                f"{', '.join(sorted(_GIT_READONLY_SUBCOMMANDS))}."
            )
        if sub == "branch" and any(a in _GIT_BRANCH_BLOCKED for a in args[1:]):
            return "ERROR: git branch mutating flags (-d/-D/-m/…) are blocked."
        return None

    if command == "curl":
        for a in args:
            if a in _CURL_BLOCKED or any(a.startswith(p) for p in _CURL_BLOCKED_PREFIXES):
                return f"ERROR: curl flag '{a}' blocked — curl is GET-only here."
        return None

    if command == "find":
        for a in args:
            if a in _FIND_BLOCKED:
                return f"ERROR: find predicate '{a}' blocked (no exec/delete/write)."
        return None

    return None


async def terminal_command(
    command: str,
    args: Optional[list[str]] = None,
    cwd: Optional[str] = None,
    timeout: Optional[float] = None,
) -> str:
    """Run an allowlisted, shell-free read-only command.

    Args:
        command: One of the allowed utilities (ls, cat, grep, jq, git, …).
        args: List of command arguments. Shell metacharacters are treated
            literally — there is no shell.
        cwd: Directory to run in, resolved under the workspace/results root.
        timeout: Seconds before the process is killed. ``None`` resolves to
            ``TERMINAL_COMMAND_TIMEOUT`` (ToolLimitsSettings, default 30.0).

    Returns:
        Combined stdout (+ stderr if any), capped at the configured byte limit,
        prefixed with the command line; or an ``ERROR:`` string.
    """
    limits = _tool_limits()
    if timeout is None:
        timeout = limits.terminal_command_timeout
    max_output_bytes = limits.terminal_max_output_bytes
    args = list(args or [])
    # Layer 1: command allowlist.
    if command not in _ALLOWED_COMMANDS:
        return (
            f"ERROR: Command '{command}' not allowed. Use one of: "
            f"{', '.join(sorted(_ALLOWED_COMMANDS))}."
        )
    # Layer 3: sub-command allowlists.
    if (err := _validate_args(command, args)):
        return err
    # Layer 4: sandboxed cwd. None return <=> rejected (err holds the message).
    resolved_cwd, err = _resolve_cwd(cwd)
    if resolved_cwd is None:
        return err

    cmd_display = " ".join([command, *args])
    logger.info(f"terminal_command: {cmd_display} (cwd={resolved_cwd})")

    try:
        # Layer 2: list-form exec, shell=False — injection impossible.
        resolved_cwd.mkdir(parents=True, exist_ok=True)
        process = await asyncio.create_subprocess_exec(
            command,
            *args,
            cwd=str(resolved_cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return f"ERROR: Command not found: {command}"
    except OSError as exc:
        return f"ERROR: Cannot execute {command}: {exc}"

    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        try:
            process.kill()
            await process.wait()
        except ProcessLookupError:
            pass
        return f"ERROR: Command timed out after {timeout}s: {cmd_display}"

    return_code = process.returncode
    out = stdout.decode("utf-8", errors="replace")
    err_text = stderr.decode("utf-8", errors="replace")

    combined = out
    if err_text.strip():
        combined += f"\n[stderr]\n{err_text}"
    if len(combined) > max_output_bytes:
        combined = combined[:max_output_bytes] + f"\n... (truncated at {max_output_bytes} bytes)"

    if return_code != 0:
        combined = f"[exit code {return_code}]\n{combined}"
    return combined


TOOL_DEFINITION = {
    "name": "terminal_command",
    "handler": terminal_command,
    "description": (
        "Run an allowlisted, read-only shell command (ls, cat, head, tail, wc, "
        "grep, rg, find, jq, tree, curl, git, pwd, echo, sort, uniq, diff, stat, "
        "file, du, df). Run without a shell — shell metacharacters are literal, "
        "so no pipes/chaining. git is read-only (status/diff/log/show/branch/"
        "ls-files), curl is GET-only, find cannot -exec/-delete. Use this for "
        "things code_executor's sandbox can't do (network via curl, jq on JSON, "
        "git history, ripgrep for fast code search). cwd is sandboxed under the "
        "project directory."
    ),
    # Command output is non-deterministic and side-effecting — never cache.
    "cacheable": False,
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": (
                    "The command to run. One of: ls, cat, head, tail, wc, grep, "
                    "rg, find, jq, tree, curl, git, pwd, echo, sort, uniq, diff, "
                    "stat, file, du, df."
                ),
                "enum": sorted(_ALLOWED_COMMANDS),
            },
            "args": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Arguments as a list (e.g. [\"-la\", \"src\"]). "
                    "Treated literally — no shell expansion."
                ),
            },
            "cwd": {
                "type": "string",
                "description": "Working directory under the project root (default: project root).",
                "default": ".",
            },
            "timeout": {
                "type": "number",
                "description": "Timeout in seconds before the command is killed (default: 30.0, configurable via TERMINAL_COMMAND_TIMEOUT).",
                "default": _SCHEMA_DEFAULT_TIMEOUT,
            },
        },
        "required": ["command"],
    },
}
