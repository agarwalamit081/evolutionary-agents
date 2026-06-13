"""Tool inspecting the runtime environment: OS, CPU, disk, RAM, packages.

Lets the agent avoid generating code that requires a missing dependency, and
reason about resource constraints before running heavy computations. Uses only
the standard library (no ``psutil`` dependency).
"""

from __future__ import annotations

import asyncio
import importlib.metadata as importlib_metadata
import os
import platform
import shutil
from pathlib import Path

from loguru import logger


_MAX_PACKAGE_PREVIEW = 200


def _read_meminfo() -> str:
    """Best-effort total-RAM summary from ``/proc/meminfo`` (Linux only)."""
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                kib = int(line.split()[1])
                return f"{kib / 1024:.0f} MB"
    except (OSError, ValueError, IndexError):
        pass
    return "unknown (not Linux or /proc unavailable)"


def _summary() -> str:
    """OS, Python, CPU, RAM, and free disk for the current working directory."""
    disk = shutil.disk_usage(os.getcwd())
    return (
        f"OS: {platform.platform()}\n"
        f"Python: {platform.python_version()}\n"
        f"CPU cores: {os.cpu_count()}\n"
        f"RAM (total): {_read_meminfo()}\n"
        f"Disk (cwd): {disk.free / (1024 ** 3):.1f} GB free of "
        f"{disk.total / (1024 ** 3):.1f} GB"
    )


def _packages() -> str:
    """Installed Python distributions (sorted, capped preview)."""
    dists = sorted(
        (d.metadata["Name"] for d in importlib_metadata.distributions()),
        key=str.lower,
    )
    if not dists:
        return "No packages detected."
    preview = "\n".join(f"- {d}" for d in dists[:_MAX_PACKAGE_PREVIEW])
    suffix = (
        f"\n... ({len(dists)} packages total, showing first {_MAX_PACKAGE_PREVIEW})"
        if len(dists) > _MAX_PACKAGE_PREVIEW
        else ""
    )
    return f"Installed packages ({len(dists)}):\n{preview}{suffix}"


async def environment_inspect(detail: str = "summary") -> str:
    """Inspect the runtime environment.

    Args:
        detail: ``"summary"`` (OS, CPU, RAM, free disk) or ``"packages"``
            (installed Python distributions). Any other value falls back to
            ``"summary"``. Defaults to ``"summary"``.

    Returns:
        Human-readable environment information.
    """
    logger.info(f"environment_inspect: detail={detail}")
    if detail.lower() == "packages":
        return await asyncio.to_thread(_packages)
    return await asyncio.to_thread(_summary)


TOOL_DEFINITION = {
    "name": "environment_inspect",
    "handler": environment_inspect,
    "description": (
        "Inspect the runtime environment. With detail='summary' (default) "
        "returns OS, Python version, CPU cores, total RAM, and free disk. "
        "With detail='packages' returns the list of installed Python "
        "distributions. Use 'packages' BEFORE writing code that imports a "
        "third-party library, to avoid failing on a missing dependency."
    ),
    # Package set / resource usage change mid-run — never cache.
    "cacheable": False,
    "parameters": {
        "type": "object",
        "properties": {
            "detail": {
                "type": "string",
                "description": "'summary' (OS/CPU/RAM/disk) or 'packages' (installed libs).",
                "default": "summary",
                "enum": ["summary", "packages"],
            },
        },
        "required": [],
    },
}
