#!/usr/bin/env python3
"""G0 reset of the agent's run-behavior state — the clean baseline for the
generation-over-generation self-improvement experiment.

The thesis needs a clean G0: a run whose recall surface carries NO prior
run's crystallized capability. That means wiping the channels that leak across
runs — prompt promotions, warm/cold/embedded memory, generated tools/sub-agents,
and Redis hot memory. This script does that, **snapshot-first** (a pg_dump to
``logs/snapshots/``), idempotently, and reversibly (``--restore``).

**Reset is DEACTIVATION, not deletion, for capability rows** (tools/sub-agents):
``is_active=false`` removes them from recall while keeping the rows for audit +
``--restore``. Memory rows (warm/cold/embeddings) ARE deleted (they are the
state itself); the snapshot is their restore path.

Discriminators (grounded against the live DB — the plan's ``source_mutation_id``
was wrong: 0 rows carry it):
- tools: ``tool_type='generated'`` (all persisted tools are generated; builtins
  are code-loaded, never in this table)
- sub-agents: ``template_type='custom'`` (the ``fixed`` 111 are builtin roles)

``results/`` clearing is a SEPARATE explicit scope (not in ``all``): the C2 DAG
input-pins a generation's goals to each other, so blanket-clearing deliverables
mid-experiment would orphan downstream inputs. Use ``--scope all,results`` only
for a full pre-experiment wipe.

DANGER: this is destructive. Always runs snapshot-first unless ``--no-snapshot``
or ``--dry-run``. The scheduler MUST be stopped and no runs in flight.

Usage::

    python scripts/clean_state.py --dry-run                     # show reset set + counts
    python scripts/clean_state.py --scope all                    # the G0 reset (snapshot first)
    python scripts/clean_state.py --scope warm,cold              # subset
    python scripts/clean_state.py --scope all,results            # + deliverables
    python scripts/clean_state.py --restore logs/snapshots/x.sql # undo from a snapshot
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Project root on sys.path so ``src.*`` imports resolve when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger  # noqa: E402
from sqlalchemy import delete, func, select, update  # noqa: E402

from src.db.models import (  # noqa: E402
    ColdMemory,
    MemoryEmbedding,
    SubAgentModel,
    ToolRegistration,
    WarmMemory,
)
from src.db.session import get_session  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SNAPSHOT_DIR = _PROJECT_ROOT / "logs" / "snapshots"
_PROMPT_DIR = _PROJECT_ROOT / ".turing" / "evolved" / "prompts"
_RESULTS_DIR = _PROJECT_ROOT / "results"


# ─── canonical reset set (pure; the unit test asserts this exactly) ───────────


@dataclass(frozen=True, slots=True)
class ResetChannel:
    """One leakable channel and how to reset it."""

    name: str  # short id used by --scope
    label: str  # human description
    kind: str  # db_delete | db_update | file_glob | file_tree | redis


# Order matters only for readability (memory → capability → infra).
RESET_CHANNELS: list[ResetChannel] = [
    ResetChannel(
        "prompts",
        ".turing/evolved/prompts/* (live prompt promotions + current.json)",
        "file_glob",
    ),
    ResetChannel(
        "warm", "warm_memories (skills/procedures/facts/folded)", "db_delete"
    ),
    ResetChannel(
        "embeddings", "memory_embeddings (recall vectors)", "db_delete"
    ),
    ResetChannel(
        "cold", "cold_memories (episodes)", "db_delete"
    ),
    ResetChannel(
        "tools",
        "tool_registrations WHERE tool_type='generated' (is_active=false)",
        "db_update",
    ),
    ResetChannel(
        "subagents",
        "sub_agent_definitions WHERE template_type='custom' (is_active=false)",
        "db_update",
    ),
    ResetChannel(
        "redis", "FLUSHDB (hot memory + run queue)", "redis"
    ),
    ResetChannel(
        "results", "results/<run_id>/ subdirs (cross-run deliverables)", "file_tree"
    ),
]

# 'all' excludes 'results' (DAG input-pinning — see module docstring).
DEFAULT_SCOPE: list[str] = [
    "prompts",
    "warm",
    "embeddings",
    "cold",
    "tools",
    "subagents",
    "redis",
]
_ALL_NAMES = {c.name for c in RESET_CHANNELS}


def select_channels(scope: str) -> list[ResetChannel]:
    """Resolve a --scope string (comma-list, or 'all') → ordered ResetChannels.

    ``all`` expands to DEFAULT_SCOPE (results excluded). Unknown names raise
    ValueError. Order follows RESET_CHANNELS so output is stable.
    """
    tokens = [t.strip() for t in scope.split(",") if t.strip()]
    if not tokens:
        raise ValueError("empty scope")
    wanted: set[str] = set()
    for t in tokens:
        if t == "all":
            wanted.update(DEFAULT_SCOPE)  # 'all' expands inline, combinable with others
        elif t in _ALL_NAMES:
            wanted.add(t)
        else:
            raise ValueError(
                f"unknown channel {t!r}; valid: {sorted(_ALL_NAMES)} (+ 'all')"
            )
    return [c for c in RESET_CHANNELS if c.name in wanted]


# ─── model + discriminator maps for the DB ops ───────────────────────────────

_DB_DELETE_MODELS = {
    "warm": WarmMemory,
    "embeddings": MemoryEmbedding,
    "cold": ColdMemory,
}


async def _apply_db_delete(session: object, channel: ResetChannel) -> int:
    model = _DB_DELETE_MODELS[channel.name]
    result = await session.execute(delete(model))  # type: ignore[arg-type]
    return int(result.rowcount or 0)


async def _count_db_delete(session: object, channel: ResetChannel) -> int:
    model = _DB_DELETE_MODELS[channel.name]
    return int(
        await session.scalar(select(func.count()).select_from(model)) or 0  # type: ignore[arg-type]
    )


async def _apply_generated_inactive(
    session: object, channel: ResetChannel
) -> int:
    """Deactivate agent-created tools (tool_type='generated') or sub-agents
    (template_type='custom'); return how many rows flipped to inactive."""
    if channel.name == "tools":
        result = await session.execute(  # type: ignore[arg-type]
            update(ToolRegistration)
            .where(
                ToolRegistration.tool_type == "generated",
                ToolRegistration.is_active.is_(True),
            )
            .values(is_active=False)
        )
    else:  # subagents
        result = await session.execute(  # type: ignore[arg-type]
            update(SubAgentModel)
            .where(
                SubAgentModel.template_type == "custom",
                SubAgentModel.is_active.is_(True),
            )
            .values(is_active=False)
        )
    return int(result.rowcount or 0)


async def _count_generated_active(
    session: object, channel: ResetChannel
) -> int:
    if channel.name == "tools":
        stmt = select(func.count()).select_from(ToolRegistration).where(
            ToolRegistration.tool_type == "generated",
            ToolRegistration.is_active.is_(True),
        )
    else:
        stmt = select(func.count()).select_from(SubAgentModel).where(
            SubAgentModel.template_type == "custom",
            SubAgentModel.is_active.is_(True),
        )
    return int(await session.scalar(stmt) or 0)  # type: ignore[arg-type]


# ─── file + redis ops ────────────────────────────────────────────────────────


def _clear_prompt_dir() -> int:
    """Unlink every file under .turing/evolved/prompts/ (keep the dir). 0 if absent."""
    if not _PROMPT_DIR.exists():
        return 0
    n = 0
    for p in _PROMPT_DIR.iterdir():
        if p.is_file():
            p.unlink()
            n += 1
    return n


def _count_prompt_dir() -> int:
    if not _PROMPT_DIR.exists():
        return 0
    return sum(1 for p in _PROMPT_DIR.iterdir() if p.is_file())


def _clear_results_subdirs() -> int:
    """rmtree every results/<run_id>/ subdir (keep the results/ root)."""
    if not _RESULTS_DIR.exists():
        return 0
    n = 0
    for p in _RESULTS_DIR.iterdir():
        if p.is_dir():
            shutil.rmtree(p)
            n += 1
    return n


def _count_results_subdirs() -> int:
    if not _RESULTS_DIR.exists():
        return 0
    return sum(1 for p in _RESULTS_DIR.iterdir() if p.is_dir())


def _redis_flushdb() -> str:
    """FLUSHDB the turing Redis DB via the compose container. Returns stdout."""
    proc = subprocess.run(
        ["docker", "compose", "exec", "-T", "redis", "redis-cli", "FLUSHDB"],
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"redis FLUSHDB failed: {proc.stderr.strip() or proc.stdout.strip()}")
    return proc.stdout.strip() or "OK"


# ─── snapshot / restore ──────────────────────────────────────────────────────


def snapshot_db() -> Path:
    """pg_dump the whole turing_agent DB (clean+if-exists) → logs/snapshots/.

    The restore path for every reset channel (memory rows are deleted, not just
    deactivated). Full-DB dump avoids partial-restore FK hazards.
    """
    _SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = _SNAPSHOT_DIR / f"clean_state-{ts}.sql"
    with path.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(
            [
                "docker", "compose", "exec", "-T", "postgres",
                "pg_dump", "--clean", "--if-exists", "-U", "postgres", "turing_agent",
            ],
            stdout=fh,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(_PROJECT_ROOT),
            check=False,
        )
    if proc.returncode != 0:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"pg_dump failed: {proc.stderr.strip()}")
    logger.info("snapshot written → {}", path)
    return path


def restore_db(path: Path) -> None:
    """Replay a snapshot (psql -f) — a FULL restore to the snapshot state."""
    if not path.exists():
        raise FileNotFoundError(path)
    proc = subprocess.run(
        [
            "docker", "compose", "exec", "-T", "postgres",
            "psql", "-U", "postgres", "-d", "turing_agent",
        ],
        stdin=path.open("r", encoding="utf-8"),
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"psql restore failed: {proc.stderr.strip()[:500]}")
    logger.info("restored DB from → {}", path)


# ─── orchestration ───────────────────────────────────────────────────────────


async def _count_for(session: object, channel: ResetChannel) -> int | str:
    if channel.kind == "db_delete":
        return await _count_db_delete(session, channel)
    if channel.kind == "db_update":
        return await _count_generated_active(session, channel)
    if channel.name == "prompts":
        return _count_prompt_dir()
    if channel.name == "results":
        return _count_results_subdirs()
    if channel.kind == "redis":
        return "all keys"
    return 0


async def _apply_one(session: object, channel: ResetChannel) -> int | str:
    if channel.kind == "db_delete":
        return await _apply_db_delete(session, channel)
    if channel.kind == "db_update":
        return await _apply_generated_inactive(session, channel)
    if channel.name == "prompts":
        return _clear_prompt_dir()
    if channel.name == "results":
        return _clear_results_subdirs()
    if channel.kind == "redis":
        return _redis_flushdb()
    return 0


async def run_reset(
    scope: str, *, dry_run: bool, snapshot: bool
) -> dict[str, object]:
    """Resolve channels, (optionally) snapshot, then count-or-apply each.

    Returns a per-channel result dict ``{name: {"count": ..., "applied": bool}}``.
    Dry-run touches nothing and never snapshots.
    """
    channels = select_channels(scope)
    print(
        f"\n═ clean_state · scope={scope!r} · dry_run={dry_run} · "
        f"{len(channels)} channel(s)"
    )

    snap_path: Path | None = None
    if not dry_run and snapshot:
        snap_path = snapshot_db()
        print(f"  snapshot → {snap_path}")

    results: dict[str, object] = {}
    async with get_session() as session:
        for c in channels:
            count = await _count_for(session, c)
            if dry_run:
                verb = "WOULD RESET"
                print(f"  [{verb}] {c.name:<10} n={count:<6} {c.label}")
                results[c.name] = {"count": count, "applied": False}
            else:
                n = await _apply_one(session, c)
                await session.commit()
                print(f"  [RESET]     {c.name:<10} n={n:<6} {c.label}")
                results[c.name] = {"count": n, "applied": True}
    if dry_run:
        print("\n  (dry-run: nothing was changed)")
    else:
        print("\n  done. restore with --restore <snapshot>")
    return {"channels": results, "snapshot": str(snap_path) if snap_path else None}


# ─── CLI ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--scope",
        default="all",
        help="comma-list of channels, or 'all' (default). 'all' excludes 'results'.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="show the exact reset set + counts, change nothing"
    )
    parser.add_argument(
        "--no-snapshot", action="store_true", help="skip the pg_dump snapshot (DANGEROUS)"
    )
    parser.add_argument(
        "--restore", metavar="SNAPSHOT", help="restore the DB from a snapshot .sql and exit"
    )
    args = parser.parse_args()

    if args.restore:
        restore_db(Path(args.restore))
        return

    # Validate scope before doing anything destructive.
    try:
        select_channels(args.scope)
    except ValueError as exc:
        parser.error(str(exc))

    asyncio.run(
        run_reset(args.scope, dry_run=args.dry_run, snapshot=not args.no_snapshot)
    )


if __name__ == "__main__":
    main()
