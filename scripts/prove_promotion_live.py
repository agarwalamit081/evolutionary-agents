"""Live proof that a real deployed PROMPT mutation writes the live prompt file.

This deterministically demonstrates the evolution→live promotion path
(``src/evolution/promote.py``) end-to-end using the **actual** mutation a real
agent run deployed — closing the gap that the shape-mismatch bug left open.

Background (battery-04 q08): run4 reached the evolve node and deployed mutation
``4c5c11d4`` (a free-text rewrite of ``prompts/system_prompt.md``). The promotion
gate then silently no-op'd because ``parse_prompt_payload`` required a JSON
``{"target_node","suffixes"}`` payload and rejected the free-text shape — so O2
never fired on a real run. The fix (commit 7d27626) makes ``parse_prompt_payload``
accept the free-text whole-file rewrite as one promoted suffix.

This script replays that **real** mutation through the **fixed** gate so we can
confirm, deterministically, that:

1. ``parse_prompt_payload`` now accepts the live free-text shape (was ``None``).
2. ``PromotionGate.promote()`` writes a versioned artifact + ``current.json``
   pointer when the canary passes — exactly the live write O2 needs.
3. The builder read-back (``gate.current_suffixes``) surfaces the promoted
   suffix verbatim.
4. A failing/inconclusive canary leaves NO pointer (the safe no-op path).

The canary is a deterministic fixed-score stub (no LLM) so the proof is
reproducible and ~free. It writes to an ephemeral temp dir so it cannot race a
concurrent live run writing the real ``.turing/evolved/prompts/`` pointer.

Usage::

    python scripts/prove_promotion_live.py                 # latest deployed PROMPT mutation
    python scripts/prove_promotion_live.py 4c5c11d4        # by id prefix
    python scripts/prove_promotion_live.py --keep          # keep the temp dir for inspection

No secrets are read or printed; connects via the app's SQLAlchemy engine.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import Text, cast, select

from src.db.models import Mutation
from src.db.session import get_session
from src.evolution.promote import PromotionGate, parse_prompt_payload
from src.graph.enums import MutationType

_PASSING_CANARY_SCORE = 0.9
_FAILING_CANARY_SCORE = 0.5
_MIN_SCORE = 0.8


async def _fetch_deployed_prompt(ident: str | None) -> Any:
    """Latest deployed PROMPT mutation (or one matching ``ident`` prefix/uuid)."""
    base = select(Mutation).where(Mutation.mutation_type == MutationType.PROMPT)
    if ident:
        if "-" in ident:
            stmt = base.where(Mutation.id == uuid.UUID(ident))
        else:
            stmt = base.where(cast(Mutation.id, Text).like(f"{ident}%"))
    else:
        stmt = base.where(Mutation.status == "deployed")
    stmt = stmt.order_by(Mutation.created_at.desc())
    async with get_session() as session:
        result = await session.execute(stmt)
        return result.scalars().first()


def _to_proposal(row: Any) -> dict[str, Any]:
    """Rebuild the proposal dict in the exact shape ``engine.run_cycle`` passes
    to ``promotion_gate.promote()`` (src/evolution/engine.py:~396-403)."""
    return {
        "mutation_type": row.mutation_type,
        "description": row.description,
        "original_content": row.original_content,
        "mutated_content": row.mutated_content,
        "target_path": row.target_path,
        "rationale": row.description,  # DB row has no rationale; description stands in
        "model_used": row.model_used,
    }


async def _passing_canary(_node: str, _suffixes: list[str]) -> float | None:
    return _PASSING_CANARY_SCORE


async def _failing_canary(_node: str, _suffixes: list[str]) -> float | None:
    return _FAILING_CANARY_SCORE


def _banner(title: str) -> None:
    print(f"\n=== {title} ===")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("id", nargs="?", help="mutation id (full UUID or short prefix)")
    parser.add_argument(
        "--keep", action="store_true", help="keep the ephemeral proof dir for inspection"
    )
    args = parser.parse_args()

    row = await _fetch_deployed_prompt(args.id)
    if row is None:
        print(
            f"No deployed PROMPT mutation found for id={args.id!r}", file=sys.stderr
        )
        return 1

    content = row.mutated_content or ""
    print(f"mutation id    : {row.id}")
    print(f"target_path    : {row.target_path}")
    print(f"status         : {row.status}")
    print(f"model_used     : {row.model_used}")
    print(f"content_len    : {len(content)} chars (free-text rewrite)")

    proposal = _to_proposal(row)
    failures: list[str] = []

    # ---- Step 1: the parser now accepts the live free-text shape ----------------
    _banner("1. parse_prompt_payload (the fix under test)")
    parsed = parse_prompt_payload(proposal)
    print(f"parsed = {None if parsed is None else (parsed[0], f'<{len(parsed[1])} suffix(es)>')}")
    if parsed is None:
        failures.append("parse_prompt_payload returned None for a real PROMPT mutation (regression)")
        print("FAIL: parser rejected the real free-text mutation — O2 cannot fire.")
    else:
        node, suffixes = parsed
        print(f"PASS: parser → node='{node}', {len(suffixes)} suffix(es) promoted")
        if len(suffixes) == 1 and suffixes[0] == content.strip():
            print("     (entire free-text block treated as ONE promoted suffix, as designed)")

    # ---- Step 2: promote() writes the live pointer on a passing canary ---------
    _banner("2. promote() with PASSING canary → live write")
    tmp_root = tempfile.mkdtemp(prefix="turing_promote_")
    tmp_dir = Path(tmp_root)
    gate = PromotionGate(
        handlers_dir=tmp_dir, canary=_passing_canary, min_score=_MIN_SCORE
    )
    result = await gate.promote(proposal)
    print(f"promote() result: {json.dumps(result, indent=2)}")
    if not result.get("promoted"):
        failures.append("promote() did not promote on a passing canary")
    else:
        version_name = result["version"]
        version_file = gate.prompts_dir / version_name
        pointer_file = gate.prompts_dir / "current.json"
        if not version_file.exists():
            failures.append(f"versioned artifact {version_name} not written")
        if not pointer_file.exists():
            failures.append("current.json pointer not written")
        read_back = gate.current_suffixes(result["node"])
        print(f"versioned artifact : {'written' if version_file.exists() else 'MISSING'} ({version_name})")
        print(f"current.json       : {'written' if pointer_file.exists() else 'MISSING'}")
        print(f"read-back suffixes : {len(read_back)} entry/entries")
        if parsed is not None and read_back != suffixes:
            failures.append(f"read-back mismatch: {read_back!r} != {suffixes!r}")
        else:
            print("PASS: live pointer written + read-back matches the promoted suffix")

    # ---- Step 3: a failing canary writes NOTHING (safe no-op) ------------------
    _banner("3. promote() with FAILING canary → no write (safe)")
    tmp_dir2 = Path(tempfile.mkdtemp(prefix="turing_promote_failing_"))
    gate2 = PromotionGate(
        handlers_dir=tmp_dir2, canary=_failing_canary, min_score=_MIN_SCORE
    )
    result2 = await gate2.promote(proposal)
    pointer2 = gate2.prompts_dir / "current.json"
    print(f"promote() result: promoted={result2.get('promoted')}, reason={result2.get('reason')!r}")
    if pointer2.exists():
        failures.append("failing canary wrote a pointer (should be a safe no-op)")
    else:
        print("PASS: failing canary left no pointer (prior state untouched)")

    # ---- Verdict ----------------------------------------------------------------
    _banner("VERDICT")
    if failures:
        print("❌ O2 live-write proof FAILED:")
        for f in failures:
            print(f"   - {f}")
        return 2
    print("✅ O2 live-write path PROVEN: real deployed mutation → fixed parse →")
    print("   canary-gated promote() → versioned artifact + current.json → read-back.")
    if args.keep:
        print(f"\nProof dir retained for inspection: {tmp_dir}")
    else:
        # Clean the ephemeral dirs; the proof is the printed verdict.
        import shutil

        shutil.rmtree(tmp_root, ignore_errors=True)
        shutil.rmtree(str(tmp_dir2), ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
