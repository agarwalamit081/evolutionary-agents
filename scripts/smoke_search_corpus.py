"""Live smoke for the Phase 1 search stack against running containers.

Exercises the REAL tool code paths (not hand-rolled HTTP) against the
SearXNG + Meilisearch containers this stack brings up:

  1. ``web_search``        — SearXNG primary, single query.
  2. ``web_search`` batch  — ``queries=[...]`` parallel fan-out (one block each).
  3. ``index_corpus``      — index a distinctive probe doc (Meilisearch keyword
                             leg + pgvector cold-memory leg when the DB is up).
  4. ``corpus_search``     — recall the probe doc by a unique token (keyword leg;
                             Reciprocal Rank Fusion over keyword + semantic legs).

Endpoints + key resolve through the app's ``SearchSettings`` (``.env`` /
process env). It never reads or prints secrets. Each step prints PASS/FAIL with
a short evidence snippet; the exit code is non-zero if any *critical* step fails
(steps 1, 3, 4 — the core SearXNG + Meilisearch paths). The pgvector semantic
leg is best-effort: it degrades gracefully when Postgres/migrations are absent
and only affects the hybrid-rank richness, not pass/fail.

The probe doc is **self-cleaning**: step 3 tags it with a marker
(``author:turing-smoke-probe-marker``, plus the always-set ``url:`` tag), and a
``finally`` cleanup deletes every episode/doc carrying those markers from BOTH
stores (pgvector via a parameterized ORM bulk delete, Meilisearch by doc id).
So repeated smoke runs do not pollute the agent's real Meilisearch index or
pgvector cold memory — and the cleanup also reaps any probe episodes left by
smoke runs that predated the marker.

Prerequisite: ``docker compose up -d searxng meilisearch`` (host ports 8081/7701).

Usage::

    python scripts/smoke_search_corpus.py
    SEARXNG_URL=http://localhost:8081 MEILISEARCH_URL=http://localhost:7701 \\
        python scripts/smoke_search_corpus.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

# Make `src` importable when run as `python scripts/smoke_search_corpus.py`
# (sys.path[0] is then scripts/, not the project root).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import get_settings  # noqa: E402
from src.tools.builtin.corpus import corpus_search, index_corpus  # noqa: E402
from src.tools.builtin.web_search import web_search  # noqa: E402

# A deliberately unique token so the corpus recall step is unambiguous. Meilisearch
# tokenizes on word boundaries, so a lowercase alphanumeric run is a clean keyword.
_PROBE_TOKEN = "turingcorpussmokeprobe7q"
_PROBE_TITLE = "Turing Agent Search Stack Smoke Probe"
_PROBE_CONTENT = (
    f"This is a synthetic probe document indexed by scripts/smoke_search_corpus.py "
    f"to verify the Meilisearch keyword leg of corpus_search end-to-end. "
    f"Unique recall token: {_PROBE_TOKEN}. The agent's research corpus stores "
    f"pages it has already read so later runs can recall them without re-scraping."
)

# The probe's identity + cleanup markers. ``index_corpus`` tags every cold-memory
# episode with ``url:<url>`` (always) and ``author:<metadata.author>`` (when set),
# so the cleanup deletes episodes matching EITHER tag — the url tag also catches
# orphan episodes left by smoke runs before the marker was added. The Meilisearch
# doc id is deterministic from the url (``_doc_id``), so one DELETE cleans it.
_PROBE_URL = "https://example.invalid/turing-smoke-probe"
_PROBE_MARKER = "turing-smoke-probe-marker"
_PROBE_URL_TAG = f"url:{_PROBE_URL}"
_PROBE_MARKER_TAG = f"author:{_PROBE_MARKER}"


def _banner(text: str) -> None:
    print(f"\n{'─' * 72}\n{text}\n{'─' * 72}")


def _resolved_endpoints() -> Any:
    s = get_settings().search
    key_shown = "<empty (keyless dev)>" if not s.meilisearch_key else f"<set, {len(s.meilisearch_key)} bytes>"
    print(f"resolved searxng_url    : {s.searxng_url}")
    print(f"resolved meilisearch_url: {s.meilisearch_url}  (key: {key_shown}, index: {s.meilisearch_index})")
    return s


async def _step_web_search_single() -> bool:
    _banner("STEP 1 — web_search (single query, SearXNG primary)")
    out = await web_search(query="postgresql pgvector extension", max_results=3)
    print(out[:600])
    ok = "ERROR" not in out and "No results" not in out
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'} — SearXNG returned results.")
    return ok


async def _step_web_search_batch() -> bool:
    _banner("STEP 2 — web_search (batch, parallel fan-out)")
    out = await web_search(
        queries=["redis streams python", "fastapi background tasks"], max_results=2
    )
    print(out[:700])
    blocks = out.count('Query: "')
    ok = blocks == 2 and "ERROR" not in out
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'} — {blocks} query block(s) returned (expected 2).")
    return ok


async def _step_index_corpus() -> bool:
    _banner("STEP 3 — index_corpus (Meilisearch keyword leg + pgvector leg)")
    out = await index_corpus(
        url=_PROBE_URL,
        content=_PROBE_CONTENT,
        title=_PROBE_TITLE,
        # metadata.author becomes an ``author:`` context-tag on the cold-memory
        # episode — the marker the cleanup deletes by (see _cleanup_probe).
        metadata={"author": _PROBE_MARKER},
    )
    print(out)
    ok = out.lower().startswith("indexed")
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'} — probe doc indexed into the corpus.")
    return ok


async def _step_corpus_search() -> bool:
    _banner("STEP 4 — corpus_search (recall probe doc by unique token, RRF hybrid)")
    # Meilisearch indexes asynchronously; poll briefly until the probe is searchable.
    found = False
    last: str = ""
    for attempt in range(1, 7):
        last = await corpus_search(query=_PROBE_TOKEN, top_k=5)
        if _PROBE_TOKEN in last or _PROBE_TITLE in last:
            found = True
            break
        await asyncio.sleep(1.0)
    print(last[:700])
    print(f"\nRESULT: {'PASS' if found else 'FAIL'} — probe doc recalled after "
          f"{attempt} attempt(s) (keyword leg via Meilisearch; semantic leg via pgvector).")
    return found


async def _cleanup_probe() -> None:
    """Remove the probe doc from both stores so the smoke leaves no trace.

    Surgical: deletes only episodes/docs carrying the probe markers — never the
    agent's real memories. The pgvector leg is self-verifying (counts before and
    after, asserts zero remain); the Meilisearch leg fires a delete by the
    deterministic doc id. Runs in ``main``'s ``finally`` so it always executes,
    even if a step raised or the run is interrupted. Failures are logged, never
    raised — cleanup must not mask the smoke result.
    """
    _banner("CLEANUP — removing the probe doc (Meilisearch + pgvector)")

    # ── pgvector cold memory: ORM bulk-delete episodes carrying a probe tag ──
    # JSONB existence (`?`) returns true when the string is a top-level array
    # element of `context_tags`; OR'd across the two markers so we catch BOTH
    # the author marker tag (new runs) and the always-set url tag (orphan
    # episodes from smoke runs before the marker existed). We use `?` per-tag
    # rather than the `?|` (any-of-array) operator because SQLAlchemy binds a
    # Python list as jsonb and Postgres has no `jsonb ?| jsonb` operator — `?|`
    # needs a `text[]` right operand. Each `?` binds a single text param, so
    # this sidesteps the array-type mismatch entirely. Parameterized ORM — no
    # string-interpolated SQL.
    pg_deleted: int | None = None
    pg_remaining: int | None = None
    try:
        import sqlalchemy as sa
        from src.db.models import ColdMemory as ColdMemoryModel
        from src.db.session import get_session

        tag_pred = sa.or_(
            *(
                ColdMemoryModel.context_tags.bool_op("?")(tag)
                for tag in (_PROBE_URL_TAG, _PROBE_MARKER_TAG)
            )
        )
        async with get_session() as session:
            before = await session.execute(
                sa.select(sa.func.count())
                .select_from(ColdMemoryModel)
                .where(tag_pred)
            )
            count_before = int(before.scalar_one())
            await session.execute(sa.delete(ColdMemoryModel).where(tag_pred))
            await session.commit()
            after = await session.execute(
                sa.select(sa.func.count())
                .select_from(ColdMemoryModel)
                .where(tag_pred)
            )
            count_after = int(after.scalar_one())
            pg_deleted = count_before - count_after
            pg_remaining = count_after
    except Exception as exc:  # cleanup must never mask the smoke result
        print(f"  ⚠ pgvector cleanup skipped: {exc}")

    # ── Meilisearch: delete the probe doc by its deterministic id ──
    meili_ok: bool | None = None
    try:
        import httpx
        from src.tools.builtin.corpus import (
            _doc_id,
            _meili_headers,
            _meili_index,
            _meili_timeout,
            _meili_url,
        )

        doc_id = _doc_id(_PROBE_URL, None, _PROBE_TITLE)
        async with httpx.AsyncClient(timeout=_meili_timeout()) as client:
            resp = await client.delete(
                f"{_meili_url().rstrip('/')}/indexes/{_meili_index()}/documents/{doc_id}",
                headers=_meili_headers(),
            )
            meili_ok = resp.status_code in (200, 202, 204)
            if not meili_ok:
                print(f"  ⚠ meilisearch delete HTTP {resp.status_code}: {resp.text[:120]}")
    except Exception as exc:
        print(f"  ⚠ meilisearch cleanup skipped: {exc}")

    pg_msg = (
        f"{pg_deleted} episode(s) deleted, {pg_remaining} remain"
        if pg_deleted is not None
        else "skipped"
    )
    print(f"  pgvector   : {pg_msg}")
    print(f"  meilisearch: {'doc deleted (task enqueued)' if meili_ok else 'skipped/failed'}")


async def main() -> int:
    _banner("Turing Agent — live search/corpus smoke (Phase 1)")
    _resolved_endpoints()

    results: dict[str, bool] = {}
    try:
        results["web_search_single"] = await _step_web_search_single()
        results["web_search_batch"] = await _step_web_search_batch()
        results["index_corpus"] = await _step_index_corpus()
        results["corpus_search"] = await _step_corpus_search()
    finally:
        # Always remove the probe doc so repeated smoke runs don't pollute the
        # agent's real Meilisearch index / pgvector cold memory.
        await _cleanup_probe()

    _banner("SUMMARY")
    for name, ok in results.items():
        print(f"  {'✅' if ok else '❌'}  {name}")
    # All four are critical: SearXNG (1,2) + Meilisearch index (3) + recall (4).
    all_ok = all(results.values())
    print(f"\n{'✅ SMOKE PASSED' if all_ok else '❌ SMOKE FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
