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
        url="https://example.invalid/turing-smoke-probe",
        content=_PROBE_CONTENT,
        title=_PROBE_TITLE,
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


async def main() -> int:
    _banner("Turing Agent — live search/corpus smoke (Phase 1)")
    _resolved_endpoints()

    results: dict[str, bool] = {}
    results["web_search_single"] = await _step_web_search_single()
    results["web_search_batch"] = await _step_web_search_batch()
    results["index_corpus"] = await _step_index_corpus()
    results["corpus_search"] = await _step_corpus_search()

    _banner("SUMMARY")
    for name, ok in results.items():
        print(f"  {'✅' if ok else '❌'}  {name}")
    # All four are critical: SearXNG (1,2) + Meilisearch index (3) + recall (4).
    all_ok = all(results.values())
    print(f"\n{'✅ SMOKE PASSED' if all_ok else '❌ SMOKE FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
