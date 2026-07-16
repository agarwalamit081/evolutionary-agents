"""Corpus tools — index gathered web research and search it (Phase 1).

Distinct from ``web_search`` (live web) and ``web_scraper`` (one page):
``index_corpus`` persists scraped pages into a *local, durable* hybrid store
and ``corpus_search`` runs a hybrid query over it. This is the agent's
"research memory" — a growing corpus of pages it has already read, so later
runs can recall them without re-scraping.

Dual-write store:
  * **Meilisearch** (service port 7700; host-mapped to 7701) — typo-tolerant
    BM25 *keyword* index, reached via its REST API over ``httpx`` (no
    ``meilisearch`` SDK dependency). Configured with
    ``distinctAttribute = content_hash`` so the same content re-indexed from
    different URLs collapses to one hit.
  * **pgvector cold memory** — *semantic* index. Reuses the existing
    ``ColdMemory``/``EmbeddingGenerator`` (litellm embeddings + hash fallback).

Search fuses the two legs with **Reciprocal Rank Fusion** (RRF; k from
``SearchSettings.corpus_rrf_k``, default 60), the
same algorithm as ``web-search/.../search_service.perform_hybrid_search``.
Batch/parallel: ``index_corpus(documents=[...])`` and
``corpus_search(queries=[...])`` fan out under ``asyncio.gather`` +
``Semaphore(SEARCH_BATCH_CONCURRENCY)``; one bad document/query never sinks
the batch.

Graceful degradation (mirrors the project's heuristic-fallback ethos): if
Meilisearch is unreachable the keyword leg is empty; if the DB/embedder is
unavailable the semantic leg is empty. A fully-offline search returns a clear
"0 results" message rather than raising.
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any

import httpx
from loguru import logger
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from src.config.settings import get_settings


# ── settings accessors (call-time; never capture get_settings() at import) ──


def _search_settings() -> Any:
    """Call-time accessor for the SearchSettings group."""
    return get_settings().search


def _meili_url() -> str:
    return str(_search_settings().meilisearch_url).rstrip("/")


def _meili_index() -> str:
    return str(_search_settings().meilisearch_index)


def _meili_timeout() -> float:
    return float(_search_settings().meilisearch_timeout)


def _meili_task_poll_interval() -> float:
    return float(_search_settings().meilisearch_task_poll_interval)


def _meili_task_max_polls() -> int:
    return max(1, int(_search_settings().meilisearch_task_max_polls))


def _meili_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    key = _search_settings().meilisearch_key
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _batch_concurrency() -> int:
    return max(1, int(_search_settings().search_batch_concurrency))


# ── errors ──────────────────────────────────────────────────────────────


class CorpusError(Exception):
    """Base class for corpus-store failures (non-fatal at the tool surface)."""


class TransientCorpusError(CorpusError):
    """Transient Meilisearch failure (429/5xx/timeout/network) — retried."""


class CorpusUnavailable(CorpusError):
    """Non-transient Meilisearch failure (4xx/config) — leg degrades to empty."""


# ── Meilisearch REST client (httpx; transient retry on 429/5xx/timeout) ──


async def _meili_request(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    *,
    json_body: Any = None,
    params: dict[str, str] | None = None,
) -> dict[str, Any]:
    """One Meilisearch REST call with transient retry; returns parsed JSON ({})."""
    url = f"{_meili_url()}{path}"
    headers = _meili_headers()
    timeout = _meili_timeout()
    limits = get_settings().tools
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(limits.corpus_search_max_attempts),
        wait=wait_exponential_jitter(
            initial=limits.corpus_retry_initial_delay,
            max=limits.corpus_retry_max_delay,
        ),
        retry=retry_if_exception_type(TransientCorpusError),
        reraise=True,
    ):
        with attempt:
            try:
                resp = await client.request(
                    method, url, json=json_body, params=params, headers=headers, timeout=timeout
                )
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                raise TransientCorpusError(f"meilisearch transport error: {exc}") from exc
            if resp.status_code == 429 or resp.status_code >= 500:
                raise TransientCorpusError(f"meilisearch {resp.status_code}")
            if 400 <= resp.status_code < 500:
                raise CorpusUnavailable(
                    f"meilisearch {resp.status_code}: {resp.text[:200]}"
                )
            try:
                data = resp.json()
            except ValueError:
                data = {}
            return data if isinstance(data, dict) else {}
    # Unreachable: AsyncRetrying(reraise=True) returns or re-raises above.
    raise CorpusError("meilisearch request exhausted retries")


_INDEX_READY = False  # one best-effort index-config attempt per process


async def _ensure_index(client: httpx.AsyncClient) -> None:
    """Best-effort create-if-absent + dedup config. Non-fatal; never retries forever."""
    global _INDEX_READY
    if _INDEX_READY:
        return
    idx = _meili_index()
    try:
        try:
            # Create the index (409 "already exists" is fine — ignore).
            await _meili_request(
                client, "POST", "/indexes", json_body={"uid": idx, "primaryKey": "id"}
            )
        except CorpusUnavailable:
            pass
        # distinctAttribute collapses re-indexed identical content to one hit.
        await _meili_request(
            client,
            "PATCH",
            f"/indexes/{idx}/settings",
            json_body={
                "searchableAttributes": ["title", "content"],
                "distinctAttribute": "content_hash",
                "filterableAttributes": ["url", "domain"],
            },
        )
        logger.debug(f"corpus meilisearch index ready: {idx}")
    except CorpusError as exc:
        # Don't abort indexing/search over a config hiccup — degrade silently.
        logger.debug(f"meilisearch index init skipped: {exc}")
    finally:
        _INDEX_READY = True


def _doc_id(url: str | None, content_hash: str | None, title: str | None) -> str:
    seed = url or content_hash or title or ""
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]


def _meili_doc(
    url: str | None,
    title: str | None,
    content: str | None,
    content_hash: str | None,
    metadata: dict[str, str] | None,
) -> dict[str, str]:
    return {
        "id": _doc_id(url, content_hash, title),
        "url": url or "",
        "title": title or "",
        "content": (content or "")[:10000],
        "content_hash": content_hash or _doc_id(None, None, content),
        "domain": (metadata or {}).get("hostname", ""),
    }


async def _meili_wait_for_task(client: httpx.AsyncClient, task_uid: str) -> None:
    """Poll GET /tasks/{uid} until the Meilisearch task is terminal.

    Indexing is async: POST /documents returns a taskUid and the actual index
    update runs in the background. Without this wait, a _meili_search in the
    same coroutine races the still-enqueued task and sees no docs. Bounded by
    MEILISEARCH_TASK_MAX_POLLS x MEILISEARCH_TASK_POLL_INTERVAL — on exhaustion
    we return best-effort (the index leg never hangs forever; eventual
    consistency + a later search covers a slow task). A failed task is logged
    and treated as a soft miss (one bad doc must not abort the whole batch).
    """
    path = f"/tasks/{task_uid}"
    interval = _meili_task_poll_interval()
    max_polls = _meili_task_max_polls()
    for _ in range(max_polls):
        try:
            task = await _meili_request(client, "GET", path)
        except CorpusError as exc:
            logger.warning(f"corpus: task {task_uid} status poll failed: {exc}")
            return
        status = str(task.get("status", "")).lower()
        if status == "succeeded":
            return
        if status == "failed":
            logger.warning(f"corpus: meilisearch task {task_uid} failed: {task.get('error')}")
            return
        await asyncio.sleep(interval)
    logger.warning(
        f"corpus: meilisearch task {task_uid} did not finish in "
        f"{max_polls * interval:.1f}s; proceeding (search may lag)"
    )


async def _meili_add_documents(
    client: httpx.AsyncClient, docs: list[dict[str, str]]
) -> str | None:
    """Index docs into Meilisearch, then await the task so a same-coroutine
    search sees them. Returns the task uid (None when there is nothing to index)."""
    if not docs:
        return None
    await _ensure_index(client)
    resp = await _meili_request(
        client,
        "POST",
        f"/indexes/{_meili_index()}/documents",
        json_body=docs,
        params={"primaryKey": "id"},
    )
    task = resp.get("taskUid")
    task_uid = str(task) if task is not None else None
    if task_uid is not None:
        await _meili_wait_for_task(client, task_uid)
    return task_uid


async def _meili_search(
    client: httpx.AsyncClient, query: str, limit: int
) -> list[dict[str, Any]]:
    """Keyword (BM25) search over the corpus; [] on any failure."""
    try:
        resp = await _meili_request(
            client,
            "POST",
            f"/indexes/{_meili_index()}/search",
            json_body={"q": query, "limit": limit},
        )
    except CorpusError as exc:
        logger.debug(f"corpus keyword leg unavailable: {exc}")
        return []
    hits = resp.get("hits", [])
    return hits if isinstance(hits, list) else []


# ── cold memory (pgvector semantic) — own session per operation ─────────


async def _cold_semantic_search(query: str, limit: int) -> list[dict[str, Any]]:
    """Semantic search over cold memory; [] if DB/embedder unavailable."""
    try:
        from src.db.session import get_session
        from src.memory.cold import ColdMemory
        from src.memory.embeddings import EmbeddingGenerator
        from src.tools.builtin.web_scraper import compute_content_hash

        generator = EmbeddingGenerator(get_settings())
        async with get_session() as session:
            memory = ColdMemory(session, generator=generator)
            rows = await memory.search_by_query(query, limit=limit)
    except Exception as exc:  # embedding/db failure must not break the tool
        logger.debug(f"corpus semantic leg unavailable: {exc}")
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        content = str(row.get("content") or "")
        out.append(
            {
                "url": "",
                "title": "",
                "content": content[:500],
                "content_hash": compute_content_hash(content),
                "similarity": row.get("similarity"),
                "_source": "semantic",
            }
        )
    return out


async def _cold_store(
    url: str | None,
    title: str | None,
    content: str,
    content_hash: str | None,
    metadata: dict[str, str] | None,
) -> str | None:
    """Persist one page as a cold-memory episode; None if DB/embedder unavailable."""
    try:
        from src.db.session import get_session
        from src.memory.cold import ColdMemory
        from src.memory.embeddings import EmbeddingGenerator

        generator = EmbeddingGenerator(get_settings())
        async with get_session() as session:
            memory = ColdMemory(session, generator=generator)
            tags = [f"url:{url}"] if url else []
            if content_hash:
                tags.append(f"hash:{content_hash}")
            for key in ("hostname", "author", "date"):
                val = (metadata or {}).get(key)
                if val:
                    tags.append(f"{key}:{val}")
            body = f"{title}\n\n{content}" if title else content
            memory_id = await memory.store(
                episode_type="learning",
                content=body,
                importance=0.6,
                context_tags=tags,
            )
            return memory_id
    except Exception as exc:  # cold write is non-critical
        logger.debug(f"corpus cold store unavailable: {exc}")
        return None


# ── Reciprocal Rank Fusion (ported from web-search search_service) ──────


def _fusion_key(hit: dict[str, Any], fallback: str) -> str:
    """Dedup key: url → content_hash → id → caller-supplied fallback."""
    return (
        hit.get("url")
        or hit.get("content_hash")
        or hit.get("id")
        or hit.get("_id")
        or fallback
    )


def rrf_fuse(
    keyword_hits: list[dict[str, Any]],
    semantic_hits: list[dict[str, Any]],
    k: int = 60,
    final_limit: int = 10,
) -> list[dict[str, Any]]:
    """Merge keyword + semantic result lists via Reciprocal Rank Fusion.

    A hit present in *both* legs (same ``url``/``content_hash``) is merged into
    one entry with a boosted score; hits in only one leg score lower. Returns
    the top ``final_limit`` by fused score, each annotated with
    ``keyword_rank``/``semantic_rank``/``rrf_score``/``_source``.
    """
    scores: dict[str, float] = {}
    details: dict[str, dict[str, Any]] = {}

    def add_rank(hit: dict[str, Any], rank: int, *, source: str) -> None:
        key = _fusion_key(hit, fallback=f"{source}-{rank}")
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        if key in details:
            if source == "semantic":
                details[key]["semantic_rank"] = rank + 1
            else:
                details[key]["keyword_rank"] = rank + 1
            return
        entry = dict(hit)
        entry.update(
            {
                "keyword_rank": (rank + 1) if source == "keyword" else None,
                "semantic_rank": (rank + 1) if source == "semantic" else None,
                "rrf_score": 0.0,
                "_source": hit.get("_source", source),
            }
        )
        details[key] = entry

    for rank, hit in enumerate(keyword_hits):
        add_rank(hit, rank, source="keyword")
    for rank, hit in enumerate(semantic_hits):
        add_rank(hit, rank, source="semantic")

    for entry in details.values():
        entry["rrf_score"] = round(scores.get(_fusion_key(entry, ""), 0.0), 6)

    ranked = sorted(details.values(), key=lambda x: x["rrf_score"], reverse=True)
    return ranked[:final_limit]


def _format_hits(query: str, hits: list[dict[str, Any]]) -> str:
    if not hits:
        return f"Corpus search for '{query[:60]}' returned 0 results."
    lines = [f"Corpus search for '{query[:60]}' — {len(hits)} result(s):"]
    for i, hit in enumerate(hits, 1):
        title = hit.get("title") or "(untitled)"
        url = hit.get("url") or ""
        snippet = " ".join((str(hit.get("content") or "")).split())[:200]
        score = hit.get("rrf_score", 0.0)
        lines.append(f"{i}. {title} / {snippet} / {url} [rrf={score}]")
    return "\n".join(lines)


# ── public tools ────────────────────────────────────────────────────────


async def _index_single(
    url: str | None,
    content: str | None,
    title: str | None,
    content_hash: str | None,
    metadata: dict[str, str] | None,
) -> bool:
    """Index one document into both stores; True if either leg succeeded."""
    from src.tools.builtin.web_scraper import compute_content_hash

    body = content or ""
    if not body and not url:
        return False
    chash = content_hash or compute_content_hash((body or url or "")[:10000])
    doc = _meili_doc(url, title, body, chash, metadata)

    meili_ok = False
    try:
        async with httpx.AsyncClient() as client:
            await _meili_add_documents(client, [doc])
        meili_ok = True
    except CorpusError as exc:
        logger.debug(f"index_corpus meilisearch leg failed: {exc}")

    cold_ok = await _cold_store(url, title, body, chash, metadata) is not None
    return meili_ok or cold_ok


async def index_corpus(
    url: str | None = None,
    content: str | None = None,
    title: str = "",
    content_hash: str | None = None,
    metadata: dict[str, str] | None = None,
    documents: list[dict[str, Any]] | None = None,
) -> str:
    """Index scraped pages into the local hybrid corpus (Meilisearch + pgvector).

    Single-doc mode: pass ``url``/``content``/``title`` (plus optional
    ``content_hash`` and ``metadata`` from ``web_scraper``'s ``extract_page``).
    Batch mode: pass ``documents=[{url, content, title, ...}, ...]`` to index
    many pages concurrently (``SEARCH_BATCH_CONCURRENCY``).

    Returns a one-line status string. Never raises — a missing Meilisearch or
    DB degrades that leg silently.
    """
    if documents is not None:
        docs = list(documents)
        if not docs:
            return "Indexed 0/0 document(s) into the corpus (nothing provided)."
        sem = asyncio.Semaphore(_batch_concurrency())

        async def _one(d: dict[str, Any]) -> bool:
            async with sem:
                return await _index_single(
                    d.get("url"),
                    d.get("content"),
                    d.get("title"),
                    d.get("content_hash"),
                    d.get("metadata"),
                )

        results = await asyncio.gather(
            *[_one(d) for d in docs], return_exceptions=True
        )
        ok = sum(1 for r in results if r is True)
        return (
            f"Indexed {ok}/{len(docs)} document(s) into the corpus "
            f"(Meilisearch + cold memory)."
        )

    ok = await _index_single(url, content, title, content_hash, metadata)
    if not ok:
        return (
            "index_corpus needs `content` (single) or `documents` (batch); "
            "nothing was indexed."
        )
    key = url or (content_hash or "")[:8] or "(content)"
    return f"Indexed '{key}' into the corpus (Meilisearch + cold memory)."


async def _search_single(query: str, top_k: int) -> str:
    """Hybrid keyword+semantic search for one query; returns a formatted block."""
    leg_limit = max(top_k * 3, 10)
    keyword_hits: list[dict[str, Any]] = []
    try:
        async with httpx.AsyncClient() as client:
            keyword_hits = await _meili_search(client, query, leg_limit)
    except CorpusError as exc:
        logger.debug(f"corpus_search keyword leg failed: {exc}")

    semantic_hits = await _cold_semantic_search(query, leg_limit)
    fused = rrf_fuse(
        keyword_hits, semantic_hits, k=get_settings().search.corpus_rrf_k, final_limit=top_k
    )
    return _format_hits(query, fused)


async def corpus_search(
    query: str = "",
    queries: list[str] | None = None,
    top_k: int = 5,
) -> str:
    """Hybrid search over the agent's gathered corpus (Meilisearch + pgvector).

    Fuses keyword (BM25) and semantic (pgvector) results via Reciprocal Rank
    Fusion. Batch mode: pass ``queries=[...]`` to run several concurrently
    (``SEARCH_BATCH_CONCURRENCY``); each query gets its own ranked block.

    Returns formatted ranked results. Never raises — an unavailable store
    yields an empty leg, and a fully-offline search returns "0 results".
    """
    if queries:
        qs = list(queries)
        if not qs:
            return "Corpus batch search: no queries provided."
        sem = asyncio.Semaphore(_batch_concurrency())

        async def _one(q: str) -> str:
            async with sem:
                try:
                    return await _search_single(q, top_k)
                except Exception as exc:  # one bad query never sinks the batch
                    logger.debug(f"corpus_search query failed: {q[:40]}: {exc}")
                    return f"Corpus search for '{q[:60]}' returned 0 results."

        blocks = await asyncio.gather(*[_one(q) for q in qs])
        return "\n\n".join(f"## {q}\n{b}" for q, b in zip(qs, blocks))

    if not query.strip():
        return "corpus_search needs a `query` (single) or `queries` (batch)."
    return await _search_single(query, top_k)


TOOL_DEFINITION_INDEX = {
    "name": "index_corpus",
    "handler": index_corpus,
    "description": (
        "Persist scraped pages into the agent's local hybrid corpus "
        "(Meilisearch keyword index + pgvector semantic memory) so later runs "
        "can recall them via corpus_search without re-scraping. Index one page "
        "(url/content/title) or a batch (documents=[...]). Best fed from "
        "web_scraper's extracted content. Idempotent: identical content "
        "re-indexed collapses to one entry. Never raises."
    ),
    # Indexing is a write — not cached. Re-indexing the same content is a no-op
    # at the store layer (distinctAttribute + content_hash), so it's safe to retry.
    "cacheable": False,
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Source URL of the page (optional)."},
            "content": {"type": "string", "description": "Extracted markdown/text of the page."},
            "title": {"type": "string", "description": "Page title.", "default": ""},
            "content_hash": {
                "type": "string",
                "description": "Optional dedup hash from web_scraper's extract_page.",
            },
            "metadata": {
                "type": "object",
                "description": "Optional page metadata (hostname/author/date) from extract_page.",
            },
            "documents": {
                "type": "array",
                "description": "Batch of {url,content,title,content_hash,metadata} dicts to index concurrently.",
                "items": {"type": "object"},
            },
        },
        "required": [],
    },
}

TOOL_DEFINITION_SEARCH = {
    "name": "corpus_search",
    "handler": corpus_search,
    "description": (
        "Hybrid search over the agent's previously-indexed research corpus "
        "(pages it has already scraped and stored). Fuses Meilisearch keyword "
        "(BM25) and pgvector semantic results via Reciprocal Rank Fusion. Use "
        "this to recall prior research instead of re-scraping the live web. "
        "Pass queries=[...] for a concurrent multi-query batch. Never raises."
    ),
    "cacheable": True,
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural language search query."},
            "queries": {
                "type": "array",
                "description": "Batch of queries to run concurrently.",
                "items": {"type": "string"},
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum results to return (default 5).",
                "default": 5,
            },
        },
        "required": [],
    },
}

# Registry convenience: expose both definitions.
TOOL_DEFINITION = TOOL_DEFINITION_SEARCH
