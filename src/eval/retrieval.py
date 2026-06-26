"""Retrieval-quality eval harness — precision@k + MRR over a seeded fixture (E1).

The measurement backbone for the recall pillar (findings-05: "strong curation,
weak recall"). Turns a ranked item list into IR metrics (precision@k, MRR,
hit-rate) scored against a gold relevance set per query, so the selection
changes in E2/F1/F3 can be A/B-tested ("did recall actually improve?").

Pure metric engine: ``run_retrieval_eval`` takes a ``Retriever`` protocol
(async ``query,k -> ranked eval_ids``), so the metric math is fully unit-
testable with a fake retriever and no DB. ``memory_retriever`` adapts a live
``MemoryManager`` to that protocol for ``main.py --retrieval-eval``.

Determinism: ``DEFAULT_FIXTURE`` / ``DEFAULT_QUERIES`` are fixed + seeded; each
fixture item carries an ``eval-id:<id>`` tag so a live recall can map a result
back to the fixture regardless of the UUID the store assigns it — the basis of
gold-set membership. The hash-fallback embeddings (when no embedding provider is
wired) are coarse but stable, so a *relative* A/B (run A vs run B) is reliable
even when the absolute MRR is not.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from src.memory.manager import MemoryManager

# Marker embedded in a fixture item's tags so a live recall can recover the
# stable ``eval_id`` from whatever UUID the store assigned the row.
EVAL_ID_TAG_PREFIX = "eval-id:"

# A retriever maps (query, limit) -> ranked ``eval_id`` list (best-first). The
# harness calls it with limit=k; a live adapter may widen the pool internally.
Retriever = Callable[[str, int], Awaitable[list[str]]]


# ─── data model ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RetrievalItem:
    """A seeded memory item tagged with a stable eval marker.

    ``eval_id`` is embedded in the stored item's tags as
    ``f"eval-id:{eval_id}"`` (see :func:`item_tags`) so a live recall can map a
    result back to the fixture regardless of the UUID the store assigns it.
    """

    eval_id: str
    name: str
    content: str
    tier: str = "skill"  # "skill" | "fact"


@dataclass(frozen=True)
class RetrievalQuery:
    """A benchmark query + the set of fixture ``eval_id``s relevant to it."""

    query: str
    gold_eval_ids: frozenset[str]


@dataclass
class QueryScore:
    """Per-query retrieval metrics."""

    query: str
    precision_at_k: float
    reciprocal_rank: float  # 1/rank of first relevant; 0 if none
    hit: bool  # any relevant in the top-k
    retrieved: list[str]
    gold: list[str]


@dataclass
class RetrievalReport:
    """Aggregate retrieval-quality report over a query set."""

    queries: list[QueryScore] = field(default_factory=list)
    k: int = 5
    precision_at_k: float = 0.0  # mean over queries
    mrr: float = 0.0  # mean reciprocal rank
    hit_rate: float = 0.0  # fraction of queries with a top-k hit

    def to_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "n_queries": len(self.queries),
            "precision_at_k": round(self.precision_at_k, 4),
            "mrr": round(self.mrr, 4),
            "hit_rate": round(self.hit_rate, 4),
            "queries": [
                {
                    "query": q.query,
                    "precision_at_k": round(q.precision_at_k, 4),
                    "reciprocal_rank": round(q.reciprocal_rank, 4),
                    "hit": q.hit,
                    "retrieved": q.retrieved,
                    "gold": q.gold,
                }
                for q in self.queries
            ],
        }


# ─── metric engine (pure) ─────────────────────────────────────────────────────


def compute_metrics(
    ranked_ids: list[str], gold: set[str], k: int
) -> tuple[float, float, bool]:
    """Return ``(precision_at_k, reciprocal_rank, hit)`` for one ranked list.

    precision@k = ``|gold ∩ top-k| / k`` (standard IR; assumes k retrieved).
    reciprocal_rank = ``1 / (1-based rank of the FIRST relevant item)`` over the
    full ranked list, else ``0``. hit = any relevant item in the top-k.
    """
    if k <= 0:
        return 0.0, 0.0, False
    topk = ranked_ids[:k]
    relevant_topk = [rid for rid in topk if rid in gold]
    precision = len(relevant_topk) / k
    reciprocal_rank = 0.0
    for idx, rid in enumerate(ranked_ids, start=1):
        if rid in gold:
            reciprocal_rank = 1.0 / idx
            break
    hit = len(relevant_topk) > 0
    return precision, reciprocal_rank, hit


async def run_retrieval_eval(
    queries: list[RetrievalQuery],
    retriever: Retriever,
    k: int = 5,
) -> RetrievalReport:
    """Score ``retriever`` against ``queries`` -> :class:`RetrievalReport`.

    Calls ``retriever(query, k)`` per query, then applies
    :func:`compute_metrics`. Never raises: a retriever error is logged at WARNING
    and that query scores zero — the recall surface is measured, not aborted.
    """
    scores: list[QueryScore] = []
    for q in queries:
        try:
            ranked = await retriever(q.query, k)
        except Exception as exc:  # noqa: BLE001 — measure, don't abort
            logger.warning("Retrieval query failed ({!r}): {}", q.query, exc)
            ranked = []
        gold = set(q.gold_eval_ids)
        precision, reciprocal_rank, hit = compute_metrics(ranked, gold, k)
        scores.append(
            QueryScore(
                query=q.query,
                precision_at_k=precision,
                reciprocal_rank=reciprocal_rank,
                hit=hit,
                retrieved=ranked[:k],
                gold=sorted(gold),
            )
        )
    n = len(scores) or 1  # guard the empty-query-set division
    return RetrievalReport(
        queries=scores,
        k=k,
        precision_at_k=sum(s.precision_at_k for s in scores) / n,
        mrr=sum(s.reciprocal_rank for s in scores) / n,
        hit_rate=sum(1 for s in scores if s.hit) / n,
    )


# ─── fixture + live-memory adapter ────────────────────────────────────────────


def item_tags(item: RetrievalItem) -> list[str]:
    """The tag set stamped onto a fixture item when it is seeded into memory.

    ``retrieval-eval`` brands every seeded row so it is identifiable (and can be
    distinguished from a run's real learned skills); ``eval-id:<id>`` is the
    stable marker :func:`eval_id_from_tags` reads back at recall time.
    """
    return ["retrieval-eval", f"{EVAL_ID_TAG_PREFIX}{item.eval_id}"]


def eval_id_from_tags(tags: list[str] | None) -> str | None:
    """Extract the ``eval-id:<id>`` marker from a recalled item's tag list."""
    for tag in tags or []:
        if isinstance(tag, str) and tag.startswith(EVAL_ID_TAG_PREFIX):
            return tag[len(EVAL_ID_TAG_PREFIX) :]
    return None


def memory_retriever(
    memory: MemoryManager,
    tier: str = "skill",
    *,
    recall_limit: int = 10,
) -> Retriever:
    """Adapt a live ``MemoryManager`` to the :data:`Retriever` protocol.

    Each recall (``retrieve_skills``/``retrieve_facts``) returns dicts carrying
    a ``tags`` list; ONLY items tagged with an ``eval-id:`` marker count, so
    pre-existing memory never pollutes the fixture measurement. The marker IS
    the fixture ``eval_id`` — the basis of gold-set membership. ``recall_limit``
    widens the pool the store ranks before the harness takes the top-k.
    """

    async def _retrieve(query: str, k: int) -> list[str]:
        limit = max(k, recall_limit)
        if tier == "fact":
            items = await memory.retrieve_facts(query=query, limit=limit)
        else:
            items = await memory.retrieve_skills(query=query, limit=limit)
        # Order-preserving dedup: ``store_skill`` has no upsert key, so a
        # re-seeded fixture stores duplicate rows — without dedup the same
        # ``eval_id`` could occupy two top-k slots and inflate precision.
        out: list[str] = []
        seen: set[str] = set()
        for item in items:
            eid = eval_id_from_tags(item.get("tags"))
            if eid is None or eid in seen:
                continue
            seen.add(eid)
            out.append(eid)
        return out

    return _retrieve


async def seed_fixture(
    memory: MemoryManager, items: list[RetrievalItem]
) -> int:
    """Seed ``items`` into warm memory under their ``eval-id`` tags.

    Idempotent-ish: re-seeding re-stores rows (warm memory has no upsert key on
    name); callers that want a clean measurement should run against a fresh
    memory or accept duplicates. Returns the count of items stored. Errors are
    logged at WARNING and skipped — a partial seed still yields a measurable
    (if lower-recall) report rather than aborting.
    """
    stored = 0
    for item in items:
        try:
            if item.tier == "fact":
                await memory.store_fact(
                    key=item.eval_id,
                    value=item.content,
                    source="retrieval-eval",
                    tags=item_tags(item),
                )
            else:
                await memory.store_skill(
                    name=item.name,
                    content=item.content,
                    tags=item_tags(item),
                )
            stored += 1
        except Exception as exc:  # noqa: BLE001 — measure, don't abort
            logger.warning("Could not seed fixture item {!r}: {}", item.eval_id, exc)
    return stored


# A curated capability fixture. Content is written so the gold item(s) for each
# query are the semantically-closest stored skills — the property a real
# embedding recall must surface (and a flat-inject-all approach hides). Clusters
# of 2 topical peers give each query a multi-item gold set so precision@k has a
# meaningful range (single-gold caps precision@k at 1/k).
DEFAULT_FIXTURE: list[RetrievalItem] = [
    RetrievalItem(
        eval_id="csv-parser",
        name="csv_parser",
        content="Parse and validate CSV files: stream rows, handle quoting and "
        "embedded newlines, expose headers and infer column layout.",
    ),
    RetrievalItem(
        eval_id="csv-type-inferrer",
        name="csv_type_inferrer",
        content="Infer and cast CSV column dtypes — int, float, bool, date — and "
        "report per-column null counts and type mismatches.",
    ),
    RetrievalItem(
        eval_id="json-schema-validator",
        name="json_schema_validator",
        content="Validate a JSON document against a JSON Schema and report "
        "path-keyed validation errors with the failing constraint.",
    ),
    RetrievalItem(
        eval_id="json-normalizer",
        name="json_normalizer",
        content="Flatten and normalize nested JSON into typed rows, merging "
        "arrays of objects and projecting onto a stable schema.",
    ),
    RetrievalItem(
        eval_id="utc-timestamp-normalizer",
        name="utc_timestamp_normalizer",
        content="Normalize mixed ISO-8601 and timezone-offset timestamps to a "
        "canonical UTC ISO-8601 string, preserving monotonic order.",
    ),
    RetrievalItem(
        eval_id="epoch-converter",
        name="epoch_converter",
        content="Convert unix epoch seconds and milliseconds to and from "
        "ISO-8601 timestamps, applying an explicit timezone offset.",
    ),
    RetrievalItem(
        eval_id="histogram-plotter",
        name="histogram_plotter",
        content="Render a histogram PNG from a numeric series via matplotlib "
        "with a configurable bin count and density toggle.",
    ),
    RetrievalItem(
        eval_id="scatter-plotter",
        name="scatter_plotter",
        content="Render a scatter plot PNG from paired x/y numeric series with "
        "axis labels and an optional regression line.",
    ),
    RetrievalItem(
        eval_id="web-page-extractor",
        name="web_page_extractor",
        content="Fetch a URL and extract the main-article text from HTML via "
        "trafilatura, returning clean markdown without boilerplate.",
    ),
    RetrievalItem(
        eval_id="markdown-to-html",
        name="markdown_to_html",
        content="Convert Markdown to sanitized HTML, stripping script/style "
        "tags and dangerous attributes while preserving structure.",
    ),
    RetrievalItem(
        eval_id="prime-sieve",
        name="prime_sieve",
        content="Generate all primes up to N via the Sieve of Eratosthenes and "
        "return them as a sorted list of integers.",
    ),
    RetrievalItem(
        eval_id="collatz-sequence",
        name="collatz_sequence",
        content="Generate the Collatz (hailstone) sequence for a starting "
        "integer up to the 1 terminator, returning the step list.",
    ),
]

# Each query's gold set is its two topical peers. k=3 leaves one distractor slot
# in the top-k so a perfect retriever scores precision 2/3 (not a trivial 1.0)
# and a recall miss scores 0 — the range the harness is built to discriminate.
DEFAULT_QUERIES: list[RetrievalQuery] = [
    RetrievalQuery(
        query="Load a CSV and check its column types",
        gold_eval_ids=frozenset({"csv-parser", "csv-type-inferrer"}),
    ),
    RetrievalQuery(
        query="Validate and normalize a JSON document",
        gold_eval_ids=frozenset({"json-schema-validator", "json-normalizer"}),
    ),
    RetrievalQuery(
        query="Work with timestamps across formats",
        gold_eval_ids=frozenset({"utc-timestamp-normalizer", "epoch-converter"}),
    ),
    RetrievalQuery(
        query="Make charts from numeric data",
        gold_eval_ids=frozenset({"histogram-plotter", "scatter-plotter"}),
    ),
    RetrievalQuery(
        query="Process text and HTML documents",
        gold_eval_ids=frozenset({"web-page-extractor", "markdown-to-html"}),
    ),
    RetrievalQuery(
        query="Run a number-theory computation",
        gold_eval_ids=frozenset({"prime-sieve", "collatz-sequence"}),
    ),
]

# Default cutoff: 2 gold + 1 distractor slot = a non-trivial precision range.
DEFAULT_K: int = 3
