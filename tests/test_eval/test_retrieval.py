"""Retrieval-quality harness (E1): pure metric math + fake-retriever report.

The metric engine is decoupled from any DB via the ``Retriever`` protocol, so
``compute_metrics`` / ``run_retrieval_eval`` are unit-testable with hand-crafted
ranked/gold lists and a fake async retriever over the real ``DEFAULT_QUERIES``.
The ``memory_retriever`` adapter (the only live-DB seam) is covered with a stub
memory; the fixture/query sanity is asserted so the gold sets never drift from
the seeded ``eval-id``s.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from src.eval.retrieval import (
    DEFAULT_FIXTURE,
    DEFAULT_K,
    DEFAULT_QUERIES,
    RetrievalItem,
    RetrievalReport,
    compute_metrics,
    eval_id_from_tags,
    item_tags,
    memory_retriever,
    run_retrieval_eval,
    seed_fixture,
)

# ─── metric engine (pure math) ────────────────────────────────────────────────


class TestComputeMetrics:
    def test_perfect_recall_two_gold(self) -> None:
        # both gold in top-k, first is gold -> precision 2/3, MRR 1.0, hit
        precision, rr, hit = compute_metrics(["a", "b", "c"], {"a", "b"}, k=3)
        assert precision == pytest.approx(2 / 3)
        assert rr == 1.0
        assert hit is True

    def test_second_position_relevant(self) -> None:
        # first relevant at rank 2, both gold still in top-k
        precision, rr, hit = compute_metrics(["c", "a", "b"], {"a", "b"}, k=3)
        assert precision == pytest.approx(2 / 3)
        assert rr == pytest.approx(0.5)
        assert hit is True

    def test_relevant_below_topk_still_reciprocal_rank(self) -> None:
        # gold present in the list but NOT in top-k -> hit False, rr > 0
        precision, rr, hit = compute_metrics(["c", "d", "e", "a"], {"a"}, k=3)
        assert precision == 0.0
        assert rr == pytest.approx(0.25)
        assert hit is False

    def test_completely_missed(self) -> None:
        precision, rr, hit = compute_metrics(["c", "d", "e"], {"a", "b"}, k=3)
        assert precision == 0.0
        assert rr == 0.0
        assert hit is False

    def test_zero_k_is_noop(self) -> None:
        precision, rr, hit = compute_metrics(["a"], {"a"}, k=0)
        assert precision == 0.0
        assert rr == 0.0
        assert hit is False

    def test_short_ranked_list_dilutes_precision(self) -> None:
        # only 1 retrieved but it is gold -> precision 1/3 (not 1.0), MRR 1.0
        precision, rr, hit = compute_metrics(["a"], {"a"}, k=3)
        assert precision == pytest.approx(1 / 3)
        assert rr == 1.0
        assert hit is True


# ─── report over the real fixture via a fake retriever ─────────────────────────

_ALL_EVAL_IDS = [it.eval_id for it in DEFAULT_FIXTURE]


def _gold_list(q: Any) -> list[str]:
    return sorted(q.gold_eval_ids)


def _distractors(q: Any, n: int) -> list[str]:
    """First ``n`` fixture ids that are NOT gold for this query (deterministic)."""
    out = [eid for eid in _ALL_EVAL_IDS if eid not in q.gold_eval_ids]
    return out[:n]


def _perfect_retriever() -> Any:
    table = {q.query: _gold_list(q) + _distractors(q, 1) for q in DEFAULT_QUERIES}

    async def _retrieve(query: str, k: int) -> list[str]:
        _ = k
        return list(table.get(query, []))

    return _retrieve


def _worst_retriever() -> Any:
    table = {q.query: _distractors(q, 3) for q in DEFAULT_QUERIES}

    async def _retrieve(query: str, k: int) -> list[str]:
        _ = k
        return list(table.get(query, []))

    return _retrieve


class TestRunRetrievalEval:
    async def test_perfect_ranking_scores_high(self) -> None:
        report = await run_retrieval_eval(
            DEFAULT_QUERIES, _perfect_retriever(), k=DEFAULT_K
        )
        # 2 gold + 1 distractor per query -> precision 2/3, first is gold -> MRR 1
        assert len(report.queries) == len(DEFAULT_QUERIES)
        assert report.precision_at_k == pytest.approx(2 / 3)
        assert report.mrr == 1.0
        assert report.hit_rate == 1.0

    async def test_worst_ranking_scores_zero(self) -> None:
        report = await run_retrieval_eval(
            DEFAULT_QUERIES, _worst_retriever(), k=DEFAULT_K
        )
        assert report.precision_at_k == 0.0
        assert report.mrr == 0.0
        assert report.hit_rate == 0.0

    async def test_retriever_error_scores_zero_not_aborts(self) -> None:
        async def _raise(query: str, k: int) -> list[str]:
            raise RuntimeError("boom")

        report = await run_retrieval_eval(DEFAULT_QUERIES, _raise, k=DEFAULT_K)
        assert report.precision_at_k == 0.0
        assert report.mrr == 0.0
        assert report.hit_rate == 0.0
        assert len(report.queries) == len(DEFAULT_QUERIES)

    async def test_empty_query_set_is_safe(self) -> None:
        async def _retrieve(query: str, k: int) -> list[str]:
            return []

        report = await run_retrieval_eval([], _retrieve, k=DEFAULT_K)
        assert report.queries == []
        # no ZeroDivisionError: aggregates are 0 over a guarded denominator
        assert report.precision_at_k == 0.0
        assert report.mrr == 0.0


# ─── eval-id tag round-trip ────────────────────────────────────────────────────


class TestTags:
    def test_item_tags_includes_marker(self) -> None:
        assert item_tags(RetrievalItem("csv-parser", "csv_parser", "x")) == [
            "retrieval-eval",
            "eval-id:csv-parser",
        ]

    def test_eval_id_from_tags_extracts(self) -> None:
        assert eval_id_from_tags(["a", "eval-id:csv-parser", "b"]) == "csv-parser"

    def test_eval_id_from_tags_none_when_absent(self) -> None:
        assert eval_id_from_tags(["a", "b"]) is None
        assert eval_id_from_tags(None) is None

    def test_tags_roundtrip(self) -> None:
        for item in DEFAULT_FIXTURE:
            assert eval_id_from_tags(item_tags(item)) == item.eval_id


# ─── memory_retriever adapter (stub memory) ────────────────────────────────────


class _StubMemory:
    """Minimal async memory surface: skill/fact recall that returns dicts."""

    def __init__(self, skills: list[dict[str, Any]], facts: list[dict[str, Any]] | None = None) -> None:
        self._skills = skills
        self._facts = facts or []
        self.seen_limit: int | None = None

    async def retrieve_skills(self, *, query: str, limit: int) -> list[dict[str, Any]]:
        self.seen_limit = limit
        return list(self._skills)

    async def retrieve_facts(self, *, query: str, limit: int) -> list[dict[str, Any]]:
        self.seen_limit = limit
        return list(self._facts)


class TestMemoryRetriever:
    async def test_extracts_eval_ids_in_order(self) -> None:
        mem = _StubMemory(
            [
                {"tags": ["eval-id:csv-parser"]},
                {"tags": ["eval-id:json-normalizer"]},
            ]
        )
        retrieve = memory_retriever(mem, recall_limit=5)
        ranked = await retrieve("query", k=3)
        assert ranked == ["csv-parser", "json-normalizer"]

    async def test_dedups_duplicate_eval_ids(self) -> None:
        mem = _StubMemory(
            [
                {"tags": ["eval-id:csv-parser"]},
                {"tags": ["eval-id:csv-parser"]},  # re-seeded duplicate
                {"tags": ["eval-id:json-normalizer"]},
            ]
        )
        ranked = await memory_retriever(mem)("query", k=3)
        assert ranked == ["csv-parser", "json-normalizer"]

    async def test_skips_items_without_marker(self) -> None:
        mem = _StubMemory(
            [
                {"tags": ["retrieval-eval"]},  # branded but no eval-id
                {"tags": []},  # a real learned skill, not part of the fixture
                {"tags": ["eval-id:csv-parser"]},
            ]
        )
        ranked = await memory_retriever(mem)("query", k=3)
        assert ranked == ["csv-parser"]

    async def test_fact_tier_uses_retrieve_facts(self) -> None:
        mem = _StubMemory(
            skills=[{"tags": ["eval-id:should-not-appear"]}],
            facts=[{"tags": ["eval-id:prime-sieve"]}],
        )
        ranked = await memory_retriever(mem, tier="fact")("query", k=3)
        assert ranked == ["prime-sieve"]

    async def test_recall_limit_widens_pool(self) -> None:
        mem = _StubMemory([{"tags": ["eval-id:csv-parser"]}])
        # k=3 but recall_limit=10 -> store is asked for the wider pool
        await memory_retriever(mem, recall_limit=10)("query", k=3)
        assert mem.seen_limit == 10


# ─── seed_fixture (stub memory) ────────────────────────────────────────────────


class _RecordingMemory:
    """Records store_* calls; can raise on a chosen eval_id to test skip-on-error."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.skill_calls: list[dict[str, Any]] = []
        self.fact_calls: list[dict[str, Any]] = []
        self._fail_on = fail_on

    async def store_skill(self, *, name: str, content: str, tags: list[str]) -> None:
        if name == self._fail_on:
            raise RuntimeError("boom")
        self.skill_calls.append({"name": name, "content": content, "tags": tags})

    async def store_fact(self, *, key: str, value: str, source: str, tags: list[str]) -> None:
        if key == self._fail_on:
            raise RuntimeError("boom")
        self.fact_calls.append({"key": key, "value": value, "tags": tags})


class TestSeedFixture:
    async def test_seeds_all_items_with_tags(self) -> None:
        items = [
            RetrievalItem("csv-parser", "csv_parser", "parse csv"),
            RetrievalItem("prime-sieve", "prime_sieve", "primes"),
        ]
        mem = _RecordingMemory()
        stored = await seed_fixture(mem, items)
        assert stored == 2
        assert {c["name"] for c in mem.skill_calls} == {"csv_parser", "prime_sieve"}
        # every stored row carries its eval-id marker
        for call in mem.skill_calls:
            assert any(t.startswith("eval-id:") for t in call["tags"])

    async def test_skips_on_error_counts_successes(self) -> None:
        items = [
            RetrievalItem("csv-parser", "csv_parser", "parse csv"),
            RetrievalItem("bad-one", "bad_one", "x"),
            RetrievalItem("prime-sieve", "prime_sieve", "primes"),
        ]
        mem = _RecordingMemory(fail_on="csv_parser")  # first item's name
        stored = await seed_fixture(mem, items)
        assert stored == 2  # the failed one skipped, the other two stored
        assert len(mem.skill_calls) == 2

    async def test_fact_tier_routes_to_store_fact(self) -> None:
        items = [RetrievalItem("a-fact", "a_fact", "a value", tier="fact")]
        mem = _RecordingMemory()
        stored = await seed_fixture(mem, items)
        assert stored == 1
        assert mem.skill_calls == []
        assert len(mem.fact_calls) == 1


# ─── fixture / query sanity ───────────────────────────────────────────────────


class TestFixtureSanity:
    def test_every_gold_exists_in_fixture(self) -> None:
        fixture_ids = {it.eval_id for it in DEFAULT_FIXTURE}
        for q in DEFAULT_QUERIES:
            assert q.gold_eval_ids, f"empty gold for {q.query!r}"
            missing = q.gold_eval_ids - fixture_ids
            assert not missing, f"gold ids not in fixture: {missing}"

    def test_default_k_leaves_distractor_slot(self) -> None:
        # k must exceed the max gold-set size so precision@k has a real range
        max_gold = max(len(q.gold_eval_ids) for q in DEFAULT_QUERIES)
        assert DEFAULT_K > max_gold

    def test_default_queries_nonempty(self) -> None:
        assert len(DEFAULT_QUERIES) >= 1

    def test_fixture_ids_unique(self) -> None:
        ids = [it.eval_id for it in DEFAULT_FIXTURE]
        assert len(ids) == len(set(ids))

    def test_report_to_dict_json_serializable(self) -> None:
        # round-trip through json (no datetime/bytes/sets that break serialization)
        report = RetrievalReport(k=3)
        as_dict = report.to_dict()
        assert json.loads(json.dumps(as_dict))["k"] == 3
        assert as_dict["n_queries"] == 0
