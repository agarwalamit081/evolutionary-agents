"""Unit tests for query-operator injection + multi-query expansion (S13).

Covers three layers, all deterministic (no network, no LLM):
  * ``_build_query`` operator injection (site/filetype/exact/exclude) — opt-in,
    default leaves the natural-language query untouched (no recall regression).
  * ``expand_query_variants`` — heuristic phrasing variants for multi-query recall.
  * ``web_search`` integration — multi_query issues the variants, merges the
    unique results, and dedups; operators thread through single + batch paths.
"""

from __future__ import annotations

import pytest

from src.tools.builtin import web_search as ws


# ── _build_query operator injection ──────────────────────────────────────


class TestBuildQueryOperators:
    def test_default_is_a_noop_on_natural_language(self) -> None:
        """No operators → query is only whitespace-collapsed (no narrowing)."""
        assert ws._build_query("  multi   word  query ") == "multi word query"

    def test_site_bare_host(self) -> None:
        assert ws._build_query("topic", site="arxiv.org") == "topic site:arxiv.org"

    def test_site_extracts_host_from_url(self) -> None:
        """A full URL reduces to its bare lowercased host (scheme/path dropped)."""
        q = ws._build_query("topic", site="https://WWW.Arxiv.org/abs/1234")
        assert q == "topic site:www.arxiv.org"

    def test_site_uses_first_of_comma_list(self) -> None:
        assert ws._build_query("topic", site="arxiv.org, b.com") == "topic site:arxiv.org"

    def test_filetype_strips_leading_dot_and_lowercases(self) -> None:
        assert ws._build_query("topic", filetype=".PDF") == "topic filetype:pdf"

    def test_filetype_drops_non_alnum(self) -> None:
        assert ws._build_query("topic", filetype="pdf") == "topic filetype:pdf"

    def test_exact_phrase_quoted_and_normalized(self) -> None:
        q = ws._build_query("topic", exact="  attention   is all you need  ")
        assert q == 'topic "attention is all you need"'

    def test_exclude_plain_term(self) -> None:
        assert ws._build_query("topic", exclude="spam") == "topic -spam"

    def test_exclude_domain_becomes_negated_site(self) -> None:
        q = ws._build_query("topic", exclude="pinterest.com")
        assert q == "topic -site:pinterest.com"

    def test_exclude_multiword_token_quoted(self) -> None:
        q = ws._build_query("topic", exclude="free shipping")
        assert q == 'topic -"free shipping"'

    def test_exclude_comma_list(self) -> None:
        q = ws._build_query("topic", exclude="pinterest.com, quizlet, free trial")
        assert q == "topic -site:pinterest.com -quizlet -\"free trial\""

    def test_all_operators_combined(self) -> None:
        q = ws._build_query(
            "topic", site="arxiv.org", filetype="pdf", exact="key phrase", exclude="blog",
        )
        assert q == 'topic site:arxiv.org filetype:pdf "key phrase" -blog'

    def test_empty_operators_all_noop(self) -> None:
        """Explicitly-empty operators must add nothing (the safety contract)."""
        assert ws._build_query("topic", site="", filetype="", exact="", exclude="") == "topic"


# ── expand_query_variants ────────────────────────────────────────────────


class TestExpandQueryVariants:
    def test_empty_query(self) -> None:
        assert ws.expand_query_variants("") == []
        assert ws.expand_query_variants("   ") == []

    def test_single_word_has_no_extra_variants(self) -> None:
        """A lone content word: no stop-word stripping, no single-token exact."""
        assert ws.expand_query_variants("photosynthesis") == ["photosynthesis"]

    def test_strips_stopwords_plus_exact_variant(self) -> None:
        out = ws.expand_query_variants("how to bake bread")
        # base leads; stop-word-stripped narrows to content words; exact wraps.
        assert out[0] == "how to bake bread"
        assert "bake bread" in out
        assert '"how to bake bread"' in out

    def test_all_stopwords_keeps_base_and_exact_only(self) -> None:
        """Stripping yields nothing, so that variant is skipped; exact still added."""
        out = ws.expand_query_variants("what is the")
        assert out == ["what is the", '"what is the"']

    def test_no_stopword_dup_when_stripped_equals_base(self) -> None:
        """A query with no stop-words must not emit a duplicate stripped variant."""
        out = ws.expand_query_variants("bake bread")
        assert out == ["bake bread", '"bake bread"']

    def test_variants_are_deduped(self) -> None:
        """No variant string repeats; base always leads."""
        out = ws.expand_query_variants("the quick brown fox")
        assert len(out) == len(set(out))
        assert out[0] == "the quick brown fox"


# ── web_search integration (multi_query merge + dedup) ──────────────────


class TestMultiQueryIntegration:
    @pytest.mark.asyncio
    async def test_default_single_query_issues_one_fetch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """multi_query=False (default) → exactly one fetch, the normalized query."""
        seen: list[str] = []

        async def fake_fetch(q: str, *_a: object) -> list[dict[str, str]]:
            seen.append(q)
            return [{"title": "t", "href": "http://x", "body": "b"}]

        monkeypatch.setattr(ws, "_fetch_results", fake_fetch)
        await ws.web_search("how to bake bread", max_results=5)
        assert seen == ["how to bake bread"]

    @pytest.mark.asyncio
    async def test_multi_query_issues_all_variants(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[str] = []

        async def fake_fetch(q: str, *_a: object) -> list[dict[str, str]]:
            seen.append(q)
            return []

        monkeypatch.setattr(ws, "_fetch_results", fake_fetch)
        await ws.web_search("how to bake bread", multi_query=True, max_results=5)
        assert seen == ["how to bake bread", "bake bread", '"how to bake bread"']

    @pytest.mark.asyncio
    async def test_multi_query_merges_and_dedups(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each variant contributes a unique URL + a shared URL; shared collapses."""
        seen: list[str] = []

        async def fake_fetch(q: str, *_a: object) -> list[dict[str, str]]:
            seen.append(q)
            return [
                {"title": f"t-{q[:6]}", "href": f"http://unique/{q[:6]}", "body": "b"},
                {"title": "shared", "href": "http://shared/1", "body": "b"},
            ]

        monkeypatch.setattr(ws, "_fetch_results", fake_fetch)
        out = await ws.web_search("how to bake bread", multi_query=True, max_results=5)

        assert len(seen) == 3  # three variants issued
        urls = [ln.split("URL:", 1)[1].strip() for ln in out.splitlines() if "URL:" in ln]
        # 3 variant-unique URLs + 1 shared (deduped from 3) = 4 distinct.
        assert len(urls) == len(set(urls)) == 4
        assert "http://shared/1" in urls

    @pytest.mark.asyncio
    async def test_multi_query_caps_at_max_results(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Merged+deduped results are capped at max_results."""

        async def fake_fetch(q: str, *_a: object) -> list[dict[str, str]]:
            return [{"title": f"t{i}-{q[:4]}", "href": f"http://x/{q[:4]}/{i}", "body": "b"} for i in range(6)]

        monkeypatch.setattr(ws, "_fetch_results", fake_fetch)
        out = await ws.web_search("how to bake bread", multi_query=True, max_results=5)
        urls = [ln for ln in out.splitlines() if "URL:" in ln]
        assert len(urls) == 5  # capped, even though 3 variants × 6 = 18 raw

    @pytest.mark.asyncio
    async def test_operators_appended_to_each_variant(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[str] = []

        async def fake_fetch(q: str, *_a: object) -> list[dict[str, str]]:
            seen.append(q)
            return []

        monkeypatch.setattr(ws, "_fetch_results", fake_fetch)
        await ws.web_search("how to bake bread", multi_query=True, site="arxiv.org", max_results=5)
        assert seen == [
            "how to bake bread site:arxiv.org",
            "bake bread site:arxiv.org",
            '"how to bake bread" site:arxiv.org',
        ]

    @pytest.mark.asyncio
    async def test_batch_threads_operators_to_each_query(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[str] = []

        async def fake_fetch(q: str, *_a: object) -> list[dict[str, str]]:
            seen.append(q)
            return [{"title": "t", "href": f"http://x/{q}", "body": "b"}]

        monkeypatch.setattr(ws, "_fetch_results", fake_fetch)
        await ws.web_search(queries=["alpha", "beta"], site="arxiv.org", max_results=5)
        assert seen == ["alpha site:arxiv.org", "beta site:arxiv.org"]
