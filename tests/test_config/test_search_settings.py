"""Tests for SearchSettings (Phase 1: SearXNG + Meilisearch corpus stack).

Same default-assertion discipline as test_settings.py: importing the suite
pulls in litellm (conftest -> graph -> gateway) which side-effect-loads .env
into os.environ, so a code default is only asserted with BOTH the os.environ
value removed (monkeypatch.delenv) AND the .env file skipped (_env_file=None).
"""

from __future__ import annotations

import pytest


class TestSearchSettings:
    """SearchSettings defaults, env overrides, and the fallback_providers property."""

    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.config.settings import SearchSettings

        for var in (
            "SEARCH_PRIMARY", "SEARXNG_URL", "SEARXNG_TIMEOUT",
            "SEARXNG_MAX_RESULTS_PER_QUERY", "MEILISEARCH_URL", "MEILISEARCH_KEY",
            "MEILISEARCH_INDEX", "MEILISEARCH_TIMEOUT", "SEARCH_FALLBACK_PROVIDERS",
            "SEARCH_BATCH_CONCURRENCY", "CHUNK_SIZE", "CHUNK_OVERLAP",
            "DEEP_CRAWL_ENABLED",
        ):
            monkeypatch.delenv(var, raising=False)

        s = SearchSettings(_env_file=None)
        assert s.search_primary == "searxng"
        assert s.searxng_url == "http://localhost:8080"
        assert s.searxng_timeout == 10.0
        assert s.searxng_max_results_per_query == 10
        assert s.meilisearch_url == "http://localhost:7700"
        assert s.meilisearch_key == ""
        assert s.meilisearch_index == "turing_corpus"
        assert s.search_batch_concurrency == 5
        assert s.chunk_size == 1200
        assert s.chunk_overlap == 150
        # Heavy providers OFF by default — they are provisioned-only.
        assert s.deep_crawl_enabled is False

    def test_fallback_providers_property_parses_and_trims(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.config.settings import SearchSettings

        monkeypatch.delenv("SEARCH_FALLBACK_PROVIDERS", raising=False)
        monkeypatch.setenv("SEARCH_FALLBACK_PROVIDERS", " tavily ,  Brave ,SERPAPI ")
        s = SearchSettings(_env_file=None)
        # Whitespace trimmed, lowercased, order preserved, empties dropped.
        assert s.fallback_providers == ["tavily", "brave", "serpapi"]

    def test_fallback_providers_default_chain_is_lightweight_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The default fallback chain excludes heavy providers (firecrawl/apify)."""
        from src.config.settings import SearchSettings

        monkeypatch.delenv("SEARCH_FALLBACK_PROVIDERS", raising=False)
        s = SearchSettings(_env_file=None)
        assert "firecrawl" not in s.fallback_providers
        assert "apify" not in s.fallback_providers
        # Lightweight paid providers ARE in the default chain.
        assert "tavily" in s.fallback_providers
        assert "serper" in s.fallback_providers

    def test_env_override_searxng_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SearXNG/Meilisearch URLs are env-overridable (compose uses internal hostnames)."""
        from src.config.settings import SearchSettings

        monkeypatch.setenv("SEARXNG_URL", "http://searxng:8080")
        monkeypatch.setenv("MEILISEARCH_URL", "http://meilisearch:7700")
        s = SearchSettings(_env_file=None)
        assert s.searxng_url == "http://searxng:8080"
        assert s.meilisearch_url == "http://meilisearch:7700"

    def test_deep_crawl_can_be_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.config.settings import SearchSettings

        monkeypatch.setenv("DEEP_CRAWL_ENABLED", "true")
        s = SearchSettings(_env_file=None)
        assert s.deep_crawl_enabled is True

    def test_search_group_wired_into_root_settings(self) -> None:
        """get_settings().search exposes the SearchSettings group."""
        from src.config.settings import SearchSettings, get_settings

        assert isinstance(get_settings().search, SearchSettings)
