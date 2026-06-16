"""Tests for ``ToolPersister.find_similar`` (B3 semantic dedup).

``find_similar`` cosine-searches active tool capability embeddings and returns
those at/above the similarity threshold. The pgvector cosine_distance math runs
in the DB; here we verify the **post-query** contract deterministically by
mocking the session: similarity = 1 - distance is computed correctly, the
threshold filter is applied, rows come back most-similar-first, and any DB error
degrades to ``[]`` (so dedup never blocks a run).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.dynamic.persister import ToolPersister


def _fake_get_session(rows: list[tuple[Any, ...]] | Exception):
    """Patch target: an async context manager yielding a mock session whose
    ``execute`` returns ``rows`` (or raises if ``rows`` is an Exception)."""

    @asynccontextmanager
    async def _ctx():  # type: ignore[no-untyped-def]
        session = MagicMock()
        if isinstance(rows, Exception):
            session.execute = AsyncMock(side_effect=rows)
        else:
            result = MagicMock()
            result.all = MagicMock(return_value=rows)
            session.execute = AsyncMock(return_value=result)
        yield session

    return _ctx


class TestFindSimilar:
    @pytest.mark.asyncio
    async def test_returns_rows_above_threshold_with_similarity(self) -> None:
        # distance 0.05 -> similarity 0.95 (>= default 0.85)
        rows = [("http_fetcher", "Fetch URLs", 0.05)]
        with patch(
            "src.db.session.get_session",
            _fake_get_session(rows),
        ):
            matches = await ToolPersister().find_similar([0.1] * 768)

        assert len(matches) == 1
        assert matches[0]["tool_name"] == "http_fetcher"
        assert matches[0]["description"] == "Fetch URLs"
        assert matches[0]["similarity"] == pytest.approx(0.95)

    @pytest.mark.asyncio
    async def test_filters_rows_below_threshold(self) -> None:
        # distance 0.5 -> similarity 0.5 (< 0.85) -> dropped
        rows = [("weak_match", "d", 0.5)]
        with patch("src.db.session.get_session", _fake_get_session(rows)):
            matches = await ToolPersister().find_similar([0.1] * 768)
        assert matches == []

    @pytest.mark.asyncio
    async def test_mixed_rows_only_keeps_above_threshold(self) -> None:
        rows = [
            ("good", "d1", 0.02),   # 0.98 >= 0.85
            ("bad", "d2", 0.40),    # 0.60 < 0.85
            ("also_good", "d3", 0.10),  # 0.90 >= 0.85
        ]
        with patch("src.db.session.get_session", _fake_get_session(rows)):
            matches = await ToolPersister().find_similar([0.1] * 768)
        names = [m["tool_name"] for m in matches]
        assert names == ["good", "also_good"]
        assert matches[0]["similarity"] == pytest.approx(0.98)
        assert matches[1]["similarity"] == pytest.approx(0.90)

    @pytest.mark.asyncio
    async def test_respects_custom_threshold(self) -> None:
        # distance 0.1 -> similarity 0.9; with threshold 0.95 it's filtered out
        rows = [("near", "d", 0.10)]
        with patch("src.db.session.get_session", _fake_get_session(rows)):
            matches = await ToolPersister().find_similar(
                [0.1] * 768, threshold=0.95
            )
        assert matches == []

    @pytest.mark.asyncio
    async def test_db_error_degrades_to_empty(self) -> None:
        """A DB failure must not raise — dedup degrades to 'create' instead."""
        with patch(
            "src.db.session.get_session",
            _fake_get_session(RuntimeError("connection lost")),
        ):
            matches = await ToolPersister().find_similar([0.1] * 768)
        assert matches == []
