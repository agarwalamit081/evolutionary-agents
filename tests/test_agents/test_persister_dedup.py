"""Tests for ``SubAgentPersister.find_similar`` (B3 semantic dedup).

Mirrors ``tests/test_tools/test_dynamic/test_persister_dedup.py`` for the
sub-agent persister. The pgvector cosine math runs in the DB; here we verify the
post-query contract deterministically via a mocked session: similarity is
``1 - distance``, the threshold filter applies, and DB errors degrade to ``[]``.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.persister import SubAgentPersister


def _fake_get_session(rows: list[tuple[Any, ...]] | Exception):
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
        rows = [("data_analyzer", "Analyzes data", 0.05)]
        with patch("src.db.session.get_session", _fake_get_session(rows)):
            matches = await SubAgentPersister().find_similar([0.1] * 768)

        assert len(matches) == 1
        assert matches[0]["name"] == "data_analyzer"
        assert matches[0]["description"] == "Analyzes data"
        assert matches[0]["similarity"] == pytest.approx(0.95)

    @pytest.mark.asyncio
    async def test_filters_rows_below_threshold(self) -> None:
        rows = [("weak", "d", 0.6)]  # similarity 0.4 < 0.85
        with patch("src.db.session.get_session", _fake_get_session(rows)):
            matches = await SubAgentPersister().find_similar([0.1] * 768)
        assert matches == []

    @pytest.mark.asyncio
    async def test_mixed_rows_only_keeps_above_threshold(self) -> None:
        rows = [
            ("good", "d1", 0.03),  # 0.97
            ("bad", "d2", 0.30),   # 0.70
        ]
        with patch("src.db.session.get_session", _fake_get_session(rows)):
            matches = await SubAgentPersister().find_similar([0.1] * 768)
        assert [m["name"] for m in matches] == ["good"]
        assert matches[0]["similarity"] == pytest.approx(0.97)

    @pytest.mark.asyncio
    async def test_db_error_degrades_to_empty(self) -> None:
        with patch(
            "src.db.session.get_session",
            _fake_get_session(RuntimeError("connection lost")),
        ):
            matches = await SubAgentPersister().find_similar([0.1] * 768)
        assert matches == []
