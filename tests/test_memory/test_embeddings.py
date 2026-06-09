"""Tests for src.memory.embeddings — embedding generation."""

from __future__ import annotations

import pytest

from src.memory.embeddings import EmbeddingGenerator


class TestHashEmbedding:
    """Tests for the _hash_embedding fallback method."""

    def test_returns_correct_dimension(self) -> None:
        """Hash embedding returns 768-dimensional vector."""
        gen = EmbeddingGenerator()
        vec = gen._hash_embedding("hello world")
        assert len(vec) == 768

    def test_is_deterministic(self) -> None:
        """Same input always produces the same vector."""
        gen = EmbeddingGenerator()
        vec1 = gen._hash_embedding("deterministic test")
        vec2 = gen._hash_embedding("deterministic test")
        assert vec1 == vec2

    def test_different_texts_differ(self) -> None:
        """Different inputs produce different vectors."""
        gen = EmbeddingGenerator()
        vec1 = gen._hash_embedding("text one")
        vec2 = gen._hash_embedding("text two")
        assert vec1 != vec2

    def test_values_in_range(self) -> None:
        """All embedding values are between -1.0 and 1.0."""
        gen = EmbeddingGenerator()
        vec = gen._hash_embedding("range check")
        assert all(-1.0 <= v <= 1.0 for v in vec)

    @pytest.mark.asyncio
    async def test_generate_returns_vector(self) -> None:
        """generate() returns a vector (hash fallback or API-based)."""
        gen = EmbeddingGenerator(settings=None)
        vec = await gen.generate("fallback test")
        assert len(vec) > 0
        # API-based returns 1536 dims; hash-based returns 768
        assert len(vec) in (768, 1536)
