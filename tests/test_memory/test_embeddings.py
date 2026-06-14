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


class TestApiEmbedding:
    """Tests for the litellm-backed API path and settings configuration (§10.2)."""

    def test_reads_model_and_dimension_from_settings(self) -> None:
        """embedding_model / embedding_dim flow from settings.llm."""
        from unittest.mock import MagicMock

        settings = MagicMock()
        settings.llm.embedding_model = "custom-embed-model"
        settings.llm.embedding_dim = 512
        gen = EmbeddingGenerator(settings=settings)
        assert gen.model == "custom-embed-model"
        assert gen.dimension == 512

    @pytest.mark.asyncio
    async def test_api_embedding_passes_model_and_dimensions(self) -> None:
        """A successful API call uses the configured model and passes dimensions."""
        from unittest.mock import AsyncMock, MagicMock, patch

        settings = MagicMock()
        settings.llm.embedding_model = "custom-embed-model"
        settings.llm.embedding_dim = 768
        gen = EmbeddingGenerator(settings=settings)

        fake_vec = [0.01] * 768
        fake_response = MagicMock()
        fake_response.data = [{"embedding": fake_vec}]
        with patch("litellm.aembedding", new=AsyncMock(return_value=fake_response)) as mock_aemb:
            vec = await gen.generate("some text")

        assert vec == fake_vec
        kwargs = mock_aemb.call_args.kwargs
        assert kwargs["model"] == "custom-embed-model"
        assert kwargs["dimensions"] == 768  # reduces output to match Vector(768)
        assert kwargs["input"] == "some text"

    @pytest.mark.asyncio
    async def test_api_failure_falls_back_to_hash(self) -> None:
        """When litellm.aembedding raises, the hash fallback is used (never raises)."""
        from unittest.mock import AsyncMock, patch

        gen = EmbeddingGenerator()  # defaults: text-embedding-3-small, 768
        with patch("litellm.aembedding", new=AsyncMock(side_effect=RuntimeError("no key"))):
            vec = await gen.generate("fallback path")
        assert len(vec) == 768  # hash fallback honors the default dimension

    @pytest.mark.asyncio
    async def test_hash_fallback_respects_configured_dimension(self) -> None:
        """A non-default configured dimension is honored by the hash fallback."""
        from unittest.mock import AsyncMock, MagicMock, patch

        settings = MagicMock()
        settings.llm.embedding_model = "any-model"
        settings.llm.embedding_dim = 512
        gen = EmbeddingGenerator(settings=settings)
        assert gen.dimension == 512
        with patch("litellm.aembedding", new=AsyncMock(side_effect=RuntimeError("no key"))):
            vec = await gen.generate("dim check")
        assert len(vec) == 512
