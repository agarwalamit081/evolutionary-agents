"""Embedding generation for cold memory vector storage.

Uses litellm.aembedding() with OpenAI's text-embedding-3-small model
for generating vector embeddings. Falls back to simple hash-based
vectors when no API key is available.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Any

from loguru import logger


class EmbeddingGenerator:
    """Generates vector embeddings for text content.

    Uses litellm.aembedding() when an API key is available.
    Falls back to deterministic hash-based pseudo-embeddings.
    """

    def __init__(self, settings: Any = None) -> None:
        self._settings = settings
        self._dimension = 768  # Match pgvector column dimension
        self._model = "text-embedding-3-small"

    async def generate(self, text: str) -> list[float]:
        """Generate an embedding vector for the given text.

        Args:
            text: Input text to embed.

        Returns:
            List of floats representing the embedding vector.
        """
        # Try API-based embedding first
        embedding = await self._api_embedding(text)
        if embedding is not None:
            return embedding

        # Fallback: deterministic hash-based pseudo-embedding
        return self._hash_embedding(text)

    async def _api_embedding(self, text: str) -> list[float] | None:
        """Generate embedding via litellm. Returns None on failure."""
        try:
            import litellm

            response = await litellm.aembedding(
                model=self._model,
                input=text[:8000],  # Truncate to token limit
            )
            if response.data and len(response.data) > 0:
                return response.data[0]["embedding"]
        except Exception as e:
            logger.debug(f"API embedding failed, using hash fallback: {e}")
        return None

    def _hash_embedding(self, text: str) -> list[float]:
        """Generate a deterministic pseudo-embedding from text hash.

        Not semantically meaningful but provides consistent vectors
        for testing and offline use.
        """
        embedding: list[float] = []
        text_bytes = text.encode("utf-8")

        for i in range(self._dimension):
            # Create a deterministic but varied float per dimension
            chunk = text_bytes[i % len(text_bytes):(i % len(text_bytes)) + 8] or text_bytes[:8]
            hash_val = hashlib.sha256(chunk + struct.pack(">I", i)).digest()
            # Convert first 4 bytes to float in [-1, 1]
            int_val = struct.unpack(">I", hash_val[:4])[0]
            embedding.append((int_val / 0xFFFFFFFF) * 2.0 - 1.0)

        return embedding
