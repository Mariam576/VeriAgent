"""Common interface for every embedding provider.

Concept - embeddings:
    An embedding model turns text into a list of numbers (a vector). Texts with
    similar meaning get vectors that point in a similar direction, so "how
    close are these two vectors?" becomes "how similar are these two texts?".
    That is what makes semantic search possible.

The rest of the application only ever calls ``embed_text`` (for a search
query) and ``embed_documents`` (for chunks being stored). It never knows
which provider is behind them.
"""

from __future__ import annotations

import asyncio
import math
from abc import ABC, abstractmethod


class EmbeddingError(Exception):
    """Raised when an embedding provider cannot produce vectors."""


class EmbeddingProvider(ABC):
    #: Unique id of the provider + model, stored with each document so that
    #: vectors from different models are never compared with each other.
    model_id: str
    dimension: int

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Embed one search query."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed many document chunks."""

    # Embedding is either CPU work (local model) or a blocking HTTP call, so
    # the async versions run it in a worker thread to keep the event loop free.
    async def aembed_text(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_text, text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self.embed_documents, texts)

    def _check_dimension(self, vectors: list[list[float]]) -> None:
        for vector in vectors:
            if len(vector) != self.dimension:
                raise EmbeddingError(
                    f"{self.model_id} returned {len(vector)}-dimensional vectors but "
                    f"EMBEDDING_DIM is {self.dimension}. Set EMBEDDING_DIM={len(vector)} "
                    "(and recreate the database if it already holds documents)."
                )


def l2_normalize(vector: list[float]) -> list[float]:
    """Scale a vector to length 1 so cosine similarity behaves consistently."""
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]
