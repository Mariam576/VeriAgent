"""Offline embeddings for tests: no model download, no network, fully deterministic.

Each word (and each pair of neighbouring words) is hashed to a position in the
vector. Two texts that share words end up with similar vectors. This measures
word overlap, not meaning, so it is for tests and offline demos only. Use the
"local" or "gemini" provider for real documents.
"""

from __future__ import annotations

import hashlib
import re

from app.embeddings.base import EmbeddingProvider, l2_normalize

_TOKEN = re.compile(r"[a-z0-9]+(?:[-./][a-z0-9]+)*")


class HashEmbeddingProvider(EmbeddingProvider):
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.model_id = f"hash:{dimension}"

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = _TOKEN.findall(text.lower())
        features = tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        if not features:
            vector[0] = 1.0  # pgvector cannot take the cosine of an all-zero vector
        return l2_normalize(vector)

    def embed_text(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]
