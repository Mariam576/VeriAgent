"""Pick the embedding provider from settings.

Everything else calls ``get_embedding_provider()`` and never imports a
specific provider, so switching is a one-line change in ``.env``.
"""

from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.embeddings.base import EmbeddingError, EmbeddingProvider


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    settings = get_settings()
    provider = settings.EMBEDDING_PROVIDER

    if provider == "local":
        from app.embeddings.local_embeddings import LocalEmbeddingProvider

        return LocalEmbeddingProvider(settings.LOCAL_EMBEDDING_MODEL, settings.EMBEDDING_DIM)

    if provider == "gemini":
        from app.embeddings.gemini_embeddings import GeminiEmbeddingProvider

        api_key = settings.GEMINI_API_KEY.get_secret_value() if settings.GEMINI_API_KEY else ""
        return GeminiEmbeddingProvider(api_key, settings.GEMINI_EMBEDDING_MODEL, settings.EMBEDDING_DIM)

    if provider == "hash":
        from app.embeddings.hash_embeddings import HashEmbeddingProvider

        return HashEmbeddingProvider(settings.EMBEDDING_DIM)

    raise EmbeddingError(f"Unsupported EMBEDDING_PROVIDER: {provider}")
