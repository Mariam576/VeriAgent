"""Gemini embedding API (uses the same GEMINI_API_KEY as the chat model)."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from app.embeddings.base import EmbeddingError, EmbeddingProvider, l2_normalize

logger = logging.getLogger(__name__)

# Seconds to wait before trying again when the API says "too many requests". A free key
# is limited per minute, and a long document needs several requests in a row, so the
# waits have to add up to more than a minute before giving up.
RETRY_WAITS = (5.0, 15.0, 30.0, 45.0)
_BUSY_SIGNS = ("429", "resource_exhausted", "resource exhausted", "quota", "rate limit", "503", "unavailable", "overloaded")


def is_busy(error: Exception) -> bool:
    """True for "slow down" and "try again later"; False for mistakes that waiting cannot fix."""
    text = str(error).lower()
    return any(sign in text for sign in _BUSY_SIGNS)


class GeminiEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_key: str, model_name: str, dimension: int) -> None:
        if not api_key:
            raise EmbeddingError("EMBEDDING_PROVIDER=gemini needs GEMINI_API_KEY to be set.")
        self.model_name = model_name
        self.model_id = f"gemini:{model_name}:{dimension}"
        self.dimension = dimension
        self._api_key = api_key
        self._client = None
        self._sleep: Callable[[float], None] = time.sleep  # replaced in tests

    def _load(self):
        if self._client is None:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            self._client = GoogleGenerativeAIEmbeddings(
                model=self.model_name,
                google_api_key=self._api_key,
                output_dimensionality=self.dimension,
            )
        return self._client

    def _request(self, call: Callable[[], Any]) -> Any:
        """Make one request; wait and repeat it while the API is busy.

        These methods run in a worker thread (see ``EmbeddingProvider``), so
        sleeping here does not hold up other requests.
        """
        for attempt, wait in enumerate((*RETRY_WAITS, None), start=1):
            try:
                return call()
            except Exception as exc:  # noqa: BLE001
                if wait is None or not is_busy(exc):
                    raise EmbeddingError(f"Gemini embedding request failed: {exc}") from exc
                logger.warning("Gemini embeddings are rate-limited (attempt %d); trying again in %.0f s", attempt, wait)
                self._sleep(wait)
        raise AssertionError("unreachable")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._request(lambda: self._load().embed_documents(texts, task_type="RETRIEVAL_DOCUMENT"))
        # Gemini only pre-normalises full-size vectors; shortened ones need it here.
        vectors = [l2_normalize(list(vector)) for vector in vectors]
        self._check_dimension(vectors)
        return vectors

    def embed_text(self, text: str) -> list[float]:
        vector = self._request(lambda: self._load().embed_query(text, task_type="RETRIEVAL_QUERY"))
        vector = l2_normalize(list(vector))
        self._check_dimension([vector])
        return vector
