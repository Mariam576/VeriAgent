"""Gemini embedding API (uses the same GEMINI_API_KEY as the chat model)."""

from __future__ import annotations

from app.embeddings.base import EmbeddingError, EmbeddingProvider, l2_normalize


class GeminiEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_key: str, model_name: str, dimension: int) -> None:
        if not api_key:
            raise EmbeddingError("EMBEDDING_PROVIDER=gemini needs GEMINI_API_KEY to be set.")
        self.model_name = model_name
        self.model_id = f"gemini:{model_name}:{dimension}"
        self.dimension = dimension
        self._api_key = api_key
        self._client = None

    def _load(self):
        if self._client is None:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            self._client = GoogleGenerativeAIEmbeddings(
                model=self.model_name,
                google_api_key=self._api_key,
                output_dimensionality=self.dimension,
            )
        return self._client

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            vectors = self._load().embed_documents(texts, task_type="RETRIEVAL_DOCUMENT")
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(f"Gemini embedding request failed: {exc}") from exc
        # Gemini only pre-normalises full-size vectors; shortened ones need it here.
        vectors = [l2_normalize(list(vector)) for vector in vectors]
        self._check_dimension(vectors)
        return vectors

    def embed_text(self, text: str) -> list[float]:
        try:
            vector = self._load().embed_query(text, task_type="RETRIEVAL_QUERY")
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(f"Gemini embedding request failed: {exc}") from exc
        vector = l2_normalize(list(vector))
        self._check_dimension([vector])
        return vector
