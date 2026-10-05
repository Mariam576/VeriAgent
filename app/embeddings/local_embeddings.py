"""Free local embeddings with a SentenceTransformer model.

No API key and no per-request cost. The model (about 90 MB for the default
all-MiniLM-L6-v2) is downloaded once on first use and then cached.
"""

from __future__ import annotations

import logging
import threading

from app.embeddings.base import EmbeddingError, EmbeddingProvider

logger = logging.getLogger(__name__)


class LocalEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str, dimension: int) -> None:
        self.model_name = model_name
        self.model_id = f"local:{model_name}"
        self.dimension = dimension
        self._model = None
        self._load_lock = threading.Lock()
        self._lock = threading.Lock()

    def _load(self):
        # Loading is slow, so do it once, and only when first needed.
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    try:
                        from sentence_transformers import SentenceTransformer
                    except ImportError as exc:
                        raise EmbeddingError(
                            "sentence-transformers is not installed. Run "
                            "'pip install sentence-transformers' or set EMBEDDING_PROVIDER=gemini."
                        ) from exc
                    logger.info("Loading local embedding model %s (first use can take a minute)", self.model_name)
                    try:
                        self._model = SentenceTransformer(self.model_name)
                    except Exception as exc:  # noqa: BLE001
                        raise EmbeddingError(f"Could not load embedding model {self.model_name}: {exc}") from exc
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        try:
            # Several searches may run at once in worker threads; the model's
            # tokenizer is not safe to share between threads, so encode one at a time.
            with self._lock:
                vectors = model.encode(
                    texts,
                    batch_size=32,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ).tolist()
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(f"Local embedding failed: {exc}") from exc
        self._check_dimension(vectors)
        return vectors

    def embed_text(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
