import math

import pytest

from app.embeddings.base import EmbeddingError, EmbeddingProvider, l2_normalize
from app.embeddings.factory import get_embedding_provider
from app.embeddings.gemini_embeddings import GeminiEmbeddingProvider
from app.embeddings.hash_embeddings import HashEmbeddingProvider


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def test_factory_returns_the_configured_provider(settings):
    provider = get_embedding_provider()
    assert isinstance(provider, EmbeddingProvider)
    assert provider.model_id == f"hash:{settings.EMBEDDING_DIM}"
    assert provider.dimension == settings.EMBEDDING_DIM


def test_hash_embeddings_are_deterministic_and_normalised():
    provider = HashEmbeddingProvider(384)
    first = provider.embed_text("Payment terms are 30 days")
    second = provider.embed_text("Payment terms are 30 days")
    assert first == second
    assert len(first) == 384
    assert math.isclose(math.sqrt(sum(v * v for v in first)), 1.0, rel_tol=1e-9)


def test_similar_text_scores_higher_than_unrelated_text():
    provider = HashEmbeddingProvider(384)
    query = provider.embed_text("payment terms invoice days")
    related = provider.embed_text("The payment terms on the invoice are 60 days.")
    unrelated = provider.embed_text("The office kitchen is cleaned every Friday afternoon.")
    assert cosine(query, related) > cosine(query, unrelated)


def test_empty_text_still_gives_a_usable_vector():
    vector = HashEmbeddingProvider(16).embed_text("")
    assert any(value != 0 for value in vector)


async def test_async_versions_match_sync_versions():
    provider = HashEmbeddingProvider(64)
    texts = ["contract value", "invoice total"]
    assert await provider.aembed_documents(texts) == provider.embed_documents(texts)
    assert await provider.aembed_text("contract value") == provider.embed_text("contract value")
    assert await provider.aembed_documents([]) == []


def test_dimension_mismatch_is_reported_clearly():
    provider = HashEmbeddingProvider(8)
    with pytest.raises(EmbeddingError, match="EMBEDDING_DIM"):
        provider._check_dimension([[0.0] * 5])


def test_gemini_embeddings_need_an_api_key():
    with pytest.raises(EmbeddingError, match="GEMINI_API_KEY"):
        GeminiEmbeddingProvider("", "gemini-embedding-001", 384)


def test_l2_normalize_handles_zero_vector():
    assert l2_normalize([0.0, 0.0]) == [0.0, 0.0]
    assert l2_normalize([3.0, 4.0]) == [0.6, 0.8]


# ------------------------------------------------------------- found in review
class BusyThenFine:
    """Stands in for the Gemini client: refuses the first calls the way the real API does."""

    def __init__(self, refusals: int, error: str = "429 RESOURCE_EXHAUSTED: quota exceeded for embed_content") -> None:
        self.refusals, self.error, self.calls = refusals, error, 0

    def embed_documents(self, texts, task_type):
        self.calls += 1
        if self.calls <= self.refusals:
            raise RuntimeError(self.error)
        return [[3.0, 4.0] for _ in texts]

    def embed_query(self, text, task_type):
        return self.embed_documents([text], task_type)[0]


def gemini_with(client) -> tuple[GeminiEmbeddingProvider, list[float]]:
    provider = GeminiEmbeddingProvider("a-key", "gemini-embedding-001", 2)
    waits: list[float] = []
    provider._client, provider._sleep = client, waits.append
    return provider, waits


def test_gemini_embeddings_wait_and_try_again_when_the_api_is_busy():
    """A long document is embedded in several requests in a row; a free key allows only so many per
    minute. The second request failed, and with it the whole upload."""
    from app.embeddings.gemini_embeddings import RETRY_WAITS

    client = BusyThenFine(refusals=2)
    provider, waits = gemini_with(client)
    assert provider.embed_documents(["a", "b"]) == [[0.6, 0.8], [0.6, 0.8]]  # normalised to length 1
    assert client.calls == 3 and waits == list(RETRY_WAITS[:2])
    assert sum(RETRY_WAITS) > 60  # patient enough to outlast a per-minute limit

    client = BusyThenFine(refusals=1, error="503 The model is overloaded")
    provider, waits = gemini_with(client)
    assert provider.embed_text("a question") == [0.6, 0.8] and len(waits) == 1


def test_gemini_embeddings_give_up_in_the_end_and_never_retry_a_real_mistake():
    from app.embeddings.gemini_embeddings import RETRY_WAITS

    client = BusyThenFine(refusals=99)
    provider, waits = gemini_with(client)
    with pytest.raises(EmbeddingError, match="429"):
        provider.embed_documents(["a"])
    assert client.calls == len(RETRY_WAITS) + 1 and waits == list(RETRY_WAITS)

    client = BusyThenFine(refusals=99, error="400 API key not valid")
    provider, waits = gemini_with(client)
    with pytest.raises(EmbeddingError, match="API key not valid"):
        provider.embed_documents(["a"])
    assert client.calls == 1 and waits == []  # waiting does not fix a wrong key
