import math

import pytest

from app.embeddings.base import EmbeddingError, EmbeddingProvider, l2_normalize
from app.embeddings.factory import get_embedding_provider
from app.embeddings.gemini_embeddings import GeminiEmbeddingProvider
from app.embeddings.hash_embeddings import HashEmbeddingProvider


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def test_factory_returns_the_configured_provider():
    provider = get_embedding_provider()
    assert isinstance(provider, EmbeddingProvider)
    assert provider.model_id == "hash:384"
    assert provider.dimension == 384


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
