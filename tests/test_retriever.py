"""Semantic retrieval against a real PostgreSQL + pgvector."""

from __future__ import annotations

import pytest

from app.database.session import get_session_factory
from app.embeddings.factory import get_embedding_provider
from app.rag.retriever import PgVectorRetriever, SearchFilters
from app.security.permissions import user_uuid_for

pytestmark = pytest.mark.integration


async def upload(client, headers, path):
    with path.open("rb") as handle:
        response = await client.post("/documents/upload", headers=headers, files={"file": (path.name, handle)})
    assert response.status_code == 201, response.text
    return response.json()


@pytest.fixture
def retriever(database):
    return PgVectorRetriever(get_session_factory(), get_embedding_provider())


@pytest.fixture
async def library(client, user_headers, samples_dir):
    """A user with the three sample documents uploaded."""
    documents = {}
    for name in ("Contract.pdf", "Invoice_102.pdf", "Procurement_Policy.pdf"):
        documents[name] = await upload(client, user_headers, samples_dir / name)
    return {"user_id": str(user_uuid_for(user_headers["X-User-Id"])), "documents": documents}


async def test_finds_the_right_page_with_source_metadata(retriever, library):
    results = await retriever.search(
        library["user_id"], "total contract value exclusive of VAT", top_k=8, rerank_top_k=4
    )
    assert 1 <= len(results) <= 4
    best = results[0]
    assert (best.filename, best.page) == ("Contract.pdf", 4)
    assert "£40,000" in best.text
    assert best.document_id == library["documents"]["Contract.pdf"]["id"]
    assert best.chunk_id and best.document_type == "contract"
    assert 0 < best.score <= 1
    assert results == sorted(results, key=lambda c: c.rerank_score, reverse=True)


async def test_top_k_limits_are_respected(retriever, library):
    results = await retriever.search(library["user_id"], "invoice payment", top_k=3, rerank_top_k=2)
    assert len(results) == 2


async def test_filter_by_document(retriever, library):
    invoice_id = library["documents"]["Invoice_102.pdf"]["id"]
    results = await retriever.search(
        library["user_id"], "payment terms days", top_k=8, rerank_top_k=4,
        filters=SearchFilters(document_ids=[invoice_id]),
    )
    assert results and {chunk.filename for chunk in results} == {"Invoice_102.pdf"}
    assert "60 days" in results[0].text


async def test_filter_by_document_type_page_and_year(retriever, library):
    by_type = await retriever.search(
        library["user_id"], "approval", top_k=8, rerank_top_k=8, filters=SearchFilters(document_type="policy")
    )
    assert by_type and {chunk.document_type for chunk in by_type} == {"policy"}

    by_page = await retriever.search(
        library["user_id"], "approval", top_k=8, rerank_top_k=8,
        filters=SearchFilters(document_type="policy", page=7),
    )
    assert [chunk.page for chunk in by_page] == [7]
    assert "Finance Director" in by_page[0].text

    assert await retriever.search(
        library["user_id"], "approval", top_k=8, rerank_top_k=8, filters=SearchFilters(year=1999)
    ) == []
    assert await retriever.search(
        library["user_id"], "approval", top_k=8, rerank_top_k=8, filters=SearchFilters(year=2026)
    )


async def test_other_users_documents_are_never_returned(retriever, library, client, samples_dir, user_headers):
    other_headers = {"X-User-Id": user_headers["X-User-Id"] + "-other"}
    other_user_id = str(user_uuid_for(other_headers["X-User-Id"]))

    # The other user has nothing yet: no results, even though matching chunks exist in the table.
    assert await retriever.search(other_user_id, "total contract value", top_k=8, rerank_top_k=4) == []

    # Asking for someone else's document by id does not help either.
    stolen = await retriever.search(
        other_user_id, "total contract value", top_k=8, rerank_top_k=4,
        filters=SearchFilters(document_ids=[library["documents"]["Contract.pdf"]["id"]]),
    )
    assert stolen == []

    await upload(client, other_headers, samples_dir / "Invoice_102.pdf")
    results = await retriever.search(other_user_id, "total contract value", top_k=8, rerank_top_k=4)
    assert results and {chunk.filename for chunk in results} == {"Invoice_102.pdf"}


async def test_searches_can_run_concurrently(retriever, library):
    import asyncio

    contract, invoice = await asyncio.gather(
        retriever.search(library["user_id"], "total contract value", top_k=8, rerank_top_k=2),
        retriever.search(library["user_id"], "total amount due invoice", top_k=8, rerank_top_k=2),
    )
    assert contract[0].filename == "Contract.pdf"
    assert invoice[0].filename == "Invoice_102.pdf"
