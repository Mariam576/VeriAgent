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


# ------------------------------------------------------------ hybrid search
@pytest.fixture
async def code_library(client, user_headers, tmp_path):
    """Several near-identical invoices that differ only in their codes: the hard case for embeddings."""
    documents = {}
    for number in (101, 102, 103, 120, 129, 192, 201, 210, 219, 291):
        path = tmp_path / f"Invoice_{number}.txt"
        path.write_text(
            f"INVOICE\nInvoice No. INV-{number}\nPurchase order: PO-{4300 + number}\n"
            "Planned maintenance services for the quarter.\nTotal amount due: £5,000\n"
            "Payment terms: 30 days from invoice date.\n",
            encoding="utf-8",
        )
        documents[number] = await upload(client, user_headers, path)
    return {"user_id": str(user_uuid_for(user_headers["X-User-Id"])), "documents": documents}


async def test_exact_identifier_is_found_by_keyword_search(code_library, database):
    hybrid = PgVectorRetriever(get_session_factory(), get_embedding_provider(), hybrid=True, keyword_top_k=8)
    results = await hybrid.search(code_library["user_id"], "Show me invoice INV-129", top_k=3, rerank_top_k=1)
    assert [chunk.filename for chunk in results] == ["Invoice_129.txt"]
    assert "keyword" in results[0].matched_by and results[0].keyword_score > 0

    by_order = await hybrid.search(code_library["user_id"], "Which invoice belongs to PO-4501?", top_k=3, rerank_top_k=1)
    assert [chunk.filename for chunk in by_order] == ["Invoice_201.txt"]


async def test_keyword_search_respects_owner_and_filters(code_library, database, user_headers):
    hybrid = PgVectorRetriever(get_session_factory(), get_embedding_provider(), hybrid=True)
    stranger = str(user_uuid_for(user_headers["X-User-Id"] + "-stranger"))
    assert await hybrid.search(stranger, "invoice INV-129", top_k=8, rerank_top_k=4) == []

    only_192 = await hybrid.search(
        code_library["user_id"], "invoice INV-129", top_k=8, rerank_top_k=4,
        filters=SearchFilters(document_ids=[code_library["documents"][192]["id"]]),
    )
    assert {chunk.filename for chunk in only_192} == {"Invoice_192.txt"}


async def test_hybrid_can_be_switched_off(code_library, database):
    vector_only = PgVectorRetriever(get_session_factory(), get_embedding_provider(), hybrid=False)
    results = await vector_only.search(code_library["user_id"], "Show me invoice INV-129", top_k=3, rerank_top_k=3)
    assert results and all(chunk.matched_by == ["vector"] and chunk.keyword_score == 0 for chunk in results)


async def test_hostile_query_text_is_harmless(code_library, database):
    hybrid = PgVectorRetriever(get_session_factory(), get_embedding_provider(), hybrid=True)
    results = await hybrid.search(
        code_library["user_id"], "invoice'); DROP TABLE documents; -- \" | & ! :* <->", top_k=4, rerank_top_k=2
    )
    assert len(results) == 2
    assert await hybrid.search(code_library["user_id"], "?? !! ::", top_k=4, rerank_top_k=2)  # no keywords: vector only
