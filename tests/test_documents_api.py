"""Upload pipeline through the real API and a real PostgreSQL + pgvector."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import func, select

from app.database.models import Document, DocumentChunk
from app.database.session import get_session_factory

pytestmark = pytest.mark.integration


async def upload(client, headers, path, **form):
    with path.open("rb") as handle:
        return await client.post(
            "/documents/upload", headers=headers, files={"file": (path.name, handle)}, data=form
        )


async def test_health(client):
    response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok" and body["database"] == "ok"
    assert body["llm_provider"] == "mock"
    assert "key" not in str(body).lower()


async def test_upload_stores_chunks_with_vectors_and_page_metadata(client, user_headers, samples_dir, settings):
    response = await upload(client, user_headers, samples_dir / "Contract.pdf")
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["filename"] == "Contract.pdf"
    assert body["document_type"] == "contract"
    assert body["status"] == "ready"
    assert body["page_count"] == 5
    assert body["chunk_count"] >= 5
    assert body["duplicate"] is False

    async with get_session_factory()() as session:
        result = await session.execute(
            select(DocumentChunk).where(DocumentChunk.document_id == uuid.UUID(body["id"]))
            .order_by(DocumentChunk.chunk_index)
        )
        chunks = result.scalars().all()
    assert len(chunks) == body["chunk_count"]
    assert all(len(chunk.embedding) == settings.EMBEDDING_DIM for chunk in chunks)
    page_4 = [chunk for chunk in chunks if chunk.page_number == 4]
    assert page_4 and "£40,000" in page_4[0].content
    assert page_4[0].meta["filename"] == "Contract.pdf"
    assert page_4[0].meta["page"] == 4
    assert page_4[0].meta["document_type"] == "contract"
    assert page_4[0].meta["year"] == 2026


async def test_same_file_twice_is_not_processed_again(client, user_headers, samples_dir):
    first = await upload(client, user_headers, samples_dir / "Invoice_102.pdf")
    second = await upload(client, user_headers, samples_dir / "Invoice_102.pdf")
    assert first.json()["id"] == second.json()["id"]
    assert second.json()["duplicate"] is True
    assert second.json()["chunk_count"] == first.json()["chunk_count"]

    listing = await client.get("/documents", headers=user_headers)
    assert len(listing.json()["documents"]) == 1


async def test_users_only_see_their_own_documents(client, samples_dir, user_headers):
    other_user = {"X-User-Id": user_headers["X-User-Id"] + "-other"}
    await upload(client, user_headers, samples_dir / "Contract.pdf")

    mine = await client.get("/documents", headers=user_headers)
    theirs = await client.get("/documents", headers=other_user)
    assert [doc["filename"] for doc in mine.json()["documents"]] == ["Contract.pdf"]
    assert theirs.json()["documents"] == []


async def test_document_type_can_be_set_by_the_user(client, user_headers, samples_dir):
    response = await upload(client, user_headers, samples_dir / "Procurement_Policy.pdf", document_type="report")
    assert response.json()["document_type"] == "report"


async def test_rejected_uploads(client, user_headers, tmp_path):
    exe = tmp_path / "tool.exe"
    exe.write_bytes(b"MZ....")
    assert (await upload(client, user_headers, exe)).status_code == 415

    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.write_bytes(b"this is not a pdf")
    assert (await upload(client, user_headers, fake_pdf)).status_code == 415

    empty = tmp_path / "empty.txt"
    empty.write_bytes(b"")
    assert (await upload(client, user_headers, empty)).status_code == 400

    bad_type = tmp_path / "note.txt"
    bad_type.write_text("hello")
    response = await upload(client, user_headers, bad_type, document_type="love_letter")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "validation_failed"


async def test_unreadable_pdf_is_recorded_as_failed(client, user_headers, tmp_path):
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"%PDF-1.7 truncated garbage")
    response = await upload(client, user_headers, broken)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "document_processing_failed"

    listing = (await client.get("/documents", headers=user_headers)).json()["documents"]
    assert listing[0]["status"] == "failed"
    assert listing[0]["chunk_count"] == 0
    assert listing[0]["error"]

    async with get_session_factory()() as session:
        count = await session.scalar(
            select(func.count(DocumentChunk.id)).join(Document).where(Document.id == uuid.UUID(listing[0]["id"]))
        )
    assert count == 0


async def test_invalid_user_header_is_rejected(client):
    response = await client.get("/documents", headers={"X-User-Id": "bad user; drop table"})
    assert response.status_code == 400
