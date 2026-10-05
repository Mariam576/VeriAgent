"""The document upload pipeline (spec section 12).

    validate -> save metadata -> extract text (keep page numbers) -> clean
    -> chunk -> embed -> store chunks in pgvector
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.database import repositories
from app.database.models import Document, DocumentChunk, User
from app.embeddings.base import EmbeddingError, EmbeddingProvider
from app.errors import AppError, DocumentProcessingError, ServiceUnavailable
from app.rag.chunking import chunk_pages
from app.rag.loaders import guess_document_type, guess_year, load_pages
from app.security.validation import validate_upload

logger = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 64


@dataclass
class IngestionResult:
    document: Document
    chunk_count: int
    duplicate: bool = False


def _write_file(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


async def ingest_document(
    session: AsyncSession,
    *,
    user: User,
    filename: str | None,
    data: bytes,
    document_type: str | None,
    settings: Settings,
    embedder: EmbeddingProvider,
) -> IngestionResult:
    # 1. Validate (extension allowlist, size limit, real file format).
    safe_name, extension = validate_upload(filename, data, settings)

    # 2. Same file uploaded before? Do not pay for embeddings twice.
    file_hash = hashlib.sha256(data).hexdigest()
    existing = await repositories.get_document_by_hash(session, user.id, file_hash)
    if existing is not None:
        if existing.status == "ready" and existing.embedding_model == embedder.model_id:
            chunk_counts = dict((doc.id, count) for doc, count in await repositories.list_documents(session, user.id))
            return IngestionResult(existing, chunk_counts.get(existing.id, 0), duplicate=True)
        # A failed attempt, or vectors from a different embedding model: start again.
        await repositories.delete_document(session, existing)

    # 3. Save metadata first, so a failure is visible in GET /documents.
    document = Document(
        id=uuid.uuid4(),
        user_id=user.id,
        filename=safe_name,
        document_type=document_type or "other",
        file_hash=file_hash,
        status="processing",
        embedding_model=embedder.model_id,
    )
    session.add(document)
    await session.commit()
    document_id = document.id
    # The stored path uses the generated id, never the user's file name.
    path = Path(settings.UPLOAD_DIR) / str(user.id) / f"{document_id}{extension}"

    try:
        await asyncio.to_thread(_write_file, path, data)

        # 4. Extract and clean text, page by page. Parsing is CPU work, so it
        #    runs in a worker thread instead of blocking the event loop.
        pages = await asyncio.to_thread(load_pages, data, extension)
        full_text = "\n".join(page.text for page in pages)
        if document_type is None:
            document.document_type = guess_document_type(safe_name, pages[0].text if pages else "")
        year = guess_year(full_text)

        # 5. Chunk.
        chunks = chunk_pages(pages, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
        if not chunks:
            raise DocumentProcessingError("The document contains no text to index.")

        # 6. Embed, in batches.
        vectors: list[list[float]] = []
        for start in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = [chunk.text for chunk in chunks[start : start + EMBED_BATCH_SIZE]]
            vectors.extend(await embedder.aembed_documents(batch))

        # 7. Store chunks + vectors + metadata. Source information is never lost.
        for chunk, vector in zip(chunks, vectors, strict=True):
            session.add(
                DocumentChunk(
                    document_id=document.id,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    content=chunk.text,
                    embedding=vector,
                    meta={
                        "document_id": str(document.id),
                        "filename": safe_name,
                        "document_type": document.document_type,
                        "page": chunk.page_number,
                        "chunk_index": chunk.chunk_index,
                        "year": year,
                    },
                )
            )
        document.page_count = len(pages)
        document.status = "ready"
        await session.commit()
        logger.info(
            "document_ingested id=%s type=%s pages=%d chunks=%d",
            document.id, document.document_type, len(pages), len(chunks),
        )
        return IngestionResult(document, len(chunks))

    except Exception as exc:
        await session.rollback()
        if isinstance(exc, EmbeddingError):
            message = f"Embeddings could not be created: {exc}"
            error: AppError = ServiceUnavailable(message)
        elif isinstance(exc, AppError):
            message, error = exc.message, exc
        else:
            logger.exception("Unexpected ingestion failure for document %s", document_id)
            message = "The document could not be processed."
            error = DocumentProcessingError(message)

        # Keep the row (status "failed" + reason) so the user can see what
        # happened, but do not keep a file that could not be used.
        failed = await session.get(Document, document_id)
        if failed is not None:
            failed.status = "failed"
            failed.error = message[:500]
            await session.commit()
        await asyncio.to_thread(path.unlink, True)
        raise error from exc
