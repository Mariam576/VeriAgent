"""The document upload pipeline (spec section 12).

    validate -> save metadata -> extract text (keep page numbers) -> clean
    -> inspect the file for signs of editing -> chunk -> embed -> store chunks in pgvector

It has a quick part (``register_document``: validate, write the row) and a
slow part (``process_document``: everything after that). An upload can run
both inside the request, or answer after the quick part and leave the slow
part to a background job (see app/jobs.py).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.database import repositories
from app.database.models import Document, DocumentChunk, DocumentForensicsRow, User
from app.embeddings.base import EmbeddingError, EmbeddingProvider
from app.errors import AppError, DocumentProcessingError, ServiceUnavailable
from app.forensics.analyze import analyze_document
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


def _extract_and_chunk(
    data: bytes, extension: str, filename: str, document_type: str | None, chunk_size: int, chunk_overlap: int,
    inspect: bool,
):
    """All the CPU-bound work on one document, done together in a worker thread."""
    pages = load_pages(data, extension)
    full_text = "\n".join(page.text for page in pages)
    document_type = document_type or guess_document_type(filename, pages[0].text if pages else "")
    forensics = analyze_document(data, extension, document_type, pages) if inspect else None
    return pages, chunk_pages(pages, chunk_size, chunk_overlap), guess_year(full_text), document_type, forensics


@dataclass
class Registered:
    """A file that passed validation and has its row in the database."""

    document: Document
    extension: str
    path: Path
    duplicate: bool = False
    chunk_count: int = 0


async def register_document(
    session: AsyncSession,
    *,
    user: User,
    filename: str | None,
    data: bytes,
    document_type: str | None,
    settings: Settings,
    embedder: EmbeddingProvider,
) -> Registered:
    """The quick part of an upload: validate the file and write its row (status "processing")."""
    # 1. Validate (extension allowlist, size limit, real file format).
    safe_name, extension = validate_upload(filename, data, settings)

    # 2. Same file uploaded before? Do not pay for embeddings twice.
    file_hash = hashlib.sha256(data).hexdigest()
    existing = await repositories.get_document_by_hash(session, user.id, file_hash)
    if existing is not None:
        path = Path(settings.UPLOAD_DIR) / str(user.id) / f"{existing.id}{extension}"
        if existing.status == "ready" and existing.embedding_model == embedder.model_id:
            chunk_counts = dict((doc.id, count) for doc, count in await repositories.list_documents(session, user.id))
            return Registered(existing, extension, path, duplicate=True, chunk_count=chunk_counts.get(existing.id, 0))
        age = datetime.now(timezone.utc) - existing.uploaded_at
        if existing.status == "processing" and age < timedelta(minutes=settings.DOCUMENT_PROCESSING_TIMEOUT_MINUTES):
            return Registered(existing, extension, path, duplicate=True)  # the same upload is still being worked on
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
    # The stored path uses the generated id, never the user's file name.
    return Registered(document, extension, Path(settings.UPLOAD_DIR) / str(user.id) / f"{document.id}{extension}")


async def process_document(
    session: AsyncSession,
    *,
    document_id: uuid.UUID,
    data: bytes,
    extension: str,
    path: Path,
    document_type: str | None,
    settings: Settings,
    embedder: EmbeddingProvider,
) -> IngestionResult:
    """The slow part of an upload: read, inspect, chunk, embed and store.

    Runs inside the request (the default) or as a background job. Either way
    the outcome is written to the document's row: "ready", or "failed" with
    the reason.
    """
    document = await session.get(Document, document_id)
    if document is None:
        raise DocumentProcessingError("The document was deleted before it could be processed.")
    safe_name = document.filename
    try:
        await asyncio.to_thread(_write_file, path, data)

        # 4. Extract and clean text page by page, then 5. chunk it. Parsing is
        #    CPU work, so it runs in a worker thread instead of blocking the event loop.
        pages, chunks, year, kind, forensics = await asyncio.to_thread(
            _extract_and_chunk, data, extension, safe_name, document_type, settings.CHUNK_SIZE,
            settings.CHUNK_OVERLAP, settings.ENABLE_FORENSICS,
        )
        if not chunks:
            raise DocumentProcessingError("The document contains no text to index.")

        # 6. Embed, in batches. (No database connection is held while waiting for the embedding provider.)
        await session.rollback()
        vectors: list[list[float]] = []
        for start in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = [chunk.text for chunk in chunks[start : start + EMBED_BATCH_SIZE]]
            vectors.extend(await embedder.aembed_documents(batch))

        # The document may have been deleted while it was being read (a background job takes a while).
        document = await session.get(Document, document_id)
        if document is None:
            await asyncio.to_thread(path.unlink, True)
            raise DocumentProcessingError("The document was deleted before it could be processed.")
        document.document_type = kind

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
        if forensics is not None:
            session.add(DocumentForensicsRow(
                document_id=document.id, version=forensics.version, score=forensics.score, level=forensics.level,
                report=forensics.model_dump(mode="json"),
            ))
            if forensics.indicators:
                logger.info(
                    "document_forensics id=%s level=%s score=%.2f indicators=%s",
                    document.id, forensics.level, forensics.score, [item.type for item in forensics.indicators],
                )
        document.page_count = len(pages)
        document.status = "ready"
        await session.commit()
        if not settings.KEEP_UPLOADED_FILES:
            # Everything the application needs is in the database now (text, vectors, inspection result).
            await asyncio.to_thread(path.unlink, True)
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
    """Upload and process a document in one go (both parts inside the caller's request)."""
    registered = await register_document(
        session, user=user, filename=filename, data=data, document_type=document_type, settings=settings, embedder=embedder,
    )
    if registered.duplicate:
        return IngestionResult(registered.document, registered.chunk_count, duplicate=True)
    return await process_document(
        session, document_id=registered.document.id, data=data, extension=registered.extension, path=registered.path,
        document_type=document_type, settings=settings, embedder=embedder,
    )


async def process_in_background(
    *, document_id: uuid.UUID, data: bytes, extension: str, path: Path, document_type: str | None,
) -> None:
    """The job behind ``POST /documents/upload?background=true``. Its result is the document's status."""
    from app.config import get_settings
    from app.database.session import get_session_factory
    from app.embeddings.factory import get_embedding_provider

    async with get_session_factory()() as session:
        try:
            await process_document(
                session, document_id=document_id, data=data, extension=extension, path=path,
                document_type=document_type, settings=get_settings(), embedder=get_embedding_provider(),
            )
        except AppError as exc:
            logger.info("background_ingestion_failed id=%s reason=%s", document_id, exc.message)  # already stored on the row
