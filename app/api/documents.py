"""Document endpoints: upload a file, list your files, read a file's key facts, delete one."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, Response, UploadFile

from app.config import get_settings
from app.database import repositories
from app.embeddings.factory import get_embedding_provider
from app.errors import FileTooLarge, NotFound
from app.rag.ingestion import ingest_document
from app.comparison.profile import PROFILED_TYPES
from app.comparison.service import NOT_PROFILED
from app.comparison.store import get_profile_store
from app.forensics.store import get_forensics_store
from app.schemas.comparison import ProfileOut
from app.schemas.document import DocumentList, DocumentOut, ForensicsOut
from app.security.permissions import CurrentUser, DbSession
from app.security.validation import validate_document_type

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentOut, status_code=201)
async def upload_document(
    user: CurrentUser,
    session: DbSession,
    file: Annotated[UploadFile, File(description="PDF, DOCX or TXT")],
    document_type: Annotated[
        str | None, Form(description="Optional: contract, invoice, policy, ... Guessed when omitted.")
    ] = None,
) -> DocumentOut:
    settings = get_settings()
    # Read at most one byte more than the limit, so a huge upload is rejected
    # without loading all of it into memory.
    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise FileTooLarge(f"The file is larger than the {settings.MAX_UPLOAD_MB} MB limit.")

    result = await ingest_document(
        session,
        user=user,
        filename=file.filename,
        data=data,
        document_type=validate_document_type(document_type),
        settings=settings,
        embedder=get_embedding_provider(),
    )
    out = DocumentOut.model_validate(result.document)
    out.chunk_count = result.chunk_count
    out.duplicate = result.duplicate
    levels = await repositories.integrity_levels(session, [result.document.id])
    if result.document.id in levels:
        out.integrity_level, out.integrity_score = levels[result.document.id]
    return out


@router.get("", response_model=DocumentList)
async def list_documents(user: CurrentUser, session: DbSession) -> DocumentList:
    rows = await repositories.list_documents(session, user.id)
    levels = await repositories.integrity_levels(session, [document.id for document, _ in rows])
    documents = []
    for document, chunk_count in rows:
        out = DocumentOut.model_validate(document)
        out.chunk_count = chunk_count
        if document.id in levels:
            out.integrity_level, out.integrity_score = levels[document.id]
        documents.append(out)
    return DocumentList(documents=documents)


@router.get("/{document_id}/profile", response_model=ProfileOut)
async def document_profile(document_id: uuid.UUID, user: CurrentUser, session: DbSession) -> ProfileOut:
    """The key facts of one document: numbers, parties, dates, totals and line items, each with page and quote.

    Built the first time it is asked for (one model call) and stored after that.
    """
    document = await repositories.get_document(session, user.id, document_id)
    if document is None or document.status != "ready":
        raise NotFound("Document not found.")
    out = ProfileOut(document_id=str(document.id), filename=document.filename, document_type=document.document_type)
    if document.document_type not in PROFILED_TYPES:
        out.note = NOT_PROFILED
        return out
    user_id = str(user.id)
    await session.rollback()  # release the connection while the model works
    batch = await get_profile_store().profiles(user_id, [str(document_id)])
    if batch.profiles:
        out.profile = batch.profiles[0].model_dump(mode="json")
        if batch.profiles[0].is_empty:
            out.note = "None of the usual key facts could be read from this document."
    return out


@router.get("/{document_id}/forensics", response_model=ForensicsOut)
async def document_forensics(document_id: uuid.UUID, user: CurrentUser, session: DbSession) -> ForensicsOut:
    """Signs that the file was altered: hidden or covered text, odd fonts, file history.

    A risk indicator, never proof. The file is inspected once, when it is uploaded.
    """
    document = await repositories.get_document(session, user.id, document_id)
    if document is None or document.status != "ready":
        raise NotFound("Document not found.")
    out = ForensicsOut(document_id=str(document.id), filename=document.filename)
    user_id = str(user.id)
    await session.rollback()
    reports = await get_forensics_store().reports(user_id, [str(document_id)])
    if reports:
        out.report = reports[0].report
    else:
        out.note = "This file was not inspected (the inspection is switched off, or the file is no longer stored)."
    return out


def _remove_stored_file(upload_dir: str, user_id: uuid.UUID, document_id: uuid.UUID) -> None:
    # The stored name is "<document id>.<extension>", so this can only ever match that one file.
    for path in (Path(upload_dir) / str(user_id)).glob(f"{document_id}.*"):
        path.unlink(missing_ok=True)


@router.delete("/{document_id}", status_code=204)
async def delete_document(document_id: uuid.UUID, user: CurrentUser, session: DbSession) -> Response:
    """Delete one of your documents: its text chunks, its vectors and the stored file.

    Reports of earlier investigations keep their text; they just no longer link to the document.
    """
    document = await repositories.get_document(session, user.id, document_id)
    if document is None:
        raise NotFound("Document not found.")  # same answer whether it does not exist or belongs to someone else
    await repositories.delete_document(session, document)
    await asyncio.to_thread(_remove_stored_file, get_settings().UPLOAD_DIR, user.id, document_id)
    return Response(status_code=204)
