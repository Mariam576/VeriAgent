"""Document endpoints: upload a file, list your files."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile

from app.config import get_settings
from app.database import repositories
from app.embeddings.factory import get_embedding_provider
from app.errors import FileTooLarge
from app.rag.ingestion import ingest_document
from app.schemas.document import DocumentList, DocumentOut
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
    return out


@router.get("", response_model=DocumentList)
async def list_documents(user: CurrentUser, session: DbSession) -> DocumentList:
    rows = await repositories.list_documents(session, user.id)
    documents = []
    for document, chunk_count in rows:
        out = DocumentOut.model_validate(document)
        out.chunk_count = chunk_count
        documents.append(out)
    return DocumentList(documents=documents)
