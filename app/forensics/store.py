"""Where forensic reports are kept and how the investigation gets them.

A report is made once, when the file is uploaded, and stored next to the
document. ``ForensicsSource`` is the small interface the graph uses;
``PgForensicsStore`` implements it on PostgreSQL and ``MemoryForensicsSource``
serves the evaluation and the unit tests.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.database.models import Document, DocumentForensicsRow
from app.forensics.analyze import analyze_document
from app.forensics.model import FORENSICS_VERSION, ForensicReport
from app.rag.loaders import load_pages

logger = logging.getLogger(__name__)


@dataclass
class DocumentForensics:
    document_id: str
    filename: str
    report: ForensicReport


class ForensicsSource(Protocol):
    async def reports(self, user_id: str, document_ids: list[str]) -> list[DocumentForensics]:
        """The forensic reports of the given documents (documents without one are left out)."""
        ...


def _analyse_stored_file(path: Path, document_type: str) -> ForensicReport | None:
    try:
        data = path.read_bytes()
        return analyze_document(data, path.suffix.lower(), document_type, load_pages(data, path.suffix.lower()))
    except Exception as exc:  # noqa: BLE001
        logger.warning("stored file %s could not be inspected: %s", path.name, exc)
        return None


class PgForensicsStore:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession], upload_dir: str) -> None:
        self._session_factory = session_factory
        self._upload_dir = Path(upload_dir)

    async def reports(self, user_id: str, document_ids: list[str]) -> list[DocumentForensics]:
        try:
            wanted = [uuid.UUID(str(document_id)) for document_id in dict.fromkeys(document_ids)]
            owner = uuid.UUID(str(user_id))
        except ValueError:
            return []
        if not wanted:
            return []
        async with self._session_factory() as session:
            rows = (await session.execute(
                select(Document, DocumentForensicsRow)
                .outerjoin(DocumentForensicsRow, DocumentForensicsRow.document_id == Document.id)
                .where(Document.id.in_(wanted), Document.user_id == owner, Document.status == "ready")  # ownership
            )).all()
            found: dict[uuid.UUID, DocumentForensics] = {}
            for document, stored in rows:
                if stored is not None and stored.version == FORENSICS_VERSION:
                    report = ForensicReport.model_validate(stored.report)
                else:
                    # Uploaded before this feature existed (or by an older version of it): inspect the stored file now.
                    files = sorted((self._upload_dir / str(document.user_id)).glob(f"{document.id}.*"))
                    report = await asyncio.to_thread(_analyse_stored_file, files[0], document.document_type) if files else None
                    if report is None:
                        continue
                    values = dict(version=report.version, score=report.score, level=report.level, report=report.model_dump(mode="json"))
                    await session.execute(
                        insert(DocumentForensicsRow).values(document_id=document.id, **values)
                        .on_conflict_do_update(index_elements=["document_id"], set_=values)
                    )
                    await session.commit()
                found[document.id] = DocumentForensics(str(document.id), document.filename, report)
        return [found[document_id] for document_id in wanted if document_id in found]


class MemoryForensicsSource:
    """Reports held in a dict: {document id: (file name, report)}."""

    def __init__(self, reports: dict[str, tuple[str, ForensicReport]] | None = None) -> None:
        self._reports = reports or {}

    def add(self, document_id: str, filename: str, report: ForensicReport) -> None:
        self._reports[document_id] = (filename, report)

    async def reports(self, user_id: str, document_ids: list[str]) -> list[DocumentForensics]:
        return [
            DocumentForensics(document_id, *self._reports[document_id])
            for document_id in dict.fromkeys(document_ids)
            if document_id in self._reports
        ]


_store: PgForensicsStore | None = None


def get_forensics_store() -> PgForensicsStore:
    global _store
    if _store is None:
        from app.config import get_settings
        from app.database.session import get_session_factory

        _store = PgForensicsStore(get_session_factory(), get_settings().UPLOAD_DIR)
    return _store


def reset_forensics_store() -> None:
    global _store
    _store = None
