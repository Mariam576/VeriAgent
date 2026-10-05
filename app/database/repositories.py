"""All database queries live here.

API routes and the agent never write SQL themselves. Every query that touches
documents or investigations takes a ``user_id`` and filters by it, so one
user can never read another user's data (ownership check). SQLAlchemy sends
values as bound parameters, which is what prevents SQL injection.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import (
    Conversation,
    Document,
    DocumentChunk,
    Finding,
    Investigation,
    Message,
    ToolCall,
    TraceEvent,
    User,
    utcnow,
)


# --------------------------------------------------------------------- users
async def get_or_create_user(session: AsyncSession, user_id: uuid.UUID, name: str) -> User:
    # ON CONFLICT DO NOTHING makes this safe when two requests arrive at once.
    await session.execute(
        pg_insert(User).values(id=user_id, name=name).on_conflict_do_nothing(index_elements=[User.id])
    )
    user = await session.get(User, user_id)
    assert user is not None
    # Commit last: that hands the database connection back to the pool. Reading
    # after the commit would keep one connection checked out for the whole request,
    # and requests that need a second one (memory, streaming) could then starve the pool.
    await session.commit()
    return user


# ----------------------------------------------------------------- documents
async def get_document_by_hash(session: AsyncSession, user_id: uuid.UUID, file_hash: str) -> Document | None:
    result = await session.execute(
        select(Document).where(Document.user_id == user_id, Document.file_hash == file_hash)
    )
    return result.scalar_one_or_none()


async def get_document(session: AsyncSession, user_id: uuid.UUID, document_id: uuid.UUID) -> Document | None:
    result = await session.execute(
        select(Document).where(Document.id == document_id, Document.user_id == user_id)
    )
    return result.scalar_one_or_none()


async def list_documents(session: AsyncSession, user_id: uuid.UUID) -> list[tuple[Document, int]]:
    """Documents of one user, newest first, each with its number of chunks."""
    chunk_count = (
        select(DocumentChunk.document_id, func.count(DocumentChunk.id).label("chunks"))
        .group_by(DocumentChunk.document_id)
        .subquery()
    )
    result = await session.execute(
        select(Document, func.coalesce(chunk_count.c.chunks, 0))
        .outerjoin(chunk_count, chunk_count.c.document_id == Document.id)
        .where(Document.user_id == user_id)
        .order_by(Document.uploaded_at.desc())
    )
    return [(row[0], int(row[1])) for row in result.all()]


async def owned_document_ids(
    session: AsyncSession, user_id: uuid.UUID, document_ids: list[uuid.UUID]
) -> set[uuid.UUID]:
    if not document_ids:
        return set()
    result = await session.execute(
        select(Document.id).where(Document.user_id == user_id, Document.id.in_(document_ids))
    )
    return set(result.scalars().all())


async def delete_document(session: AsyncSession, document: Document) -> None:
    await session.execute(delete(Document).where(Document.id == document.id))
    await session.commit()


# ------------------------------------------------------------- conversations
async def create_conversation(session: AsyncSession, user_id: uuid.UUID) -> Conversation:
    conversation = Conversation(user_id=user_id)
    session.add(conversation)
    await session.commit()
    return conversation


async def get_conversation(
    session: AsyncSession, user_id: uuid.UUID, conversation_id: uuid.UUID
) -> Conversation | None:
    result = await session.execute(
        select(Conversation).where(Conversation.id == conversation_id, Conversation.user_id == user_id)
    )
    return result.scalar_one_or_none()


async def add_message(session: AsyncSession, conversation_id: uuid.UUID, role: str, content: str) -> Message:
    message = Message(conversation_id=conversation_id, role=role, content=content)
    session.add(message)
    conversation = await session.get(Conversation, conversation_id)
    if conversation is not None:
        conversation.updated_at = utcnow()
    await session.commit()
    return message


async def recent_messages(session: AsyncSession, conversation_id: uuid.UUID, limit: int = 6) -> list[Message]:
    result = await session.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
    )
    return list(reversed(result.scalars().all()))


# ------------------------------------------------------------ investigations
async def create_investigation(
    session: AsyncSession, user_id: uuid.UUID, question: str, conversation_id: uuid.UUID | None
) -> Investigation:
    investigation = Investigation(user_id=user_id, question=question, conversation_id=conversation_id)
    session.add(investigation)
    await session.commit()
    return investigation


async def get_investigation(
    session: AsyncSession, user_id: uuid.UUID, investigation_id: uuid.UUID
) -> Investigation | None:
    result = await session.execute(
        select(Investigation).where(Investigation.id == investigation_id, Investigation.user_id == user_id)
    )
    return result.scalar_one_or_none()


async def list_investigations(session: AsyncSession, user_id: uuid.UUID, limit: int = 50) -> list[Investigation]:
    result = await session.execute(
        select(Investigation)
        .where(Investigation.user_id == user_id)
        .order_by(Investigation.created_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


async def list_findings(session: AsyncSession, investigation_id: uuid.UUID) -> list[Finding]:
    result = await session.execute(
        select(Finding).where(Finding.investigation_id == investigation_id).order_by(Finding.position)
    )
    return list(result.scalars().all())


async def save_investigation_result(
    session: AsyncSession,
    investigation: Investigation,
    *,
    status: str,
    risk_level: str | None,
    confidence_score: float | None,
    report: dict[str, Any] | None,
    findings: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
    approval_status: str = "not_required",
    approval_note: str | None = None,
    completed: bool = True,
) -> None:
    """Store the outcome of a run.

    An investigation can be saved twice: once as a draft when it pauses for a
    human decision (``completed=False``) and once when it is final. The second
    save replaces the findings and tool calls of the first.
    """
    investigation.status = status
    investigation.error = None
    investigation.risk_level = risk_level
    investigation.confidence_score = confidence_score
    investigation.report = report
    investigation.approval_status = approval_status
    investigation.approval_note = approval_note
    if completed:
        investigation.completed_at = utcnow()
        if approval_status in ("approved", "rejected", "modified"):
            investigation.approved_at = utcnow()

    # A cited document may have been deleted since the evidence was read (for example
    # while the investigation waited for a decision). The finding keeps its text; only
    # the link to the document is dropped, because the row it pointed to is gone.
    cited = {finding["source_document_id"] for finding in findings if finding.get("source_document_id")}
    existing: set[uuid.UUID] = set()
    if cited:
        result = await session.execute(select(Document.id).where(Document.id.in_(cited)))
        existing = set(result.scalars().all())

    await session.execute(delete(Finding).where(Finding.investigation_id == investigation.id))
    await session.execute(delete(ToolCall).where(ToolCall.investigation_id == investigation.id))
    for position, finding in enumerate(findings):
        if finding.get("source_document_id") not in existing:
            finding = {**finding, "source_document_id": None}
        session.add(Finding(investigation_id=investigation.id, position=position, **finding))
    for call in tool_calls:
        session.add(ToolCall(investigation_id=investigation.id, **call))
    await session.commit()


async def mark_investigation_failed(session: AsyncSession, investigation_id: uuid.UUID, reason: str | None = None) -> None:
    await session.execute(
        update(Investigation)
        .where(Investigation.id == investigation_id)
        .values(status="failed", error=reason, completed_at=utcnow())
    )
    await session.commit()


async def mark_investigation_interrupted(session: AsyncSession, investigation_id: uuid.UUID, reason: str) -> None:
    """A run stopped part-way but its checkpoint is intact: POST /investigations/{id}/resume continues it."""
    await session.execute(
        update(Investigation)
        .where(Investigation.id == investigation_id, Investigation.completed_at.is_(None))
        .values(status="interrupted", error=reason)
    )
    await session.commit()


async def claim_interrupted(session: AsyncSession, investigation_id: uuid.UUID) -> bool:
    """Take the right to resume an interrupted investigation (atomic, like claim_pending_approval)."""
    result = await session.execute(
        update(Investigation)
        .where(Investigation.id == investigation_id, Investigation.status == "interrupted")
        .values(status="running", error=None)
    )
    await session.commit()
    return result.rowcount == 1


async def set_approval_status(session: AsyncSession, investigation_id: uuid.UUID, status: str) -> None:
    await session.execute(
        update(Investigation).where(Investigation.id == investigation_id).values(approval_status=status)
    )
    await session.commit()


async def claim_pending_approval(session: AsyncSession, investigation_id: uuid.UUID) -> bool:
    """Take the right to decide a paused investigation. Returns False if it is not waiting.

    One UPDATE with the condition in its WHERE clause is atomic: when two
    people click "Approve" and "Reject" at the same moment, the database lets
    exactly one of them change ``pending`` to ``deciding``. The other request
    updates no row and is told that the decision was already taken.
    """
    result = await session.execute(
        update(Investigation)
        .where(Investigation.id == investigation_id, Investigation.approval_status == "pending")
        .values(approval_status="deciding")
    )
    await session.commit()
    return result.rowcount == 1


# -------------------------------------------------------------------- traces
async def save_trace(session: AsyncSession, investigation_id: uuid.UUID, events: list[Any]) -> None:
    """Store the events of one run. A later run (after a human decision) gets the next run number."""
    if not events:
        return
    last_run = await session.scalar(
        select(func.max(TraceEvent.run)).where(TraceEvent.investigation_id == investigation_id)
    )
    run = (last_run or 0) + 1
    for event in events:
        session.add(
            TraceEvent(
                investigation_id=investigation_id, run=run, seq=event.seq, kind=event.kind,
                name=str(event.name)[:100], latency_ms=event.latency_ms, offset_ms=event.offset_ms,
                data=event.data,
            )
        )
    await session.commit()


async def list_trace(session: AsyncSession, investigation_id: uuid.UUID) -> list[TraceEvent]:
    result = await session.execute(
        select(TraceEvent)
        .where(TraceEvent.investigation_id == investigation_id)
        .order_by(TraceEvent.run, TraceEvent.seq)
    )
    return list(result.scalars().all())
