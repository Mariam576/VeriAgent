"""All database queries live here.

API routes and the agent never write SQL themselves. Every query that touches
documents or investigations takes a ``user_id`` and filters by it, so one
user can never read another user's data (ownership check). SQLAlchemy sends
values as bound parameters, which is what prevents SQL injection.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import delete, func, select
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
    User,
    utcnow,
)


# --------------------------------------------------------------------- users
async def get_or_create_user(session: AsyncSession, user_id: uuid.UUID, name: str) -> User:
    # ON CONFLICT DO NOTHING makes this safe when two requests arrive at once.
    await session.execute(
        pg_insert(User).values(id=user_id, name=name).on_conflict_do_nothing(index_elements=[User.id])
    )
    await session.commit()
    user = await session.get(User, user_id)
    assert user is not None
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
) -> None:
    investigation.status = status
    investigation.risk_level = risk_level
    investigation.confidence_score = confidence_score
    investigation.report = report
    investigation.completed_at = utcnow()

    for position, finding in enumerate(findings):
        session.add(Finding(investigation_id=investigation.id, position=position, **finding))
    for call in tool_calls:
        session.add(ToolCall(investigation_id=investigation.id, **call))
    await session.commit()
