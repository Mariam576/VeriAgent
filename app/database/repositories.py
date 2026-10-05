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
    DocumentForensicsRow,
    Finding,
    Investigation,
    Message,
    RefreshToken,
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


# ------------------------------------------------------------------ accounts
async def get_user(session: AsyncSession, user_id: uuid.UUID) -> User | None:
    user = await session.get(User, user_id)
    await session.commit()  # hand the connection back (see get_or_create_user)
    return user


async def get_user_by_email(session: AsyncSession, email: str) -> User | None:
    result = await session.execute(select(User).where(User.email == email.strip().lower()))
    return result.scalar_one_or_none()


async def get_user_by_google_sub(session: AsyncSession, google_sub: str) -> User | None:
    result = await session.execute(select(User).where(User.google_sub == google_sub))
    return result.scalar_one_or_none()


async def create_account(
    session: AsyncSession, *, email: str, name: str, password_hash: str | None = None,
    google_sub: str | None = None, email_verified: bool = False,
) -> User | None:
    """Create an account. Returns None when the email address is already taken.

    The unique index on ``email`` decides, not a read beforehand, so two
    registrations at the same moment cannot both succeed.
    """
    statement = (
        pg_insert(User)
        .values(
            id=uuid.uuid4(), email=email.strip().lower(), name=name.strip()[:120], password_hash=password_hash,
            google_sub=google_sub, email_verified=email_verified, is_active=True,
        )
        .on_conflict_do_nothing()
        .returning(User.id)
    )
    user_id = (await session.execute(statement)).scalar_one_or_none()
    await session.commit()
    return await session.get(User, user_id) if user_id else None


async def store_refresh_token(
    session: AsyncSession, user_id: uuid.UUID, token_hash: str, expires_at: Any, family_id: uuid.UUID | None = None
) -> RefreshToken:
    token = RefreshToken(user_id=user_id, token_hash=token_hash, expires_at=expires_at, family_id=family_id or uuid.uuid4())
    session.add(token)
    await session.commit()
    return token


async def use_refresh_token(session: AsyncSession, token_hash: str) -> tuple[RefreshToken | None, bool]:
    """Spend a refresh token. Returns (token, ok).

    ``ok`` is True only for exactly one caller: the UPDATE that marks the
    token revoked has ``revoked_at IS NULL`` in its WHERE clause, so if two
    requests arrive with the same token, the database lets one through.
    A token that exists but was already spent is returned with ok=False,
    which is the sign that a copy of it is in someone else's hands.
    """
    token = (await session.execute(select(RefreshToken).where(RefreshToken.token_hash == token_hash))).scalar_one_or_none()
    if token is None:
        await session.commit()
        return None, False
    result = await session.execute(
        update(RefreshToken)
        .where(RefreshToken.id == token.id, RefreshToken.revoked_at.is_(None), RefreshToken.expires_at > utcnow())
        .values(revoked_at=utcnow())
    )
    await session.commit()
    return token, result.rowcount == 1


async def revoke_token_family(session: AsyncSession, family_id: uuid.UUID) -> None:
    await session.execute(
        update(RefreshToken)
        .where(RefreshToken.family_id == family_id, RefreshToken.revoked_at.is_(None))
        .values(revoked_at=utcnow())
    )
    await session.commit()


async def revoke_all_refresh_tokens(session: AsyncSession, user_id: uuid.UUID) -> None:
    """Sign the user out everywhere (after a password change, or when an account is taken back)."""
    await session.execute(
        update(RefreshToken)
        .where(RefreshToken.user_id == user_id, RefreshToken.revoked_at.is_(None))
        .values(revoked_at=utcnow())
    )
    await session.commit()


async def end_access_tokens(session: AsyncSession, user_id: uuid.UUID) -> None:
    """Make every access token issued so far for this account worthless, at once.

    Other devices notice at their next request, exchange their refresh cookie
    for a new access token and carry on - unless their refresh token was
    revoked as well, in which case they are signed out.
    """
    await session.execute(update(User).where(User.id == user_id).values(token_version=User.token_version + 1))
    await session.commit()


async def delete_expired_refresh_tokens(session: AsyncSession) -> int:
    result = await session.execute(delete(RefreshToken).where(RefreshToken.expires_at < utcnow()))
    await session.commit()
    return result.rowcount


async def count_recent_investigations(session: AsyncSession, user_id: uuid.UUID, since: Any) -> int:
    count = await session.scalar(
        select(func.count(Investigation.id)).where(Investigation.user_id == user_id, Investigation.created_at >= since)
    )
    return int(count or 0)


async def delete_user(session: AsyncSession, user_id: uuid.UUID) -> None:
    """Delete an account and, through ON DELETE CASCADE, everything that belongs to it."""
    await session.execute(delete(User).where(User.id == user_id))
    await session.commit()


# ----------------------------------------------------------------- documents
async def count_documents(session: AsyncSession, user_id: uuid.UUID) -> int:
    count = await session.scalar(select(func.count(Document.id)).where(Document.user_id == user_id))
    return int(count or 0)


async def count_chunks(session: AsyncSession, document_id: uuid.UUID) -> int:
    count = await session.scalar(select(func.count(DocumentChunk.id)).where(DocumentChunk.document_id == document_id))
    return int(count or 0)


async def integrity_levels(session: AsyncSession, document_ids: list[uuid.UUID]) -> dict[uuid.UUID, tuple[str, float]]:
    """{document id: (level, score)} of the forensic reports that exist for these documents."""
    if not document_ids:
        return {}
    result = await session.execute(
        select(DocumentForensicsRow.document_id, DocumentForensicsRow.level, DocumentForensicsRow.score)
        .where(DocumentForensicsRow.document_id.in_(document_ids))
    )
    return {document_id: (level, score) for document_id, level, score in result.all()}


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
        .where(
            Investigation.id == investigation_id,
            Investigation.approval_status == "pending",
            Investigation.completed_at.is_(None),  # a failed or expired investigation cannot be decided any more
        )
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
