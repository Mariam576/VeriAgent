"""POST /chat - ask a question inside a conversation.

Concept - short-term memory:
    Each chat turn is stored in the ``messages`` table. When the next question
    arrives, the last few turns are given to the request-analysis step, which
    rewrites a follow-up such as "What about the payment terms?" into a
    standalone question.

    Checkpointing (see ``app/agents/checkpoint.py``) is a different kind of
    memory: it saves the state of one investigation so that it can pause and
    resume. The conversation itself stays in the ``messages`` table.

``POST /chat`` waits for the answer; ``POST /chat/stream`` sends progress
events while the investigation runs (see ``app/api/sse.py``). Both run the
same code.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.service import stream_investigation
from app.api.investigations import investigation_event
from app.api.sse import sse_response
from app.database import repositories
from app.database.models import User
from app.database.session import get_session_factory
from app.errors import NotFound
from app.schemas.response import ChatRequest, ChatResponse
from app.security.permissions import CurrentUser, DbSession

router = APIRouter(tags=["chat"])

HISTORY_TURNS = 6


async def chat_events(session: AsyncSession, user: User, request: ChatRequest) -> AsyncIterator[dict[str, Any]]:
    """One chat turn as a stream of events; the last one is ``{"type": "answer", "response": ChatResponse}``."""
    if request.conversation_id is None:
        conversation = await repositories.create_conversation(session, user.id)
    else:
        conversation = await repositories.get_conversation(session, user.id, request.conversation_id)
        if conversation is None:
            raise NotFound("Conversation not found.")
    conversation_id = conversation.id

    earlier = await repositories.recent_messages(session, conversation_id, limit=HISTORY_TURNS)
    history = [{"role": message.role, "content": message.content} for message in earlier]

    investigation_id = None
    investigation = stream_investigation(
        session, user=user, question=request.message, conversation_id=conversation_id, history=history
    )
    async with aclosing(investigation) as events:  # closes the run promptly if our caller stops reading
        async for item in events:
            if item["type"] == "started":
                investigation_id = item["investigation"].id
                yield {**item, "conversation_id": conversation_id}
                continue
            if item["type"] != "report":
                yield item
                continue
            report = item["report"]
            answer = report.summary
            if report.human_review.status == "pending":
                answer += " This result is waiting for a human decision before it is final."
            # Stored only after a successful run, so a failed turn does not pollute the history.
            # Stored before the report is sent, so a client that stops reading at the report
            # still has the turn in its conversation.
            await repositories.add_message(session, conversation_id, "user", request.message.strip())
            await repositories.add_message(session, conversation_id, "assistant", answer)
            yield item
            yield {
                "type": "answer",
                "response": ChatResponse(
                    conversation_id=conversation_id,
                    investigation_id=investigation_id,
                    answer=answer,
                    risk_level=report.risk_level,
                    confidence=report.confidence,
                    status=report.status,
                    requires_human_review=report.requires_human_review,
                    approval_status=report.human_review.status,
                    citations=report.sources,
                ),
            }


def chat_event(item: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if item["type"] == "started":
        return "started", {
            "conversation_id": str(item["conversation_id"]),
            "investigation_id": str(item["investigation"].id),
        }
    if item["type"] == "answer":
        return "answer", item["response"].model_dump(mode="json")
    return investigation_event(item)


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, user: CurrentUser, session: DbSession) -> ChatResponse:
    response: ChatResponse | None = None
    async for item in chat_events(session, user, request):
        if item["type"] == "answer":
            response = item["response"]
    assert response is not None
    return response


@router.post("/chat/stream", response_class=StreamingResponse)
async def chat_stream(request: ChatRequest, user: CurrentUser) -> StreamingResponse:
    """The same chat turn as Server-Sent Events.

    Events: ``started`` (conversation and investigation ids), ``progress`` and
    ``tool`` while the agent works, ``report`` (the full structured report)
    and finally ``answer`` (the same body ``POST /chat`` returns).
    """
    session = get_session_factory()()  # the stream outlives the request handler, so it owns its session
    events = chat_events(session, user, request)
    try:
        first = await events.__anext__()  # an unknown conversation or an invalid question fails here, as plain HTTP
    except BaseException:
        await events.aclose()
        await session.close()
        raise

    async def body() -> AsyncIterator[tuple[str, dict[str, Any]]]:
        try:
            yield chat_event(first)
            async for item in events:
                yield chat_event(item)
        finally:
            await events.aclose()
            await session.close()

    return sse_response(body())
