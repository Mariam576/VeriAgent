"""POST /chat - ask a question inside a conversation.

Concept - short-term memory (Version 1 form):
    Each chat turn is stored in the ``messages`` table. When the next question
    arrives, the last few turns are given to the request-analysis step, which
    rewrites a follow-up such as "What about the payment terms?" into a
    standalone question.

    Checkpointing (see ``app/agents/checkpoint.py``) is a different kind of
    memory: it saves the state of one investigation so that it can pause and
    resume. The conversation itself stays in the ``messages`` table.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.agents.service import run_investigation
from app.database import repositories
from app.errors import NotFound
from app.schemas.response import ChatRequest, ChatResponse
from app.security.permissions import CurrentUser, DbSession

router = APIRouter(tags=["chat"])

HISTORY_TURNS = 6


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, user: CurrentUser, session: DbSession) -> ChatResponse:
    if request.conversation_id is None:
        conversation = await repositories.create_conversation(session, user.id)
    else:
        conversation = await repositories.get_conversation(session, user.id, request.conversation_id)
        if conversation is None:
            raise NotFound("Conversation not found.")
    conversation_id = conversation.id

    earlier = await repositories.recent_messages(session, conversation_id, limit=HISTORY_TURNS)
    history = [{"role": message.role, "content": message.content} for message in earlier]

    investigation, report = await run_investigation(
        session, user=user, question=request.message, conversation_id=conversation_id, history=history
    )
    investigation_id = investigation.id

    # Stored only after a successful run, so a failed turn does not pollute the history.
    await repositories.add_message(session, conversation_id, "user", request.message.strip())
    answer = report.summary
    if report.human_review.status == "pending":
        answer += " This result is waiting for a human decision before it is final."
    await repositories.add_message(session, conversation_id, "assistant", answer)

    return ChatResponse(
        conversation_id=conversation_id,
        investigation_id=investigation_id,
        answer=answer,
        risk_level=report.risk_level,
        confidence=report.confidence,
        status=report.status,
        requires_human_review=report.requires_human_review,
        approval_status=report.human_review.status,
        citations=report.sources,
    )
