"""Compare documents side by side: POST /comparisons."""

from __future__ import annotations

from fastapi import APIRouter

from app.comparison.service import compare_documents
from app.comparison.store import get_profile_store
from app.schemas.comparison import ComparisonOut, ComparisonRequest
from app.security.permissions import CurrentUser, DbSession
from app.security.rate_limit import RateLimiter, check

router = APIRouter(prefix="/comparisons", tags=["comparisons"])

# A comparison can cost one model call per document, so it is limited per user.
comparisons_by_user = RateLimiter(30, 3600)


@router.post("", response_model=ComparisonOut)
async def compare(request: ComparisonRequest, user: CurrentUser, session: DbSession) -> ComparisonOut:
    """Put up to six of your documents side by side and list where they disagree.

    Returns the key facts of each document (with page and quote), their line
    items and the differences found. No question, no search, no risk level:
    for that, start an investigation.
    """
    check(comparisons_by_user, str(user.id), "comparisons")
    return await compare_documents(session, get_profile_store(), user.id, request.document_ids)
