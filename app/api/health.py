"""GET /health - is the service up, and how is it configured?

Only non-secret configuration is returned (provider and model names, never keys).
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.config import get_settings
from app.database.session import check_db

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    version: str
    database: str
    llm_provider: str
    llm_model: str
    embedding_provider: str


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    settings = get_settings()
    db_ok = await check_db()
    return HealthResponse(
        status="ok" if db_ok else "degraded",
        version=settings.APP_VERSION,
        database="ok" if db_ok else "unreachable",
        llm_provider=settings.LLM_PROVIDER,
        llm_model=settings.llm_model_name or "(not set)",
        embedding_provider=settings.EMBEDDING_PROVIDER,
    )
