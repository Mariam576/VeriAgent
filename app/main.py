"""FastAPI application entry point.

Run locally:  uvicorn app.main:app --reload
Run in Docker: docker compose up --build
Interactive API docs: http://localhost:8000/docs
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.agents.checkpoint import close_checkpointer, get_checkpointer
from app.agents.service import reset_graph
from app.api import chat, documents, health, investigations
from app.config import get_settings
from app.database.session import dispose_engine, init_db
from app.embeddings.base import EmbeddingError
from app.errors import AppError
from app.llm.base import LLMConfigurationError, LLMError
from app.logging_config import configure_logging
from app.security.guardrails import redact_secrets

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Runs once at startup (before ``yield``) and once at shutdown (after)."""
    settings = get_settings()
    configure_logging(settings.LOG_LEVEL)
    await init_db()
    logger.info(
        "%s %s started (llm=%s, embeddings=%s)",
        settings.APP_NAME,
        settings.APP_VERSION,
        settings.LLM_PROVIDER,
        settings.EMBEDDING_PROVIDER,
    )
    await get_checkpointer()  # opens its connection pool and creates the checkpoint tables
    yield
    reset_graph()
    await close_checkpointer()
    await dispose_engine()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "Adversarial evidence-verification agent: retrieves evidence across business "
            "documents, detects contradictions and missing support, calculates with "
            "deterministic tools and returns a cited risk report."
        ),
        lifespan=lifespan,
    )

    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    @app.exception_handler(LLMError)
    async def handle_llm_error(_: Request, exc: LLMError) -> JSONResponse:
        code = "llm_not_configured" if isinstance(exc, LLMConfigurationError) else "llm_unavailable"
        message = redact_secrets(str(exc), get_settings())  # provider errors must never leak a key
        return JSONResponse(status_code=503, content={"error": {"code": code, "message": message}})

    @app.exception_handler(EmbeddingError)
    async def handle_embedding_error(_: Request, exc: EmbeddingError) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={"error": {"code": "embeddings_unavailable", "message": redact_secrets(str(exc), get_settings())}},
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        # Log everything, tell the client nothing internal.
        logger.exception("Unhandled error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal_error", "message": "Something went wrong on the server."}},
        )

    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(investigations.router)
    app.include_router(chat.router)
    return app


app = create_app()
