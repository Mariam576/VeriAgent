"""FastAPI application entry point.

Run locally:  uvicorn app.main:app --reload
Run in Docker: docker compose up --build
Interactive API docs: http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import traceback
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.agents.checkpoint import close_checkpointer, get_checkpointer, purge_stale_checkpoints
from app.agents.service import reset_graph
from app.api import actions, auth, chat, comparisons, documents, health, investigations, memory
from app.api.sse import error_payload
from app.config import get_settings
from app.database.session import dispose_engine, init_db
from app.embeddings.base import EmbeddingError
from app.errors import AppError
from app.llm.base import LLMError
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
    await purge_stale_checkpoints()
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
    @app.exception_handler(LLMError)
    @app.exception_handler(EmbeddingError)
    async def handle_known_error(_: Request, exc: Exception) -> JSONResponse:
        status_code, payload = error_payload(exc)
        return JSONResponse(status_code=status_code, content={"error": payload})

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        # Log everything (with secrets removed), tell the client nothing internal.
        details = "".join(traceback.format_exception(exc))
        logger.error("Unhandled error:\n%s", redact_secrets(details, get_settings()))
        status_code, payload = error_payload(exc)
        return JSONResponse(status_code=status_code, content={"error": payload})

    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(documents.router)
    app.include_router(investigations.router)
    app.include_router(comparisons.router)
    app.include_router(actions.router)
    app.include_router(chat.router)
    app.include_router(memory.router)
    return app


app = create_app()
