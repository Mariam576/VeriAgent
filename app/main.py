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

from app.api import health
from app.config import get_settings
from app.database.session import dispose_engine, init_db
from app.errors import AppError
from app.logging_config import configure_logging

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
    yield
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

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        # Log everything, tell the client nothing internal.
        logger.exception("Unhandled error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal_error", "message": "Something went wrong on the server."}},
        )

    app.include_router(health.router)
    return app


app = create_app()
