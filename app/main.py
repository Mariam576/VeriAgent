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
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.agents.checkpoint import close_checkpointer, get_checkpointer
from app.agents.service import reset_graph
from app.api import actions, auth, chat, comparisons, documents, health, investigations, memory
from app.api.sse import error_payload
from app.config import get_settings
from app.database.session import dispose_engine, init_db
from app.embeddings.base import EmbeddingError
from app.errors import AppError
from app.llm.base import LLMError
from app.jobs import get_jobs
from app.logging_config import configure_logging
from app.maintenance import run_maintenance
from app.security.guardrails import redact_secrets
from app.startup import check_configuration
from app.web import mount_interface

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Runs once at startup (before ``yield``) and once at shutdown (after)."""
    settings = get_settings()
    configure_logging(settings.LOG_LEVEL)
    check_configuration(settings)  # refuse to start with settings that cannot work (app/startup.py)
    await init_db()
    logger.info(
        "%s %s started (llm=%s, embeddings=%s)",
        settings.APP_NAME,
        settings.APP_VERSION,
        settings.LLM_PROVIDER,
        settings.EMBEDDING_PROVIDER,
    )
    await get_checkpointer()  # opens its connection pool and creates the checkpoint tables
    await run_maintenance(at_startup=True)
    jobs = get_jobs()
    if settings.MAINTENANCE_INTERVAL_MINUTES > 0:
        jobs.every("maintenance", settings.MAINTENANCE_INTERVAL_MINUTES * 60, run_maintenance)
    yield
    await jobs.shutdown()  # let a document that is being processed finish, within reason
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

    @app.exception_handler(RequestValidationError)
    async def handle_invalid_request(_: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's own answer repeats what was sent - including a password that was too long - and
        # fails altogether when the input cannot be written as text. This one names the field and the
        # problem, and nothing else.
        fields = [
            {"field": ".".join(str(part) for part in error.get("loc", ()) if part != "body"), "problem": str(error.get("msg", "invalid"))[:200]}
            for error in exc.errors()[:10]
        ]
        first = fields[0] if fields else {"field": "", "problem": "invalid"}
        message = f"{first['field'] or 'The request'}: {first['problem']}"
        return JSONResponse(status_code=422, content={"error": {"code": "invalid_request", "message": message, "fields": fields}})

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
    mount_interface(app, settings.FRONTEND_DIR)  # last: files are served only where no API route matched
    return app


app = create_app()
