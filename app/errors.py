"""Application errors that map cleanly to HTTP responses.

Raising one of these anywhere in the code produces a short, safe JSON error
for the client. Internal details (stack traces, SQL, keys) stay in the logs.
"""

from __future__ import annotations


class AppError(Exception):
    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ValidationFailed(AppError):
    status_code = 400
    code = "validation_failed"


class UnsupportedFileType(AppError):
    status_code = 415
    code = "unsupported_file_type"


class FileTooLarge(AppError):
    status_code = 413
    code = "file_too_large"


class NotFound(AppError):
    status_code = 404
    code = "not_found"


class DocumentProcessingError(AppError):
    status_code = 422
    code = "document_processing_failed"


class ServiceUnavailable(AppError):
    status_code = 503
    code = "service_unavailable"


class Unauthorized(AppError):
    status_code = 401
    code = "unauthorized"


class Forbidden(AppError):
    status_code = 403
    code = "forbidden"


class TooManyRequests(AppError):
    status_code = 429
    code = "too_many_requests"


class Conflict(AppError):
    status_code = 409
    code = "conflict"


def safe_error(exc: BaseException) -> tuple[int, dict[str, str]]:
    """HTTP status and a {"code", "message"} that is safe to show for ANY error.

    Errors raised on purpose (AppError, LLM and embedding errors) carry a
    message written for the user. Everything else is unexpected: its text may
    contain SQL, file paths or model output, so only a generic sentence leaves
    the server and the details stay in the log.
    """
    from app.config import get_settings
    from app.embeddings.base import EmbeddingError
    from app.llm.base import LLMConfigurationError, LLMError
    from app.security.guardrails import redact_secrets

    settings = get_settings()
    if isinstance(exc, AppError):
        return exc.status_code, {"code": exc.code, "message": exc.message}
    if isinstance(exc, LLMError):
        code = "llm_not_configured" if isinstance(exc, LLMConfigurationError) else "llm_unavailable"
        return 503, {"code": code, "message": redact_secrets(str(exc), settings)}  # provider errors must never leak a key
    if isinstance(exc, EmbeddingError):
        return 503, {"code": "embeddings_unavailable", "message": redact_secrets(str(exc), settings)}
    return 500, {"code": "internal_error", "message": "Something went wrong on the server."}
