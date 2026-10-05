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


class Conflict(AppError):
    status_code = 409
    code = "conflict"
