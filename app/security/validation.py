"""Input validation that does not depend on any LLM.

Rule of thumb used across the project: anything that can be checked with
plain code is checked with plain code. The LLM is never the only safety net.
"""

from __future__ import annotations

import io
import re
import unicodedata
import zipfile
from pathlib import PurePosixPath, PureWindowsPath

from app.config import Settings
from app.errors import FileTooLarge, UnsupportedFileType, ValidationFailed

DOCUMENT_TYPES = (
    "contract",
    "invoice",
    "policy",
    "purchase_order",
    "quotation",
    "receipt",
    "financial_statement",
    "report",
    "other",
)

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Letters and digits of any language are kept (so Urdu, Arabic or Chinese file
# names survive); quotes, slashes, control characters and symbols are replaced.
_UNSAFE_FILENAME_CHARS = re.compile(r"[^\w._ ()\-]", re.UNICODE)
UTF16_BOMS = (b"\xff\xfe", b"\xfe\xff")
# A .docx is a zip archive. Refuse one that would unpack to far more than it weighs.
MAX_DOCX_UNPACKED_BYTES = 200 * 1024 * 1024


def sanitize_filename(filename: str | None) -> str:
    """Keep only the base name and safe characters.

    The stored file never uses this name for its path (a random id is used),
    so this is for display. It still strips directories so that a name like
    ``../../etc/passwd`` can never be shown or logged as a path.
    """
    raw = (filename or "").strip()
    base = PureWindowsPath(PurePosixPath(raw).name).name
    base = unicodedata.normalize("NFKC", base)
    base = _UNSAFE_FILENAME_CHARS.sub("_", base).strip(" .")
    return base[:200] or "document"


def file_extension(filename: str) -> str:
    return PurePosixPath(filename).suffix.lower()


def validate_upload(filename: str | None, data: bytes, settings: Settings) -> tuple[str, str]:
    """Check an uploaded file. Returns ``(safe_filename, extension)`` or raises."""
    safe_name = sanitize_filename(filename)
    extension = file_extension(safe_name)

    if extension not in settings.allowed_extensions:
        allowed = ", ".join(sorted(settings.allowed_extensions))
        raise UnsupportedFileType(f"File type '{extension or 'unknown'}' is not allowed. Allowed: {allowed}.")
    if not data:
        raise ValidationFailed("The uploaded file is empty.")
    if len(data) > settings.max_upload_bytes:
        raise FileTooLarge(f"The file is larger than the {settings.MAX_UPLOAD_MB} MB limit.")

    # The extension is only a claim. Check that the content really is that format.
    if extension == ".pdf" and not data.lstrip()[:5].startswith(b"%PDF-"):
        raise UnsupportedFileType("The file has a .pdf extension but is not a PDF.")
    if extension == ".docx":
        if not data.startswith(b"PK"):
            raise UnsupportedFileType("The file has a .docx extension but is not a Word document.")
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                unpacked = sum(member.file_size for member in archive.infolist())
        except zipfile.BadZipFile as exc:
            raise UnsupportedFileType("The file has a .docx extension but is not a Word document.") from exc
        if unpacked > MAX_DOCX_UNPACKED_BYTES:
            raise FileTooLarge("The Word document is too large when unpacked.")
    # UTF-16 text (what Windows PowerShell writes by default) legitimately contains zero bytes.
    if extension == ".txt" and not data.startswith(UTF16_BOMS) and b"\x00" in data[:4096]:
        raise UnsupportedFileType("The file has a .txt extension but contains binary data.")

    return safe_name, extension


def validate_document_type(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    cleaned = value.strip().lower().replace(" ", "_").replace("-", "_")
    if cleaned not in DOCUMENT_TYPES:
        raise ValidationFailed(f"document_type must be one of: {', '.join(DOCUMENT_TYPES)}.")
    return cleaned


def validate_question(question: str | None, settings: Settings) -> str:
    """Clean a user question and enforce the length limit."""
    cleaned = _CONTROL_CHARS.sub(" ", question or "").strip()
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    if not cleaned:
        raise ValidationFailed("The question is empty.")
    if len(cleaned) > settings.MAX_QUESTION_CHARS:
        raise ValidationFailed(f"The question is longer than {settings.MAX_QUESTION_CHARS} characters.")
    return cleaned
