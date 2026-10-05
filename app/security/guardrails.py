"""Application-level guardrails (spec sections 35 and 37).

These controls are plain code. They work even if the LLM misbehaves:

* documents are wrapped and labelled as untrusted before the model sees them,
* text that looks like an instruction to an AI is detected and reported,
* secrets are removed from anything that leaves the application.

Hard limits (iterations, retrieved chunks, upload size, allowed tools) live in
``app.config`` and ``app.tools.registry`` and are enforced where they apply.
"""

from __future__ import annotations

import re

from app.config import Settings
from app.schemas.investigation import RetrievedChunk

UNTRUSTED_OPEN = "<<<UNTRUSTED DOCUMENT CONTENT"
UNTRUSTED_CLOSE = "<<<END UNTRUSTED DOCUMENT CONTENT>>>"

# Phrases that are typical of prompt-injection attempts and very unlikely in a
# normal contract, invoice or policy. Each pattern needs wording that addresses
# the reader's *instructions* or an *AI system*; ordinary business sentences
# such as "please disregard the previous invoice" or "the assistant manager
# must approve" must not match (see tests/test_guardrails.py).
# A match never blocks the document: the text is still treated as evidence,
# and the attempt itself is reported as a risk.
_AI = r"(?:ai|a\.i\.|llm|language\s+model|chat\s?bot|automated)\s+(?:assistant|agent|system|model|reviewer|tool)s?"
_INJECTION_PATTERNS = [
    r"(?:ignore|disregard|forget|override)\s+(?:all\s+|any\s+|the\s+|your\s+|these\s+|those\s+)?(?:previous|prior|above|earlier|preceding|system)\s+(?:instructions?|prompts?|rules?|guidelines|directions)",
    r"(?:ignore|disregard|override)\s+(?:the\s+|your\s+)?system\s+(?:prompt|message)",
    r"(?<!not\s)(?<!never\s)(?:reveal|print|send|leak|exfiltrate|email|forward)\b[^.\n]{0,40}\b(?:system\s+prompt|api[\s_-]?keys?)",
    r"\byou\s+are\s+now\s+(?:an?\s+|the\s+)?(?:[\w-]+\s+){0,3}(?:assistant|ai|model|chat\s?bot|bot|agent)\b",
    r"(?:system|developer)\s+(?:message|prompt|instructions?)\s+(?:for|to)\s+(?:any\s+|the\s+|an\s+|all\s+)?(?:ai|assistant|model|llm)",
    rf"\b{_AI}\b[^.\n]{{0,60}}\b(?:must|should|shall|is\s+instructed\s+to)\b[^.\n]{{0,40}}\b(?:approve|ignore|report|say|respond)",
    r"report\s+(?:the\s+)?risk(?:\s+level)?\s+as\s+(?:low|none|minimal)",
    r"do\s+not\s+(?:mention|report|flag|disclose)\b[^.\n]{0,60}\b(?:difference|discrepanc|mismatch|contradiction)",
]
_INJECTION_REGEX = [re.compile(pattern, re.IGNORECASE) for pattern in _INJECTION_PATTERNS]

# Things that look like API keys (Google "AIza...", OpenAI/xAI "sk-..."/"xai-...").
_KEY_LIKE = re.compile(r"\b(?:AIza[0-9A-Za-z_\-]{20,}|sk-[0-9A-Za-z_\-]{20,}|xai-[0-9A-Za-z_\-]{20,})\b")


def detect_prompt_injection(text: str) -> list[str]:
    """Return the suspicious phrases found in ``text`` (empty list = nothing found)."""
    flattened = re.sub(r"\s+", " ", text)
    found: list[str] = []
    for regex in _INJECTION_REGEX:
        match = regex.search(flattened)
        if match:
            phrase = match.group(0).strip()
            if phrase.lower() not in {existing.lower() for existing in found}:
                found.append(phrase[:120])
    return found


def neutralize_delimiters(text: str) -> str:
    """Stop document text from imitating our block markers.

    If a document contained the closing marker itself, the model could be
    tricked into thinking the untrusted block had ended. Replacing the angle
    brackets makes that impossible.
    """
    return text.replace("<<<", "‹‹‹").replace(">>>", "›››")


def wrap_untrusted(chunk: RetrievedChunk) -> str:
    """Wrap one chunk so the model can tell evidence from instructions."""
    header = f'{UNTRUSTED_OPEN} ref={chunk.ref} file="{chunk.filename}" page={chunk.page}>>>'
    return f"{header}\n{neutralize_delimiters(chunk.text)}\n{UNTRUSTED_CLOSE}"


def redact_secrets(text: str, settings: Settings) -> str:
    """Remove API keys and passwords from text before it is returned or logged."""
    if not text:
        return text
    secrets = []
    for secret in (settings.GEMINI_API_KEY, settings.GROK_API_KEY):
        if secret is not None and len(secret.get_secret_value()) >= 6:
            secrets.append(secret.get_secret_value())
    for secret in secrets:
        text = text.replace(secret, "[REDACTED]")
    text = _KEY_LIKE.sub("[REDACTED]", text)
    # Database password inside a connection URL: postgresql://user:password@host
    text = re.sub(r"(postgres(?:ql)?(?:\+\w+)?://[^:/\s]+:)[^@\s]+@", r"\1[REDACTED]@", text)
    return text


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for item in items:
        marker = re.sub(r"\s+", " ", item).strip().lower()
        if marker and marker not in seen:
            seen.add(marker)
            result.append(item.strip())
    return result
