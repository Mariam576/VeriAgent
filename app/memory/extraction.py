"""Decide what, if anything, in a user's message should be remembered (spec section 27).

Concept - memory extraction:
    Most messages are questions and contain nothing worth keeping. Storing
    every message would fill the memory with noise and make recall useless.
    So a message is only remembered when it states something durable: a
    standing preference ("from now on keep answers brief") or a lasting fact
    ("remember that Vendor ABC was flagged for duplicate invoices in 2025").

Two steps keep this cheap and safe:

1. A free keyword check decides whether the message could contain a memory
   at all. Only then is the language model asked (one call, structured
   output). Ordinary questions never cost a call.
2. Whatever the model proposes is validated in code before it is stored:
   preference keys and values must be on the allowlist, and a fact must be a
   short statement, not a question.
"""

from __future__ import annotations

import re

from app.agents import prompts
from app.config import Settings
from app.memory.profile import PREFERENCES, normalize_preference
from app.schemas.memory import MemoryCandidate
from app.security.guardrails import redact_secrets

MAX_NOTE_CHARS = 400
MIN_NOTE_CHARS = 10

# Phrases that signal "this is about the future", not about the current question.
_GATE = re.compile(
    r"\b(?:from now on|in (?:the )?future|going forward|by default|always|never|"
    r"i(?:'d| would)? prefer|my preference|remember(?: that)?|keep in mind|"
    r"note that|for future reference|don'?t forget|for the record)\b",
    re.IGNORECASE,
)
# "remember that ...", "note that ...", "for the record, ..." introduce a fact to keep.
# "remember to check the invoice" is an instruction for now, so "that" is required.
_REMEMBER_FACT = re.compile(
    r"\b(?:(?:remember|keep in mind|note|don'?t forget)\s+that\b|for future reference\b[\s,:]*|for the record\b[\s,:]*)\s*(.+)",
    re.IGNORECASE | re.DOTALL,
)
# A preference must be about the answers, not about the documents ("short payment terms").
_ABOUT_ANSWERS = re.compile(
    r"\b(?:answers?|repl(?:y|ies)|responses?|reports?|summar(?:y|ies)|output|results?|it|things)\b", re.IGNORECASE
)
# Words that start a request. In front of a memory phrase they mean "do this AND remember that".
_REQUEST = re.compile(
    r"\b(?:compare|review|check|verify|investigate|calculate|analy[sz]e|examine|audit|find|search|look|show|"
    r"tell|list|summari[sz]e|explain|approve|reject|pay|what|which|who|whom|when|where|why|how)\b",
    re.IGNORECASE,
)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

_VALUE_PATTERNS: dict[str, dict[str, re.Pattern[str]]] = {
    "detail_level": {
        "brief": re.compile(r"\b(?:brief|short|shorter|concise|to the point)\b", re.IGNORECASE),
        "detailed": re.compile(r"\b(?:detailed|in detail|full detail|thorough|as much detail)\b", re.IGNORECASE),
        "standard": re.compile(r"\b(?:standard|normal|default) (?:detail|length|answers?|reports?)\b", re.IGNORECASE),
    },
    "response_format": {
        "markdown": re.compile(r"\bmarkdown\b", re.IGNORECASE),
        "text": re.compile(r"\bplain[\s-]?text\b", re.IGNORECASE),
    },
}

MEMORY_TASK = """YOUR TASK NOW: decide whether the user's message contains something worth remembering in later sessions.
Save only what is durable, useful later and not trivial. Most messages contain nothing to save.
- A standing preference about how answers should look is memory_type "profile". Use only the allowed
  keys and values listed in the request. If the wish does not fit an allowed key, save nothing.
- A lasting fact about the user's vendors, contracts or earlier findings is memory_type "semantic".
  Write it as one self-contained sentence. Do not add anything the user did not say.
- A question, a request to investigate something, or a greeting is memory_type "none".
Set only_memory to true when the message asks for nothing else."""


def might_contain_memory(message: str) -> bool:
    """Free pre-check. False means: do not spend a model call on this message."""
    return bool(_GATE.search(message))


def build_memory_prompt(message: str) -> str:
    allowed = {key: list(preference.allowed_values) for key, preference in PREFERENCES.items()}
    return "\n\n".join(
        [
            prompts.json_section("ALLOWED PREFERENCES", allowed),
            prompts.text_section("USER REQUEST", message),
        ]
    )


def _split_sentences(message: str) -> list[str]:
    return [sentence.strip() for sentence in _SENTENCE_END.split(message.strip()) if sentence.strip()]


def is_only_memory(message: str) -> bool:
    """True when, apart from the thing to remember, the message asks for nothing.

    This decides whether the investigation is skipped, so it is decided by
    code and it errs on the side of investigating: any question mark, any
    sentence without a memory phrase, or a request in front of the memory
    phrase ("Compare the invoice with the contract and remember that ...")
    means there is still something to do.
    """
    for sentence in _split_sentences(message):
        marker = _GATE.search(sentence)
        if marker is None:
            if len(sentence.split()) >= 2:
                return False  # a sentence that is not about remembering anything
        elif sentence.rstrip().endswith("?") or _REQUEST.search(sentence[: marker.start()]):
            return False
    return True


def fallback_memory(message: str) -> MemoryCandidate:
    """Rule-based extraction: used by the offline mock LLM and when the model is unavailable."""
    nothing = MemoryCandidate(should_save=False, memory_type="none", key=None, value=None, text=None, only_memory=False)
    # A question is never a memory: "Does the report always show short terms?" asks, it does not instruct.
    marked = [
        sentence for sentence in _split_sentences(message)
        if _GATE.search(sentence) and not sentence.rstrip().endswith("?")
    ]
    if not marked:
        return nothing
    only_memory = is_only_memory(message)

    for sentence in marked:
        if not _ABOUT_ANSWERS.search(sentence):
            continue
        for key, values in _VALUE_PATTERNS.items():
            for value, pattern in values.items():
                if pattern.search(sentence):
                    return MemoryCandidate(
                        should_save=True, memory_type="profile", key=key, value=value, text=None, only_memory=only_memory
                    )
    for sentence in marked:
        match = _REMEMBER_FACT.search(sentence)
        if match:
            fact = match.group(1).strip().rstrip(".").strip()
            if len(fact.split()) >= 4:
                return MemoryCandidate(
                    should_save=True, memory_type="semantic", key=None, value=None,
                    text=fact[0].upper() + fact[1:] + ".", only_memory=only_memory,
                )
    return nothing


def clean_note(text: str | None, settings: Settings) -> str | None:
    """A fact that is fit to store: one line, bounded, no secrets, not a question."""
    cleaned = re.sub(r"\s+", " ", re.sub(r"[\x00-\x1f\x7f]", " ", text or "")).strip()
    cleaned = redact_secrets(cleaned, settings)[:MAX_NOTE_CHARS].strip()
    if len(cleaned) < MIN_NOTE_CHARS or cleaned.endswith("?"):
        return None
    return cleaned


def accepted_memory(candidate: MemoryCandidate, settings: Settings) -> tuple[str, str, str] | None:
    """Validate the model's proposal.

    Returns ("profile", key, value), ("semantic", text, "") or None when
    nothing may be stored. The model's own ``should_save`` is necessary but
    not sufficient: the allowlist and the text checks decide.
    """
    if not candidate.should_save or candidate.memory_type == "none":
        return None
    if candidate.memory_type == "profile":
        pair = normalize_preference(candidate.key, candidate.value)
        return ("profile", pair[0], pair[1]) if pair else None
    note = clean_note(candidate.text, settings)
    return ("semantic", note, "") if note else None


def mock_memory(system: str, user: str) -> MemoryCandidate:
    """Handler for the offline mock LLM."""
    return fallback_memory(prompts.read_section(user, "USER REQUEST") or "")
