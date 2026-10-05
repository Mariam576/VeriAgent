"""Do two company names mean the same company?

Documents rarely spell a name the same way twice: "Brightline Engineering
Services Ltd", "Brightline Engineering Services Limited", "BRIGHTLINE
ENGINEERING SERVICES LTD." are one company. On the other hand "Brightline
Engineering Solutions Ltd" is a different company whose name merely looks
alike, and that is exactly what a fake supplier would choose.

So the comparison has more outcomes than "equal" and "not equal":

    same         equal once spelling noise is removed (capitals, "Ltd" for "Limited", "&" for "and")
    short_form   one name is the other with words left out or added: "Redbridge Lifts" and
                 "Redbridge Lifts Services Ltd". Usually the same company, written short on a
                 letterhead - but adding a word is also how a look-alike company is named
                 ("... Services", "... Holdings", "... 2026"), so it is worth a look
    similar      nearly the same letters but not the same words (a typing error, or a look-alike)
    different    anything else
    unknown      one of the names is empty, so nothing can be said
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Literal

NameMatch = Literal["same", "short_form", "similar", "different", "unknown"]
# The outcomes that mean "probably the same company". Used where the question is which documents belong together.
RELATED = ("same", "short_form", "similar")

# Legal forms and filler words say nothing about which company is meant.
LEGAL_FORMS = (
    "ltd", "limited", "inc", "incorporated", "llc", "llp", "lp", "plc", "gmbh", "ag", "corp", "corporation",
    "co", "company", "pvt", "private", "pty", "sa", "bv", "nv", "srl", "sarl", "oy", "ab", "kg", "spa",
    "sdn", "bhd", "pte",
)
_NOISE = set(LEGAL_FORMS) | {"the", "and", "of"}
SIMILAR_RATIO = 0.85


def name_tokens(name: str | None) -> list[str]:
    """'Brightline Engineering Services Ltd.' -> ['brightline', 'engineering', 'services']"""
    text = unicodedata.normalize("NFKD", name or "")
    text = "".join(char for char in text if not unicodedata.combining(char)).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"(?<=\w)['’](?=s\b)", "", text)  # "Lloyd's" -> "lloyds"
    words = re.findall(r"[a-z0-9]+", text)
    return [word for word in words if word not in _NOISE]


def compare_names(first: str | None, second: str | None) -> NameMatch:
    a, b = name_tokens(first), name_tokens(second)
    if not a or not b:
        return "unknown"
    if a == b or set(a) == set(b):
        return "same"
    # A short form: "Brightline Engineering" for "Brightline Engineering Services". The first word must
    # agree, because that is the distinctive one ("Services" alone would match half of all suppliers).
    shorter, longer = sorted((a, b), key=len)
    if shorter[0] == longer[0] and set(shorter) < set(longer):
        return "short_form"
    if SequenceMatcher(None, " ".join(a), " ".join(b)).ratio() >= SIMILAR_RATIO:
        return "similar"
    return "different"


def same_identifier(first: str | None, second: str | None) -> bool:
    """Do two reference numbers mean the same document? 'MC-2026-011' = 'mc 2026/011' = 'MC2026011'.

    A reference may also leave out a word prefix ("2026-004" for "Contract-2026-004"), which is
    accepted when the shared part is long enough to be meaningful.
    """
    a = re.sub(r"[^A-Z0-9]", "", (first or "").upper())
    b = re.sub(r"[^A-Z0-9]", "", (second or "").upper())
    if not a or not b:
        return False
    if a == b:
        return True
    shorter, longer = sorted((a, b), key=len)
    return len(shorter) >= 4 and sum(char.isdigit() for char in shorter) >= 3 and longer.endswith(shorter)
