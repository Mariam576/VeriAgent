"""Show text from documents and models as plain text inside Markdown.

Streamlit renders most strings as Markdown. Text that comes from an uploaded
document (a quote, a file name, a sentence the model copied) is untrusted: an
invoice could contain ``![logo](http://attacker/x.png)`` and the browser would
fetch that address the moment the report is displayed, or ``[Pay here](...)``
and the report would carry a phishing link. Amounts would also be mangled,
because ``$40,000 ... $58,000`` is read as a formula.

``plain()`` makes every such string display exactly as written: no images, no
links, no formatting. The API has the same function in ``app/formatting.py``
(``escape_markdown``) for reports rendered as Markdown; the two must behave
identically, and a test checks that.
"""

from __future__ import annotations

import re

# Every character Markdown (or Streamlit's extensions: $ for formulas) gives a meaning to.
_SPECIAL = re.compile(r"([\\`*_{}\[\]<>()#+!|~$])")
_SCHEME = re.compile(r"\b(https?|ftp|mailto|file|javascript|data)(:)(/{0,2})", re.IGNORECASE)
_WWW = re.compile(r"\bwww\.", re.IGNORECASE)
_ZERO_WIDTH = "​"


def plain(text: object) -> str:
    """Escape ``text`` so that Markdown shows it literally, on one line."""
    value = " ".join(str(text if text is not None else "").split())  # also removes line breaks and control spacing
    # A bare address would be turned into a link automatically; an invisible break prevents that.
    value = _SCHEME.sub(lambda match: f"{match.group(1)}{match.group(2)}{_ZERO_WIDTH}{match.group(3)}", value)
    value = _WWW.sub(f"www{_ZERO_WIDTH}.", value)
    return _SPECIAL.sub(r"\\\1", value)
