"""Build a document profile with rules only.

Used in two situations, like the other rule-based stand-ins in this project:

1. the LLM is unavailable after its retries (fallback), and
2. ``LLM_PROVIDER=mock`` (tests and offline demos).

The rules understand the common wordings: "Invoice No. INV-102",
"Contract reference: ...", "From: ... / To: ...", "between A (the "Client")
and B (the "Supplier")", "Invoice date: 3 March 2026", "starts on ... and
continues for twelve months", "description: £amount" lines and
"40 items at $205 each". A real model reads far more layouts; the rules are
deliberately strict, because a field that is not found only means "no check",
while a field that is read wrongly would raise a false alarm.

Every value is taken from the text before any injected instruction
(``trusted_part``): a sentence that follows "ignore previous instructions"
is never a source of facts.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from app.comparison.adjustments import find_adjustments, is_credit_note
from app.comparison.dates import find_dates, months_in_text
from app.comparison.names import LEGAL_FORMS
from app.comparison.profile import (
    MAX_DESCRIPTION_CHARS,
    MAX_IDENTIFIER_CHARS,
    MAX_LINE_ITEMS,
    MAX_NAME_CHARS,
    MAX_PARTIES,
    MAX_QUOTE_CHARS,
    DocumentProfile,
    LineItem,
    PartyRole,
    ProfileField,
)
from app.numbers import NUMBER_CORE, parse_number
from app.patterns import MONEY, find_amounts, find_payment_terms, iter_segments, money_amount, money_currency, money_readings, sentence_bounds
from app.rag.loaders import PageText
from app.security.guardrails import trusted_part

# A reference number: letters, digits and separators, with at least one digit ("INV-102", "MC-2026-011").
_ID = r"(?P<id>[A-Za-z0-9][A-Za-z0-9/_.-]*\d[A-Za-z0-9/_-]*)"
_LABEL = r"(?:no\.?|number|num\.?|#|ref(?:erence)?\.?)"
_INVOICE_NUMBER = re.compile(rf"\binvoice\s*{_LABEL}\s*[:#]?\s*{_ID}", re.I)
_CONTRACT_NUMBER = re.compile(rf"\b(?:contract|agreement)\s*{_LABEL}\s*[:#]?\s*{_ID}", re.I)
_PO_NUMBER = re.compile(rf"\b(?:purchase\s+order|p\.?o\.?(?![a-z]))\s*{_LABEL}\s*[:#]?\s*{_ID}", re.I)
# On an invoice the purchase order is often given without "No.": "Purchase order: PO-4421".
# The short form needs a label or a separator ("PO No. 4421", "PO: 4421", "P.O. #4421"): "PO16 7GZ" is a postcode.
_PO_REFERENCE = re.compile(
    rf"\bpurchase\s+order(?:\s*{_LABEL})?\s*[:#]?\s*{_ID}|\bp\.?o\.?(?![a-z])\s*(?:{_LABEL}\s*[:#]?|[:#])\s*{_ID.replace('P<id>', 'P<id2>')}",
    re.I,
)

_CREDIT_NOTE_NUMBER = re.compile(rf"\bcredit\s+(?:note|memo)\s*{_LABEL}?\s*[:#]?\s*{_ID}", re.I)

# A date label counts in this order: first the label that names this kind of document's own date,
# and only if there is none a bare "Date:" at the start of a line (or "dated ..."). A bare "date"
# anywhere would also be the end of "Required delivery date:" and "Order date:" on an invoice.
_BARE_DATE = re.compile(r"(?:(?:^|(?<=[\n.;]))[ \t]*date\s*:\s*|\bdated\s+(?:the\s+)?)", re.I | re.M)
_DATE_LABELS = {
    "invoice": [
        re.compile(r"\b(?:invoice\s+date|date\s+of\s+(?:invoice|issue)|issue\s+date|issued(?:\s+on)?|tax\s+point(?:\s+date)?)\s*:?\s*", re.I),
        _BARE_DATE,
    ],
    "purchase_order": [
        re.compile(r"\b(?:order\s+date|date\s+of\s+order|p\.?o\.?\s+date|purchase\s+order\s+date|ordered\s+on)\s*:?\s*", re.I),
        _BARE_DATE,
    ],
    "contract": [
        re.compile(
            r"\b(?:made|dated|entered\s+into|executed)\s+(?:on\s+|as\s+of\s+)?(?:this\s+)?"
            r"|\b(?:agreement\s+date|contract\s+date)\s*:\s*",
            re.I,
        ),
        _BARE_DATE,
    ],
}
# The term of a contract. The sentence has to be about the contract itself ("This agreement
# starts on ...", "The term ends on ..."): "prices are fixed until 31 March" and "invoicing
# begins on 1 June" also contain a date after "until" and "begins", and say nothing about the term.
_TERM_SUBJECT = r"(?:agreement|contract|term|engagement|appointment|services?)"
_TERM_START = [
    re.compile(r"\b(?:commencement\s+date|start\s+date|effective\s+date)\s*:?\s*(?:is\s+|shall\s+be\s+)?", re.I),
    re.compile(
        rf"\b{_TERM_SUBJECT}\b[^.\n]{{0,80}}?\b(?:starts?|commences?|begins?|commencing|starting|effective|takes\s+effect)\s+(?:on\s+|from\s+)?",
        re.I,
    ),
]
_TERM_END = [
    re.compile(r"\b(?:expiry\s+date|expiration\s+date|end\s+date|termination\s+date)\s*:?\s*(?:is\s+|shall\s+be\s+)?", re.I),
    re.compile(
        rf"\b{_TERM_SUBJECT}\b[^.\n]{{0,80}}?\b(?:ends?|expires?|expiring|terminates?|ending|(?:continues?\s+|runs?\s+)?until)\s+(?:on\s+)?",
        re.I,
    ),
]
_NOT_THE_TERM = re.compile(r"\b(?:prices?|pricing|rates?|fees?|charges?|quot\w+|offer|valid|fixed|invoic\w+|payments?|deliver\w+|notice|warrant\w+)\b", re.I)

_SUPPLIER_LABELS = ("from", "supplier", "vendor", "seller", "contractor", "service provider", "issued by", "payee")
_CUSTOMER_LABELS = ("to", "bill to", "billed to", "invoice to", "buyer", "customer", "client", "purchaser", "sold to", "ordered by", "ship to", "deliver to")
# A purchase order is written by the buyer: there "From:" is the customer and "To:" the supplier.
_SWAPPED_ON_AN_ORDER = {"from": "customer", "to": "supplier"}
# "To: Accounts Payable, Northwind Facilities Ltd": the department is not the company.
_DEPARTMENT = re.compile(
    r"^(?:the\s+)?(?:accounts?(?:\s+(?:payable|receivable))?|finance|purchasing|procurement|billing|payables)"
    r"(?:\s+(?:department|dept\.?|team|office|manager))?$|^(?:attn\.?|attention|f\.?a\.?o\.?|c/o)\b.*$",
    re.I,
)
_ROLE_WORDS: dict[str, PartyRole] = {
    **{word: "supplier" for word in ("supplier", "vendor", "seller", "contractor", "provider", "service provider", "consultant", "licensor")},
    # "the Company" is left out on purpose: half of all contracts call the supplier that, the other half the customer.
    **{word: "customer" for word in ("client", "customer", "buyer", "purchaser", "employer", "licensee")},
}
# A label counts only at the start of a line or sentence: "To: Northwind", not "payable to: ...".
_PARTY_LABEL = re.compile(
    r"(?:^|(?<=[\n.;])[ \t]*)(?P<label>" + "|".join(sorted(_SUPPLIER_LABELS + _CUSTOMER_LABELS, key=len, reverse=True)) + r")[ \t]*:[ \t]*",
    re.I | re.M,
)
_BETWEEN = re.compile(r"\bbetween\b\s*:?\s*", re.I)
_ROLE_MARKER = re.compile(r"\(\s*(?:the\s+|each\s+a\s+)?[\"“”'‘’]?(?P<role>[A-Za-z][A-Za-z ]{2,24}?)[\"“”'‘’]?\s*\)")
_ENUMERATION = re.compile(r"\s*\(?\b(?:\d|[a-z]|i{1,3})\)\s*", re.I)
_LEGAL_FORM = re.compile(r"\b(?:" + "|".join(LEGAL_FORMS) + r")\b\.?(?![\w-])", re.I)
_HARD_STOPS = (",", ". ", ";", " (the", " (each", "\n", ":", " - ")
_SOFT_STOPS = (" and ", " for ", " of ", " whose ", " a company")

_TAX_LABEL = re.compile(r"\b(?:vat|gst|sales\s+tax|tax)\b", re.I)
_NOT_A_LINE = re.compile(
    r"\b(?:total|sub[\s-]?total|amount\s+due|balance|vat|gst|tax|deposit|discount|limit|threshold|value|"
    r"purchase\s+order|contract|invoice|payment|terms?)\b",
    re.I,
)
_NUMBER = NUMBER_CORE + r"(?![\w]|[.,]\d)"
_QUANTITY_LINE = re.compile(
    rf"^\W*(?P<qty>{_NUMBER})\s*(?:x\s+)?(?P<desc>[^.:\n|]{{3,120}}?)\s+(?:at|@)\s+(?P<price>[^\s|]+(?:\s[^\s|]+)?)\s+(?:each|per\s+\w+)",
    re.I,
)
_CELL_NUMBER = re.compile(rf"^\s*(?:£|\$|€|₹)?\s*(?P<num>{_NUMBER})\s*$")


def _clip(text: str, limit: int = MAX_QUOTE_CHARS) -> str:
    return text.strip()[:limit].strip()


def _sentence(text: str, start: int, end: int) -> str:
    """The line or sentence around a match, cut to quote length but always containing the match."""
    left, right = sentence_bounds(text, start, end)
    if right - left > MAX_QUOTE_CHARS:
        left = max(left, start - 80)
        right = min(right, max(end, left + MAX_QUOTE_CHARS))
    return text[left:right].strip(" \n")


def _name_at(text: str, start: int, *, soft_stops: bool) -> tuple[str, int]:
    """Read a company name that begins at ``text[start]``. Returns (name, index just after it).

    A name ends at its legal form ("... Ltd", "... Inc.") when one comes before any
    punctuation, otherwise at the first punctuation mark or line break.
    """
    window = text[start : start + MAX_NAME_CHARS + 40]
    stops = _HARD_STOPS + (_SOFT_STOPS if soft_stops else ())
    hard_cut = min((pos for stop in _HARD_STOPS if (pos := window.find(stop)) != -1), default=len(window))
    any_cut = min((pos for stop in stops if (pos := window.find(stop)) != -1), default=len(window))
    legal = None
    for legal in _LEGAL_FORM.finditer(window):
        if legal.start() > hard_cut:
            legal = None
            break
        following = window[legal.end() : legal.end() + 8].lstrip(" ).")
        if not _LEGAL_FORM.match(following):
            break  # "(Pvt) Ltd", "Co. Ltd": keep reading until the last legal form in a row
    cut = legal.end() if legal is not None and legal.start() <= hard_cut else any_cut
    name = window[:cut].strip(" \t.,;:-\"'“”")
    return name[:MAX_NAME_CHARS], start + cut


def _pages(pages: list[PageText]) -> Iterator[tuple[int, str]]:
    for page in pages:
        text = trusted_part(page.text)
        if text.strip():
            yield page.page_number, text


def _identifier(pattern: re.Pattern[str], pages: list[PageText]) -> ProfileField | None:
    for page, text in _pages(pages):
        match = pattern.search(text)
        if match:
            value = (match.group("id") or match.groupdict().get("id2") or "").rstrip(".-/_")[:MAX_IDENTIFIER_CHARS]
            return ProfileField(text=value, page=page, quote=_clip(match.group(0).rstrip(".")))
    return None


def _date_after(text: str, position: int) -> tuple[list, str, int] | None:
    """A date that starts right at ``position``: (readings, as written, end index)."""
    for found in find_dates(text[position : position + 40]):
        if found.start <= 1:
            return list(found.readings), found.text, position + found.end
        break
    return None


def _labelled_date(
    patterns: list[re.Pattern[str]], pages: list[PageText], veto: re.Pattern[str] | None = None
) -> tuple[ProfileField, int, int] | None:
    """The first date that directly follows a label: (field, page, end index on that page).

    ``patterns`` are tried in order, each over the whole document, so a weaker
    label is only used when the stronger one occurs nowhere. ``veto``: words
    that must not stand in the matched label.
    """
    for pattern in patterns:
        for page, text in _pages(pages):
            for match in pattern.finditer(text):
                if veto is not None and veto.search(match.group(0)):
                    continue
                hit = _date_after(text, match.end())
                if hit:
                    readings, written, end = hit
                    field = ProfileField(text=written, dates=readings, page=page, quote=_sentence(text, match.start(), end))
                    return field, page, end
    return None


def _term(pages: list[PageText]) -> tuple[ProfileField | None, ProfileField | None, ProfileField | None]:
    start = end = months = None
    found = _labelled_date(_TERM_START, pages, _NOT_THE_TERM)
    if found:
        start, page, position = found
        text = next(text for number, text in _pages(pages) if number == page)
        _, sentence_end = sentence_bounds(text, position, position)
        for count, begin, finish in months_in_text(text[position:sentence_end]):
            months = ProfileField(
                number=float(count), unit="months", page=page,
                quote=_sentence(text, position + begin, position + finish),
            )
            break
    found = _labelled_date(_TERM_END, pages, _NOT_THE_TERM)
    if found:
        end = found[0]
    return start, end, months


def _labelled_parties(pages: list[PageText], document_type: str = "invoice") -> list[ProfileField]:
    parties = []
    for page, text in _pages(pages):
        for match in _PARTY_LABEL.finditer(text):
            label = match.group("label").lower()
            name, name_end = _name_at(text, match.end(), soft_stops=False)
            # "To: Accounts Payable, Northwind Facilities Ltd": step over the department to the company.
            for _ in range(2):
                if not _DEPARTMENT.match(name) or not text.startswith(",", name_end):
                    break
                name, name_end = _name_at(text, name_end + 1 + (len(text[name_end + 1:]) - len(text[name_end + 1:].lstrip(" "))), soft_stops=False)
            if len(name) < 2 or not re.search(r"[A-Za-z]", name) or _DEPARTMENT.match(name):
                continue
            role: PartyRole = "supplier" if label in _SUPPLIER_LABELS else "customer"
            if document_type == "purchase_order":
                role = _SWAPPED_ON_AN_ORDER.get(label, role)  # type: ignore[assignment]
            parties.append(ProfileField(text=name, role=role, page=page, quote=_clip(text[match.start("label") : name_end])))
    return parties


def _contract_parties(pages: list[PageText]) -> list[ProfileField]:
    """'... between A (the "Client") and B (the "Supplier")', with or without the roles."""
    for page, text in _pages(pages):
        for between in _BETWEEN.finditer(text):
            parties: list[ProfileField] = []
            position = between.end()
            for _ in range(2):
                enumeration = _ENUMERATION.match(text, position)
                if enumeration:
                    position = enumeration.end()
                name, name_end = _name_at(text, position, soft_stops=True)
                if len(name) < 2 or not name[0].isupper():
                    break
                # A role, if one is given, stands in brackets before the next party starts.
                tail = text[name_end : name_end + 260]
                separator = re.search(r"\band\b(?=\s*(?:\(\d\)\s*)?[A-Z(])", tail)
                marker = _ROLE_MARKER.search(tail)
                role = None
                quote_end = name_end
                if marker and (separator is None or marker.start() < separator.start() or not parties):
                    role = _ROLE_WORDS.get(marker.group("role").strip().lower())
                    if role is not None:
                        quote_end = name_end + marker.end()
                        separator = re.search(r"\band\b", tail[marker.end():])
                        separator_end = marker.end() + separator.end() if separator else None
                    else:
                        separator_end = separator.end() if separator else None
                else:
                    separator_end = separator.end() if separator else None
                quote = text[position:quote_end]
                if len(quote) > MAX_QUOTE_CHARS:  # a long address between the name and the role
                    quote = name
                parties.append(ProfileField(text=name, role=role, page=page, quote=_clip(quote)))
                if separator_end is None:
                    break
                position = name_end + separator_end
                while position < len(text) and text[position] in " \t\n":
                    position += 1
            if len(parties) == 2:
                return parties
    return []


def _amount_fields(pages: list[PageText]) -> tuple[ProfileField | None, ProfileField | None, ProfileField | None]:
    """(total, subtotal, tax). The total is the amount behind the strongest "total" label."""
    best: tuple[int, float, ProfileField] | None = None
    subtotal = tax = None
    for page, text in _pages(pages):
        for hit in find_amounts(text):
            field = ProfileField(number=hit.value, unit=hit.unit or None, page=page, quote=_sentence(text, hit.start, hit.end))
            if hit.subtotal:
                subtotal = subtotal or field
            elif hit.priority:
                # A higher-ranked label wins; between equals the larger amount (a total is never smaller than its parts).
                if best is None or (hit.priority, abs(hit.value)) > (best[0], best[1]):
                    best = (hit.priority, abs(hit.value), field)
            elif tax is None and _TAX_LABEL.search(hit.before) and re.search(r"[:(%]|\bat\b", hit.before):
                tax = field
    return (best[2] if best else None), subtotal, tax


def _payment_terms(pages: list[PageText]) -> ProfileField | None:
    for page, text in _pages(pages):
        flattened = re.sub(r"\s+", " ", text)
        terms = find_payment_terms(flattened)
        if terms:
            days, wording = terms
            return ProfileField(number=days, unit="days", page=page, quote=_clip(wording))
    return None


def _cell_number(cell: str) -> float | None:
    match = _CELL_NUMBER.match(cell)
    readings = parse_number(match.group("num")) if match else []
    return readings[0] if readings else None


def _unit_price(price: re.Match[str], quantity: float, stated: float | None) -> float | None:
    """The price per unit of a "40 items at $205 each" line.

    "$3.459 per gallon" can be read as three dollars and a bit, or as three
    thousand. A stated line total settles it: the reading that multiplies out
    to that total is the price. Without one the price stays unread - an
    unread value means "not checked", a misread one would be a false alarm.
    """
    readings = money_readings(price)
    if len(readings) == 1:
        return readings[0]
    if stated is not None:
        fitting = [reading for reading in readings if abs(quantity * reading - stated) <= 0.011]
        if len(fitting) == 1:
            return fitting[0]
    return None


def _line_items(pages: list[PageText]) -> list[LineItem]:
    items: list[LineItem] = []

    def add(description: str, quantity: float | None, price: float | None, amount: float | None, unit: str | None, page: int, quote: str) -> None:
        description = " ".join(description.split()).strip(" -:")[:MAX_DESCRIPTION_CHARS]
        if len(description) < 3 or len(items) >= MAX_LINE_ITEMS or (quantity is None and price is None and amount is None):
            return
        # Two identical lines are two lines: an invoice may bill the same call-out twice.
        items.append(LineItem(description=description, quantity=quantity, unit_price=price, amount=amount, unit=unit, page=page, quote=_clip(quote)))

    for page, text in _pages(pages):
        for _, _, stripped in iter_segments(text):  # sentences and lines, one at a time
            if not stripped:
                continue

            # 1. A table row from a Word document: "Description | 4 | £250 | £1,000".
            cells = [cell.strip() for cell in stripped.split("|")]
            if len(cells) >= 4 and not _NOT_A_LINE.search(cells[0]):
                numbers = [_cell_number(cell) for cell in cells[-3:]]
                if all(number is not None for number in numbers):
                    money = MONEY.search(stripped)
                    add(cells[0], numbers[0], numbers[1], numbers[2], money_currency(money) if money else None, page, stripped)
                    continue

            # 2. "40 sterilisation pouches cartons at $205 each"
            match = _QUANTITY_LINE.match(stripped)
            if match:
                price = MONEY.search(match.group("price"))
                quantity = parse_number(match.group("qty"))
                if price and quantity and money_amount(price) is not None:
                    rest = stripped[match.end():]
                    stated = MONEY.search(rest) if re.match(r"\s*[:=,]", rest) else None  # "... each: $8,200"
                    stated_amount = money_amount(stated) if stated else None
                    add(
                        match.group("desc"), quantity[0], _unit_price(price, quantity[0], stated_amount),
                        stated_amount, money_currency(price) or None, page,
                        stripped[: stated.end() + match.end()] if stated else match.group(0),
                    )
                    continue

            # 3. "Description of the work: £12,000"
            amounts = list(MONEY.finditer(stripped))
            if len(amounts) == 1 and stripped[amounts[0].end():].strip(" .)") == "":
                before = stripped[: amounts[0].start()].rstrip("(-−– ")
                if re.search(r":\s*$", before) and not _NOT_A_LINE.search(before) and money_amount(amounts[0]) is not None:
                    add(before, None, None, money_amount(amounts[0]), money_currency(amounts[0]) or None, page, stripped)
    return items


def enrich(profile: DocumentProfile, pages: list[PageText]) -> DocumentProfile:
    """Add what code finds in the text of every document, however the rest of the profile was built."""
    profile.adjustments, profile.unquantified_adjustment = find_adjustments(pages)
    profile.credit_note = profile.document_type == "invoice" and is_credit_note(pages)
    return profile


def build_profile_with_rules(
    *, document_id: str, filename: str, document_type: str, pages: list[PageText]
) -> DocumentProfile:
    profile = DocumentProfile(document_id=document_id, filename=filename, document_type=document_type, built_by="rules")

    if document_type == "invoice":
        if is_credit_note(pages):
            # "Credit against Invoice No. INV-300" names the invoice being corrected, not this document.
            profile.number = _identifier(_CREDIT_NOTE_NUMBER, pages)
        else:
            profile.number = _identifier(_INVOICE_NUMBER, pages)
        profile.contract_reference = _identifier(_CONTRACT_NUMBER, pages)
        profile.purchase_order_reference = _identifier(_PO_REFERENCE, pages)
    elif document_type == "purchase_order":
        profile.number = _identifier(_PO_NUMBER, pages)
        profile.contract_reference = _identifier(_CONTRACT_NUMBER, pages)
    else:
        profile.number = _identifier(_CONTRACT_NUMBER, pages)

    parties = _labelled_parties(pages, document_type)
    if document_type == "contract" or not parties:
        parties = _contract_parties(pages) or parties
    unique: list[ProfileField] = []
    for party in parties:  # one supplier, one customer; parties without a role are all kept
        if party.role is None or all(other.role != party.role for other in unique):
            unique.append(party)
    profile.parties = unique[:MAX_PARTIES]

    dated = _labelled_date(_DATE_LABELS.get(document_type, _DATE_LABELS["invoice"]), pages)
    profile.date = dated[0] if dated else None
    if document_type == "contract":
        profile.term_start, profile.term_end, profile.term_months = _term(pages)

    profile.total, profile.subtotal, profile.tax = _amount_fields(pages)
    profile.payment_terms_days = _payment_terms(pages)
    if document_type != "contract":
        profile.line_items = _line_items(pages)
    return enrich(profile, pages)
