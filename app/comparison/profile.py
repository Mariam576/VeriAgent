"""The profile of a document: its key facts in a fixed shape.

Concept - why a profile?
    Search answers "which passages are about this question?". Comparing two
    documents needs something else: the same facts from each of them, side by
    side. Who issued it, to whom, under which contract, on which date, for
    which items at which price. A profile is that list of facts for one
    document, and two profiles can be compared field by field with plain code.

Every value keeps the page it was found on and the exact words that state it
(``quote``). A finding that comes out of a comparison can therefore cite its
sources just like any other finding in a report.

Profiles are built for the document types that take part in a comparison:
contracts, purchase orders and invoices.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

# Raise this number when the way profiles are built changes. Stored profiles
# with a lower version are rebuilt the next time they are needed.
PROFILE_VERSION = 1

PROFILED_TYPES = ("contract", "purchase_order", "invoice")
REFERENCE_TYPES = ("contract", "purchase_order")  # documents an invoice is checked against

PartyRole = Literal["supplier", "customer"]

MAX_NAME_CHARS = 120
MAX_IDENTIFIER_CHARS = 60
MAX_DESCRIPTION_CHARS = 160
MAX_QUOTE_CHARS = 300
MAX_LINE_ITEMS = 60
MAX_PARTIES = 6


class ProfileField(BaseModel):
    """One fact with its source. Which of text / number / dates is filled depends on the field."""

    text: str | None = None  # a name, a reference number, or a date as the document writes it
    number: float | None = None  # an amount or a count
    unit: str | None = None  # ISO currency code, "days" or "months"
    dates: list[date] = Field(default_factory=list)  # one reading, or two for an ambiguous numeric date
    role: PartyRole | None = None  # parties only
    page: int
    quote: str


class LineItem(BaseModel):
    description: str
    quantity: float | None = None
    unit_price: float | None = None
    amount: float | None = None  # the line total as the document states it
    unit: str | None = None  # currency
    page: int
    quote: str

    @property
    def effective_amount(self) -> float | None:
        """The stated line total, or quantity x unit price when no line total is written."""
        if self.amount is not None:
            return self.amount
        if self.quantity is not None and self.unit_price is not None:
            return round(self.quantity * self.unit_price, 2)
        return None


class DocumentProfile(BaseModel):
    document_id: str
    filename: str
    document_type: str
    built_by: Literal["llm", "rules"] = "rules"
    version: int = PROFILE_VERSION

    number: ProfileField | None = None  # the document's own number (invoice no., contract no., PO no.)
    contract_reference: ProfileField | None = None  # the contract this document says it belongs to
    purchase_order_reference: ProfileField | None = None
    parties: list[ProfileField] = Field(default_factory=list)
    date: ProfileField | None = None  # invoice date, order date, or the date a contract was made
    term_start: ProfileField | None = None
    term_end: ProfileField | None = None
    term_months: ProfileField | None = None
    total: ProfileField | None = None
    subtotal: ProfileField | None = None
    tax: ProfileField | None = None
    payment_terms_days: ProfileField | None = None
    line_items: list[LineItem] = Field(default_factory=list)

    def party(self, role: PartyRole) -> ProfileField | None:
        return next((party for party in self.parties if party.role == role), None)

    @property
    def supplier(self) -> ProfileField | None:
        return self.party("supplier")

    @property
    def customer(self) -> ProfileField | None:
        return self.party("customer")

    @property
    def is_empty(self) -> bool:
        """True when nothing at all could be read (a scan without text, an unusual layout)."""
        single = (
            self.number, self.contract_reference, self.purchase_order_reference, self.date, self.term_start,
            self.term_end, self.term_months, self.total, self.subtotal, self.tax, self.payment_terms_days,
        )
        return not any(single) and not self.parties and not self.line_items


# How each field is called when it is shown to a person.
FIELD_LABELS: dict[str, str] = {
    "number": "Document number",
    "contract_reference": "Contract reference",
    "purchase_order_reference": "Purchase order reference",
    "supplier": "Supplier",
    "customer": "Customer",
    "date": "Date",
    "term_start": "Term starts",
    "term_end": "Term ends",
    "term_months": "Term length",
    "total": "Total amount",
    "subtotal": "Subtotal",
    "tax": "Tax",
    "payment_terms_days": "Payment terms",
}
