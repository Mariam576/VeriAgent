"""Document comparison: the cases an independent review found.

Version 3's comparison was checked by a reviewer who fed it ordinary business
documents and looked for accusations, and fraudulent ones and looked for
silence. It found both. Every case is kept here.

The rule that came out of it: do not accuse what you do not model. A partial
delivery, a monthly invoice, a discount, a second tax, an invoice from another
supplier in the same folder - all of them look like a mismatch to a check that
only knows "the two numbers differ".
"""

from __future__ import annotations

import pytest

from app.comparison.engine import check_duplicates, compare_profiles, unrelated_pairs
from app.comparison.llm import ProfileExtraction, ProfileFact, ProfileLine, ground_extraction
from app.comparison.names import compare_names
from app.comparison.rules import build_profile_with_rules
from app.comparison.store import MemoryProfileSource
from app.evaluation.corpus import InMemoryRetriever
from app.numbers import number_appears, parse_number
from app.rag.citations import is_grounded, quote_in_text
from app.rag.loaders import PageText
from app.schemas.investigation import EvidenceItem, SufficiencyAssessment
from app.tools.contradiction import find_contradictions
from tests.helpers import ScriptedLLM, run_graph


# ------------------------------------------------------------------ helpers
def fact(field, quote, text=None, number=None, unit=None, page=1) -> ProfileFact:
    return ProfileFact(field=field, text=text, number=number, unit=unit, page=page, quote=quote)


def item(description, quote, quantity=None, price=None, amount=None, unit="GBP", page=1) -> ProfileLine:
    return ProfileLine(description=description, quantity=quantity, unit_price=price, amount=amount, unit=unit, page=page, quote=quote)


def read_by_model(name: str, kind: str, text: str, facts=(), lines=()):
    """A profile as a faithful model would return it, passed through the real grounding."""
    built, dropped = ground_extraction(
        ProfileExtraction(facts=list(facts), line_items=list(lines)),
        document_id=name, filename=f"{name}.pdf", document_type=kind, pages=[PageText(1, text)],
    )
    assert dropped == 0, "the test's own extraction is not grounded"
    return built


def read_by_rules(name: str, kind: str, text: str):
    return build_profile_with_rules(document_id=name, filename=f"{name}.pdf", document_type=kind, pages=[PageText(1, text)])


def kinds(profiles, **options) -> list[tuple[str, str]]:
    return [(finding.type, finding.severity) for finding in compare_profiles(profiles, **options)]


def order_for_gloves(quantity: int = 100):
    text = (
        "PURCHASE ORDER PO-4421\nDate: 2 March 2026\nBuyer: Northwind Facilities Ltd\nSupplier: Kestrel Packaging Ltd\n"
        f"Nitrile gloves box of 100   Qty {quantity}   Unit price £5.00   £{quantity * 5:.2f}\nTotal: £{quantity * 5:.2f}"
    )
    row = f"Nitrile gloves box of 100   Qty {quantity}   Unit price £5.00   £{quantity * 5:.2f}"
    return read_by_model("po", "purchase_order", text, [
        fact("document_number", "PURCHASE ORDER PO-4421", text="PO-4421"),
        fact("supplier", "Supplier: Kestrel Packaging Ltd", text="Kestrel Packaging Ltd"),
        fact("customer", "Buyer: Northwind Facilities Ltd", text="Northwind Facilities Ltd"),
        fact("total_amount", f"Total: £{quantity * 5:.2f}", number=quantity * 5, unit="GBP"),
    ], [item("Nitrile gloves box of 100", row, quantity, 5.0, quantity * 5.0)])


def invoice_for_gloves(number: str, quantity: int, supplier: str = "Kestrel Packaging Ltd", date: str = "10 March 2026"):
    row = f"Nitrile gloves box of 100   Qty {quantity}   Unit price £5.00   £{quantity * 5:.2f}"
    text = (
        f"INVOICE {number}\nInvoice date: {date}\nFrom: {supplier}\nTo: Northwind Facilities Ltd\n"
        f"Purchase order: PO-4421\n{row}\nTotal due: £{quantity * 5:.2f}"
    )
    return read_by_model(number, "invoice", text, [
        fact("document_number", f"INVOICE {number}", text=number),
        fact("purchase_order_reference", "Purchase order: PO-4421", text="PO-4421"),
        fact("document_date", f"Invoice date: {date}", text=date),
        fact("supplier", f"From: {supplier}", text=supplier),
        fact("total_amount", f"Total due: £{quantity * 5:.2f}", number=quantity * 5, unit="GBP"),
    ], [item("Nitrile gloves box of 100", row, quantity, 5.0, quantity * 5.0)])


# =========================================================== ordinary business
def test_a_partial_delivery_is_not_a_mismatch():
    """40 of the 100 boxes ordered are delivered and invoiced. Nothing is wrong with that."""
    pair = [order_for_gloves(100), invoice_for_gloves("INV-1", 40)]
    assert kinds(pair) == [] and kinds(pair, basic=True) == []


def test_billing_more_than_was_ordered_is_reported_on_one_invoice_or_over_several():
    assert kinds([order_for_gloves(100), invoice_for_gloves("INV-1", 120)]) == [("quantity_mismatch", "medium")]

    # 60 + 60: each invoice alone is a part delivery, together they bill 120 of the 100 ordered.
    [finding] = compare_profiles([order_for_gloves(100), invoice_for_gloves("INV-1", 60), invoice_for_gloves("INV-2", 60, date="20 March 2026")])
    assert (finding.type, finding.value_a, finding.value_b, finding.operands, finding.operation) == ("quantity_mismatch", 100, 120, [60, 60], "sum")
    assert "INV-1.pdf and INV-2.pdf together bill 120" in finding.description
    assert [cited.filename for cited in finding.cited] == ["po.pdf", "INV-1.pdf", "INV-2.pdf"]

    # 60 + 40 is the whole order, delivered in two parts.
    assert kinds([order_for_gloves(100), invoice_for_gloves("INV-1", 60), invoice_for_gloves("INV-2", 40, date="20 March 2026")]) == []


def test_goods_that_differ_only_by_a_short_code_are_matched_to_the_right_line():
    """Sizes M and L, listed in the other order on the invoice. It used to report four mismatches."""
    order_text = "PURCHASE ORDER PO-500\nSupplier: Kestrel Packaging Ltd\nNitrile gloves size M  100  £5.00\nNitrile gloves size L  50  £5.50\n"
    order = read_by_model("po", "purchase_order", order_text, [fact("document_number", "PURCHASE ORDER PO-500", text="PO-500")], [
        item("Nitrile gloves size M", "Nitrile gloves size M  100  £5.00", 100, 5.0),
        item("Nitrile gloves size L", "Nitrile gloves size L  50  £5.50", 50, 5.5),
    ])
    invoice_text = "INVOICE INV-2\nPurchase order: PO-500\nNitrile gloves size L  50  £5.50  £275.00\nNitrile gloves size M  100  £5.00  £500.00\nTotal due: £775.00"
    invoice = read_by_model("inv", "invoice", invoice_text, [
        fact("document_number", "INVOICE INV-2", text="INV-2"), fact("purchase_order_reference", "Purchase order: PO-500", text="PO-500"),
        fact("total_amount", "Total due: £775.00", number=775, unit="GBP"),
    ], [
        item("Nitrile gloves size L", "Nitrile gloves size L  50  £5.50  £275.00", 50, 5.5, 275.0),
        item("Nitrile gloves size M", "Nitrile gloves size M  100  £5.00  £500.00", 100, 5.0, 500.0),
    ])
    assert kinds([order, invoice]) == []


def test_singular_and_plural_are_the_same_goods_and_a_delivery_charge_is_not_an_unordered_item():
    order = read_by_model("po", "purchase_order", "PURCHASE ORDER PO-700\nA4 copier paper box  20  £18.00\nWine glass  100  £2.00\n", [
        fact("document_number", "PURCHASE ORDER PO-700", text="PO-700"),
    ], [item("A4 copier paper box", "A4 copier paper box  20  £18.00", 20, 18), item("Wine glass", "Wine glass  100  £2.00", 100, 2)])
    invoice_text = "INVOICE INV-8\nPurchase order: PO-700\nA4 copier paper box  20  £18.00  £360.00\nWine glasses  100  £2.00  £200.00\nDelivery: £45.00\nTotal due: £605.00"
    invoice = read_by_model("inv", "invoice", invoice_text, [
        fact("document_number", "INVOICE INV-8", text="INV-8"), fact("purchase_order_reference", "Purchase order: PO-700", text="PO-700"),
        fact("total_amount", "Total due: £605.00", number=605, unit="GBP"),
    ], [
        item("A4 copier paper box", "A4 copier paper box  20  £18.00  £360.00", 20, 18, 360),
        item("Wine glasses", "Wine glasses  100  £2.00  £200.00", 100, 2, 200), item("Delivery", "Delivery: £45.00", amount=45),
    ])
    assert kinds([order, invoice]) == []


# ---- arithmetic that real invoices explain themselves
DISCOUNT = (
    "INVOICE INV-3\nCartons  10  £60.00  £600.00\nLabels  20  £20.00  £400.00\nDiscount 10%: -£100.00\nSubtotal: £900.00\nVAT 20%: £180.00\nTotal due: £1,080.00",
    [fact("subtotal", "Subtotal: £900.00", number=900, unit="GBP"), fact("tax_amount", "VAT 20%: £180.00", number=180, unit="GBP"),
     fact("total_amount", "Total due: £1,080.00", number=1080, unit="GBP")],
    [item("Cartons", "Cartons  10  £60.00  £600.00", 10, 60, 600), item("Labels", "Labels  20  £20.00  £400.00", 20, 20, 400)],
)
TWO_TAXES = (
    "TAX INVOICE No. GST/26/014\nCotton yarn 40s  100 kg  ₹600.00  ₹60,000.00\nPolyester yarn  50 kg  ₹800.00  ₹40,000.00\nCGST 9%: ₹9,000.00\nSGST 9%: ₹9,000.00\nGrand total: ₹1,18,000.00",
    [fact("tax_amount", "CGST 9%: ₹9,000.00", number=9000, unit="INR"), fact("total_amount", "Grand total: ₹1,18,000.00", number=118000, unit="INR")],
    [item("Cotton yarn 40s", "Cotton yarn 40s  100 kg  ₹600.00  ₹60,000.00", 100, 600, 60000, "INR"),
     item("Polyester yarn", "Polyester yarn  50 kg  ₹800.00  ₹40,000.00", 50, 800, 40000, "INR")],
)
ROUNDING = (
    "INVOICE INV-5\nConsulting hours  2.5  £19.99  £49.98\nSite visit  1  £100.00  £100.00\nTotal due: £149.98",
    [fact("total_amount", "Total due: £149.98", number=149.98, unit="GBP")],
    [item("Consulting hours", "Consulting hours  2.5  £19.99  £49.98", 2.5, 19.99, 49.98), item("Site visit", "Site visit  1  £100.00  £100.00", 1, 100, 100)],
)
LINE_DISCOUNT = (
    "INVOICE INV-6\nCartons  Qty 10  Unit £100.00  Disc 10%  £900.00\nLabels  Qty 1  Unit £50.00  Disc 0%  £50.00\nTotal due: £950.00",
    [fact("total_amount", "Total due: £950.00", number=950, unit="GBP")],
    [item("Cartons", "Cartons  Qty 10  Unit £100.00  Disc 10%  £900.00", 10, 100, 900), item("Labels", "Labels  Qty 1  Unit £50.00  Disc 0%  £50.00", 1, 50, 50)],
)
DEPOSIT = (
    "INVOICE B-9\nDesign work: £6,000\nInstallation: £4,000\nTotal: £10,000\nLess deposit paid: £3,000\nBalance due: £7,000",
    [fact("total_amount", "Balance due: £7,000", number=7000, unit="GBP")],
    [item("Design work", "Design work: £6,000", amount=6000), item("Installation", "Installation: £4,000", amount=4000)],
)


@pytest.mark.parametrize("name, document", [
    ("a discount above the subtotal", DISCOUNT), ("two taxes and no subtotal", TWO_TAXES), ("rounding to the cent", ROUNDING),
    ("a discount column on the line", LINE_DISCOUNT), ("a deposit already paid", DEPOSIT),
])
def test_arithmetic_that_the_page_explains_is_not_reported(name, document):
    text, facts, lines = document
    built = read_by_model("inv", "invoice", text, facts, lines)
    assert compare_profiles([built]) == [], name


def test_arithmetic_that_nothing_on_the_page_explains_is_still_reported():
    # Lines of 600 and 400, but a subtotal of 1,100: the VAT is there and does not explain it.
    text = "INVOICE INV-9\nCartons  10  £60.00  £600.00\nLabels  20  £20.00  £400.00\nSubtotal: £1,100.00\nVAT 20%: £220.00\nTotal due: £1,320.00"
    wrong = read_by_model("inv", "invoice", text, [
        fact("subtotal", "Subtotal: £1,100.00", number=1100, unit="GBP"), fact("tax_amount", "VAT 20%: £220.00", number=220, unit="GBP"),
        fact("total_amount", "Total due: £1,320.00", number=1320, unit="GBP"),
    ], DISCOUNT[2])
    [finding] = compare_profiles([wrong])
    assert (finding.type, finding.severity, finding.value_a, finding.value_b) == ("line_items_total_mismatch", "medium", 1000, 1100)

    # A tax is added, never taken off: lines of 1,000 and VAT of 200 do not explain a total of 800.
    text = "INVOICE INV-10\nCartons  10  £60.00  £600.00\nLabels  20  £20.00  £400.00\nVAT 20%: £200.00\nTotal due: £800.00"
    backwards = read_by_model("inv", "invoice", text, [fact("total_amount", "Total due: £800.00", number=800, unit="GBP")], DISCOUNT[2])
    assert kinds([backwards]) == [("line_items_total_mismatch", "medium")]

    # A discount mentioned without an amount could explain a difference: said, and said softly.
    text = "INVOICE INV-11\nCartons  10  £60.00  £600.00\nLabels  20  £20.00  £400.00\nLess 10% trade discount\nTotal due: £900.00"
    unquantified = read_by_model("inv", "invoice", text, [fact("total_amount", "Total due: £900.00", number=900, unit="GBP")], DISCOUNT[2])
    [finding] = compare_profiles([unquantified])
    assert finding.severity == "low" and "without an amount" in finding.description

    # A line that is simply multiplied wrongly.
    text = "INVOICE INV-12\nSite survey  2  £450.00  £950.00\nTotal due: £950.00"
    multiplied = read_by_model("inv", "invoice", text, [fact("total_amount", "Total due: £950.00", number=950, unit="GBP")],
                               [item("Site survey", "Site survey  2  £450.00  £950.00", 2, 450, 950)])
    assert ("line_arithmetic_mismatch", "medium") in kinds([multiplied])


# ================================================== which documents belong together
CONTRACT = (
    'MAINTENANCE CONTRACT Contract No. MC-2026-011.\nThis contract is made on 5 January 2026 between Harlow Print Works Ltd '
    '(the "Client") and Redbridge Lifts Ltd (the "Supplier").\nThe term starts on 1 February 2026 and continues for twelve months.\n'
    "The total contract value is £40,000."
)


def invoice_text(number="INV-311", supplier="Redbridge Lifts Ltd", reference="Contract reference: MC-2026-011\n", amount="£40,000", date="12 March 2026") -> str:
    return (
        f"INVOICE\nInvoice No. {number}\nInvoice date: {date}\nFrom: {supplier}\nTo: Harlow Print Works Ltd\n{reference}"
        f"Lift maintenance: {amount}\nTotal amount due: {amount}\n"
    )


def test_an_invoice_from_another_supplier_that_cites_nothing_is_left_alone():
    """It used to be checked against the only contract in the folder and accused of coming from the wrong supplier."""
    contract = read_by_rules("contract", "contract", CONTRACT)
    other = read_by_rules("other", "invoice", invoice_text("INV-88", "Tolland Electrical Ltd", reference="", amount="£900"))
    assert compare_profiles([contract, other], basic=True) == []
    assert unrelated_pairs([contract, other]) == {("other", "contract")}

    # The same invoice citing a contract number that is not in the folder: still simply another piece of business.
    elsewhere = read_by_rules("other", "invoice", invoice_text("INV-88", "Tolland Electrical Ltd", reference="Contract reference: TE-77\n", amount="£900"))
    assert compare_profiles([contract, elsewhere], basic=True) == []


def test_an_invoice_that_cites_the_contract_is_checked_against_it_whoever_sends_it():
    contract = read_by_rules("contract", "contract", CONTRACT)
    impostor = read_by_rules("inv", "invoice", invoice_text("INV-355", "Tolland Electrical Ltd"))
    [finding] = compare_profiles([contract, impostor])
    assert (finding.type, finding.severity) == ("party_mismatch", "high")
    assert unrelated_pairs([contract, impostor]) == set()  # it says itself that it belongs to this contract

    # From the right supplier, but citing another number: worth saying.
    miscited = read_by_rules("inv", "invoice", invoice_text("INV-356", reference="Contract reference: MC-2026-019\n"))
    assert kinds([contract, miscited]) == [("reference_mismatch", "medium")]


def test_a_name_with_a_word_added_is_a_reason_to_look_not_the_same_company():
    """Missed before: "Redbridge Lifts Services Ltd" invoicing under a contract with "Redbridge Lifts Ltd"."""
    assert compare_names("Redbridge Lifts Ltd", "Redbridge Lifts Services Ltd") == "short_form"
    contract = read_by_rules("contract", "contract", CONTRACT)
    look_alike = read_by_rules("inv", "invoice", invoice_text("INV-355", "Redbridge Lifts Services Ltd"))
    [finding] = compare_profiles([contract, look_alike])
    assert (finding.type, finding.severity, finding.text_a, finding.text_b) == ("party_mismatch", "low", "Redbridge Lifts Ltd", "Redbridge Lifts Services Ltd")
    assert "shorter form of the other" in finding.description


def test_monthly_invoices_under_a_contract_are_not_mismatches_until_they_exceed_it():
    contract = read_by_rules("contract", "contract", CONTRACT)
    months = [read_by_rules(f"inv{m}", "invoice", invoice_text(f"INV-{m}", amount="£3,333.33", date=f"5 {name} 2026"))
              for m, name in ((4, "April"), (5, "May"), (6, "June"))]
    assert compare_profiles([contract, *months], basic=True) == []

    # £15,000 three times is £45,000 under a £40,000 contract: no single invoice shows it.
    large = [read_by_rules(f"inv{m}", "invoice", invoice_text(f"INV-{m}", amount="£15,000", date=f"5 {name} 2026"))
             for m, name in ((4, "April"), (5, "May"), (6, "June"))]
    [finding] = compare_profiles([contract, *large], basic=True)
    assert (finding.type, finding.value_a, finding.value_b) == ("amount_mismatch", 40000, 45000)
    assert "together bill £45,000" in finding.description

    # One invoice above the contract value is reported on its own, as before.
    over = read_by_rules("inv", "invoice", invoice_text(amount="£58,000"))
    assert kinds([contract, over], basic=True) == [("amount_mismatch", "high")]


# ==================================================================== missed before
def test_a_line_billed_at_more_than_was_ordered_is_found_without_quantities():
    order = "PURCHASE ORDER\nPurchase Order No. PO-9102\nBuyer: Orchard Foods Ltd\nSupplier: Kestrel Packaging Ltd\nPallet wrapping machine service: £1,350\nSpare film rolls: £400\n"
    invoice = "INVOICE\nInvoice No. INV-3310\nFrom: Kestrel Packaging Ltd\nTo: Orchard Foods Ltd\nPurchase order: PO-9102\nPallet wrapping machine service: £2,350\nSpare film rolls: £400\nTotal amount due: £2,750\n"
    [finding] = compare_profiles([read_by_rules("po", "purchase_order", order), read_by_rules("inv", "invoice", invoice)])
    assert (finding.type, finding.value_a, finding.value_b) == ("line_amount_mismatch", 1350, 2350)


def test_a_price_in_another_currency_is_reported():
    order = "PURCHASE ORDER\nPurchase Order No. PO-7781\nSupplier: Merrow Medical Supplies\n40 sterilisation pouches cartons at $205 each\n"
    invoice = "INVOICE\nInvoice No. INV-129\nFrom: Merrow Medical Supplies\nPurchase order: PO-7781\n40 sterilisation pouches cartons at £205 each: £8,200\nTotal amount due: £8,200\n"
    [finding] = compare_profiles([read_by_rules("po", "purchase_order", order), read_by_rules("inv", "invoice", invoice)])
    assert (finding.type, finding.unit_a, finding.unit_b) == ("unit_mismatch", "USD", "GBP") and "currencies differ" in finding.description


def plain_invoice(name: str, number: str, date: str, supplier: str = "Tolland Electrical Ltd", total: str = "£3,150"):
    return read_by_rules(name, "invoice", f"INVOICE\nInvoice No. {number}\nInvoice date: {date}\nFrom: {supplier}\nTo: Aldgate Storage Ltd\nTotal amount due: {total}\n")


def test_a_duplicate_is_found_when_the_suppliers_name_is_retyped():
    [finding] = check_duplicates([plain_invoice("a", "INV-880", "2 April 2026"), plain_invoice("b", "INV-880", "2 April 2026", "Tolland Electricals Ltd")])
    assert (finding.type, finding.severity) == ("duplicate_invoice", "high")
    assert "written differently" in finding.description
    # Two really different suppliers may use the same number.
    assert check_duplicates([plain_invoice("a", "INV-880", "2 April 2026"), plain_invoice("b", "INV-880", "2 April 2026", "Fenwick Cleaning Services Ltd")]) == []


def test_two_dates_that_only_share_a_reading_are_two_days():
    """03/02/2026 and 02/03/2026: 3 February and 2 March, or the other way round. Never the same day."""
    feb, mar = plain_invoice("a", "INV-201", "03/02/2026"), plain_invoice("b", "INV-202", "02/03/2026")
    assert check_duplicates([feb, mar]) == []
    # The same day, written the same way or once in words: the same bill under a new number is worth a look.
    assert [finding.type for finding in check_duplicates([feb, plain_invoice("b", "INV-202", "03/02/2026")])] == ["duplicate_invoice"]
    assert [finding.type for finding in check_duplicates([feb, plain_invoice("b", "INV-202", "3 February 2026")])] == ["duplicate_invoice"]


def test_a_credit_note_is_not_a_second_copy_of_the_invoice_it_corrects():
    invoice = read_by_rules("inv", "invoice", "INVOICE\nInvoice No. INV-300\nFrom: Kestrel Packaging Ltd\nTo: Northwind Ltd\nInvoice date: 20 March 2026\nPackaging: £1,200.00\nTotal: £1,200.00\n")
    credit = read_by_rules("cn", "invoice", "CREDIT NOTE CN-12\nFrom: Kestrel Packaging Ltd\nTo: Northwind Ltd\nDate: 25 March 2026\nCredit against Invoice No. INV-300\nPackaging returned: -£1,200.00\nTotal: -£1,200.00\n")
    assert credit.credit_note and credit.total.number == -1200 and (credit.number is None or credit.number.text == "CN-12")
    assert not invoice.credit_note
    assert compare_profiles([invoice, credit]) == []


# ========================================================== the rule-based reader
def test_a_price_with_three_decimals_is_not_read_as_thousands():
    assert parse_number("0.125") == [0.125]
    text = "Invoice No. INV-1\nFrom: Acme Labels Ltd\n10,000 shipping labels at $0.125 each: $1,250.00\n2,000 gallons diesel at $3.459 per gallon: $6,918.00\nTotal due: $8,168.00\n"
    built = read_by_rules("inv", "invoice", text)
    assert [(line.quantity, line.unit_price, line.amount) for line in built.line_items] == [(10000, 0.125, 1250), (2000, 3.459, 6918)]
    assert compare_profiles([built]) == []
    # Without a line total nothing settles what "$3.459" means: the price stays unread rather than misread.
    alone = read_by_rules("inv", "invoice", "Invoice No. INV-2\n2,000 gallons diesel at $3.459 per gallon\n")
    assert [(line.quantity, line.unit_price) for line in alone.line_items] == [(2000, None)]


def test_labels_are_read_for_the_kind_of_document_they_stand_on():
    order = read_by_rules("po", "purchase_order", "PURCHASE ORDER\nPO Number: PO-4421\nRequired delivery date: 30 April 2026\nOrder date: 2 March 2026\nFrom: Northwind Facilities Ltd\nSupplier: Kestrel Packaging Ltd\n")
    # On an order "From:" is the buyer; and its date is the order date, not the delivery date that comes first.
    assert (order.customer.text, order.supplier.text, order.date.text) == ("Northwind Facilities Ltd", "Kestrel Packaging Ltd", "2 March 2026")

    invoice = read_by_rules("inv", "invoice", "INVOICE\nInvoice No. K-78\nOrder date: 2 March 2026\nInvoice date: 20 March 2026\nFrom: Kestrel Packaging Ltd\n4 Harbour Road, Fareham PO16 7GZ\nTo: Accounts Payable, Northwind Facilities Ltd\nPurchase order: PO-4421\n")
    assert invoice.date.text == "20 March 2026"
    assert invoice.customer.text == "Northwind Facilities Ltd"  # not "Accounts Payable"
    assert invoice.purchase_order_reference.text == "PO-4421"  # not the "16" of the postcode PO16
    assert compare_profiles([order, invoice]) == []


def test_only_sentences_about_the_contract_itself_give_its_term():
    text = (
        'Agreement No. SA-2026-007\nThis Agreement is made on 5 January 2026 between Brightline Engineering Services Ltd (the "Company") '
        'and Northwind Facilities Ltd (the "Customer").\nThe prices in Schedule 1 are fixed until 31 March 2026.\n'
        "Invoicing begins on 1 June 2026 for the first quarter.\nThe term starts on 1 February 2026 and continues for twenty four months.\n"
    )
    contract = read_by_rules("contract", "contract", text)
    assert contract.term_start.text == "1 February 2026" and contract.term_months.number == 24
    assert contract.term_end is None  # "fixed until 31 March 2026" is about prices
    # "the Company" may be either side: no role is guessed, so a correct invoice from that company raises nothing.
    assert [party.role for party in contract.parties] == [None, "customer"]
    invoice = read_by_rules("inv", "invoice", "INVOICE\nInvoice No. INV-102\nFrom: Brightline Engineering Services Ltd\nTo: Northwind Facilities Ltd\nInvoice date: 3 September 2026\nContract reference: SA-2026-007\nTotal due: £3,000\n")
    assert compare_profiles([contract, invoice]) == []


def test_two_identical_lines_are_two_lines():
    built = read_by_rules("inv", "invoice", "INVOICE\nInvoice No. INV-41\nQuarterly inspection: £900\nCall-out visit: £350\nCall-out visit: £350\nTotal amount due: £1,600\n")
    assert [line.amount for line in built.line_items] == [900, 350, 350] and compare_profiles([built]) == []


# ================================================================== grounding
@pytest.mark.parametrize("value, text, expected", [
    (1250, "Unit price: € 12,50", False),  # twelve and a half
    (123456, "Total due: £1,234.56", False),
    (25, "VAT at 2.5%", False),
    (5800000, "Total: 58,000.00", False),
    (58000, "Total: 58,000.00", True),
    (150000, "PKR 1,50,000", True),  # lakh grouping still reads
    (1500000, "EUR 1.500.000", True),
])
def test_a_number_only_counts_as_written_if_it_is_that_number(value, text, expected):
    assert number_appears(value, text) is expected


PAGE = (
    "INVOICE Invoice No. INV-102\nTotal amount due: £40,000\nPayment terms: 30 days.\n"
    "Note: invoices above the approval limit of £58,000 need two signatures.\nIssued 3 March 2026. Goods received 28 March 2026."
)


def test_a_quote_cannot_be_stitched_from_a_label_and_a_figure_that_stand_apart():
    assert not quote_in_text("Total amount due: ... £58,000", PAGE, 2, strict_gaps=True)
    assert not quote_in_text("Issued ... 28 March 2026", PAGE, strict_gaps=True)
    assert quote_in_text("invoices above ... limit of £58,000", PAGE, 2, strict_gaps=True)  # a few words left out of one sentence
    assert not is_grounded("Total amount due: ... £58,000", 58000, PAGE)
    assert is_grounded("Total amount due: £40,000", 40000, PAGE)

    stitched = ProfileExtraction(facts=[
        fact("total_amount", "Total amount due: ... £58,000", number=58000, unit="GBP"),
        fact("document_date", "Issued ... 28 March 2026", text="28 March 2026"),
        fact("subtotal", "£40,000", number=4000000, unit="GBP"),
    ], line_items=[])
    built, dropped = ground_extraction(stitched, document_id="d", filename="Invoice_102.pdf", document_type="invoice", pages=[PageText(1, PAGE)])
    assert dropped == 3 and built.total is None and built.date is None and built.subtotal is None


# ================================================== the comparison of totals from the search
def stated(identifier: str, filename: str, kind: str, amount: float) -> EvidenceItem:
    return EvidenceItem(
        id=identifier, claim=f"{filename} states {amount}", key="total_amount", value_number=amount, unit="GBP",
        chunk_id=identifier, document_id=f"doc:{filename}", filename=filename, document_type=kind, page=1,
    )


def test_totals_are_only_held_against_each_other_where_that_means_something():
    contract = stated("E1", "Contract.pdf", "contract", 40000)
    # An invoice above the contract value: the classic finding.
    [over] = find_contradictions([contract, stated("E2", "Invoice_A.pdf", "invoice", 58000)])
    assert (over.type, over.value_a, over.value_b) == ("amount_mismatch", 40000, 58000)
    # An invoice below it is an instalment, and two invoices are two bills.
    assert find_contradictions([contract, stated("E2", "Invoice_A.pdf", "invoice", 3333.33)]) == []
    assert find_contradictions([stated("E2", "Invoice_A.pdf", "invoice", 900), stated("E3", "Invoice_B.pdf", "invoice", 3150)]) == []
    # Two contracts that state different values for the same thing still contradict each other.
    assert [item.type for item in find_contradictions([contract, stated("E4", "Contract_copy.pdf", "contract", 45000)])] == ["amount_mismatch"]


# ==================================================================== in the graph
CONTRACT_PAGES = [
    'MAINTENANCE CONTRACT Contract No. MC-2026-011. This contract is made on 5 January 2026 between Harlow Print Works Ltd '
    '(the "Client") and Redbridge Lifts Ltd (the "Supplier") for the maintenance of three passenger lifts.',
    "4. Charges and Payment. 4.1 The total contract value is £40,000 for the Term, exclusive of VAT. 4.2 Payment terms: 30 days from the date of invoice.",
]
INVOICE_311 = [
    "INVOICE Invoice No. INV-311. Invoice date: 20 February 2026. From: Redbridge Lifts Ltd. To: Harlow Print Works Ltd. "
    "Contract reference: MC-2026-011. Lift maintenance, annual: £40,000. Total amount due: £40,000. Payment terms: 30 days from the date of invoice."
]


def other_invoice(amount: str) -> list[str]:
    return [
        "INVOICE Invoice No. INV-88. Invoice date: 1 March 2026. From: Tolland Electrical Ltd. To: Harlow Print Works Ltd. "
        f"Emergency lighting test and repairs at the print works. Total amount due: {amount}. Payment terms: 30 days from the date of invoice."
    ]


async def investigate(question: str, *documents: tuple[str, list[str]]) -> dict:
    retriever = InMemoryRetriever()
    for filename, pages in documents:
        retriever.add_text(filename, pages)
    llm = ScriptedLLM()
    state = await run_graph(question, retriever, llm, profiles=MemoryProfileSource(retriever.chunks, llm))
    return state["report"]


@pytest.mark.parametrize("amount", ["£900", "£52,000"])
async def test_an_unrelated_invoice_in_the_library_does_not_make_a_matching_invoice_look_wrong(amount):
    """Found in review: risk went from low to critical because another supplier's invoice lay in the same folder.
    £52,000 is above the contract value: without the profiles the search-level comparison would report that too."""
    report = await investigate(
        "Does invoice INV-311 match the maintenance contract MC-2026-011?",
        ("Maintenance_Contract.txt", CONTRACT_PAGES), ("Invoice_311.txt", INVOICE_311), ("Invoice_88.txt", other_invoice(amount)),
    )
    assert report["contradictions"] == [] and report["risk_level"] == "low" and report["status"] == "no_issues_found"


async def test_a_part_delivery_ends_as_low_risk_in_an_investigation():
    order = "PURCHASE ORDER\nPurchase Order No. PO-7781\nDate: 3 June 2026\nBuyer: Castlegate Dental Group\nSupplier: Merrow Medical Supplies\n40 sterilisation pouches cartons at $205 each\n"
    invoice = (
        "INVOICE\nInvoice No. INV-129\nInvoice date: 18 June 2026\nFrom: Merrow Medical Supplies\nTo: Castlegate Dental Group\nPurchase order: PO-7781\n"
        "First part delivery.\n15 sterilisation pouches cartons at $205 each: $3,075\nTotal amount due: $3,075\n"
    )
    report = await investigate("Does invoice INV-129 match purchase order PO-7781?", ("Purchase_Order_PO-7781.txt", [order]), ("Invoice_INV-129.txt", [invoice]))
    assert report["contradictions"] == [] and report["risk_level"] == "low"


async def test_passages_fetched_only_to_cite_a_comparison_do_not_use_up_the_search():
    """Found in review: with the comparison on, the follow-up search the agent asked for was no longer run."""
    question = "Was the extra charge on invoice INV-9 approved?"
    filler = "Extra charge on invoice INV-9: was the extra charge approved? Extra charge approved invoice INV-9 note {n}."
    contract = [
        'SERVICE CONTRACT Contract No. SC-1. This contract is made on 5 January 2026 between Aldgate Storage Ltd (the "Client") and Fenwick Cleaning Services Ltd (the "Supplier").',
        "The term starts on 1 March 2026 and continues for twelve months.",
        filler.format(n=1) + " The total contract value is £400.", filler.format(n=2), filler.format(n=3), filler.format(n=11), filler.format(n=12),
    ]
    order = [
        "PURCHASE ORDER Purchase Order No. PO-5. Date: 2 March 2026. Buyer: Aldgate Storage Ltd. Supplier: Fenwick Cleaning Services Ltd.",
        "10 floor polish drums at £40 each", filler.format(n=4), filler.format(n=5), filler.format(n=13), filler.format(n=14),
    ]
    invoice = [
        "INVOICE Invoice No. INV-9. From: Tolland Electrical Ltd. To: Aldgate Storage Ltd. Contract reference: SC-1. Purchase order: PO-5.",
        "Invoice date: 10 February 2026.", "14 floor polish drums at £45 each: £630", filler.format(n=6) + " Total amount due: £630.",
        filler.format(n=7), filler.format(n=8), filler.format(n=15), filler.format(n=16),
    ]
    memo = ["MEMO. Variation order VO-7 for additional cleaning rounds was signed by the Finance Director on 1 February 2026."]
    retriever = InMemoryRetriever()
    retriever.add_text("Contract_SC-1.txt", contract, "contract")
    retriever.add_text("Purchase_Order_PO-5.txt", order, "purchase_order")
    retriever.add_text("Invoice_INV-9.txt", invoice, "invoice")
    retriever.add_text("Memo.txt", memo, "other")
    llm = ScriptedLLM({SufficiencyAssessment: [
        SufficiencyAssessment(sufficient=False, missing=["A signed variation order"], follow_up_queries=["variation order VO-7 signed Finance Director"]),
        SufficiencyAssessment(sufficient=True, missing=[], follow_up_queries=[]),
    ]})
    state = await run_graph(question, retriever, llm, profiles=MemoryProfileSource(retriever.chunks, llm))
    assert state["retrieval_rounds"] == 2
    assert not any("limit of" in note for note in state["limitations"])
    cited_only = [chunk for chunk in state["retrieved_documents"] if chunk["matched_by"] == ["comparison"]]
    searched = [chunk for chunk in state["retrieved_documents"] if chunk["matched_by"] != ["comparison"]]
    assert cited_only and len(searched) <= state_settings().MAX_RETRIEVED_CHUNKS
    references = [chunk["ref"] for chunk in state["retrieved_documents"]]
    assert len(references) == len(set(references))  # every passage keeps one reference number of its own


def state_settings():
    from app.config import get_settings

    return get_settings()


def test_the_columns_of_a_table_row_are_read_as_separate_numbers_too():
    """Found in review: "Widget A 5 250 1,250.00" was read as 5,250 and 1,250, so the line was silently not checked."""
    row = "Widget A 5 250 1,250.00"
    assert number_appears(5, row) and number_appears(250, row) and number_appears(1250, row) and number_appears(5250, row)
    assert not number_appears(52, row) and not number_appears(2501, row)
    built = read_by_model("inv", "invoice", row, [], [item("Widget A", row, 5, 250, 1250)])
    assert [(line.quantity, line.unit_price, line.amount) for line in built.line_items] == [(5, 250, 1250)]
