"""Document comparison: what a second, independent review found in the fixes of the first.

The first review ended with the rule "do not accuse what you do not model".
The second one looked at what that rule had been turned into, and found that
each piece of leniency could be used as a hiding place:

    * an invoice from a look-alike supplier that leaves out the contract
      number was no longer compared with anything - and nobody was told,
    * the words "credit note" near the top of an invoice switched every check off,
    * a charge counted as a line and again as "an adjustment" explained any total.

It also found ordinary business that was still accused: a joint venture, a
trading name, a price per thousand, a second contract in the same folder.

The rule that came out of it: leniency needs bounds, and silence needs a reason.
"""

from __future__ import annotations

import pytest

from app.comparison.engine import compare, named_pairs
from app.comparison.names import compare_names, may_belong_together, same_identifier
from app.comparison.service import compare_documents  # noqa: F401  (imported so a renamed entry point fails here)
from app.numbers import number_appears, number_stated
from app.patterns import find_amounts
from app.rag.citations import is_grounded
from tests.test_comparison_review import (
    CONTRACT_PAGES,
    INVOICE_311,
    fact,
    investigate,
    item,
    kinds,
    other_invoice,
    read_by_model,
    read_by_rules,
)

CONTRACT = (
    'MAINTENANCE CONTRACT\nContract No. MC-2026-011\nThis contract is made on 5 January 2026 between Harlow Print Works Ltd (the "Client") '
    'and Redbridge Lifts Ltd (the "Supplier").\nThe term starts on 1 February 2026 and continues for twelve months.\n'
    "The total contract value is £40,000 for the Term, exclusive of VAT.\nPayment terms: 30 days from the date of invoice."
)


def contract(text: str = CONTRACT, name: str = "contract"):
    return read_by_rules(name, "contract", text)


def invoice(
    name: str = "inv", *, heading: str = "INVOICE", number: str = "INV-311", supplier: str = "Redbridge Lifts Ltd",
    date: str = "20 February 2026", amount: str = "3,333.33", reference: str = "Contract reference: MC-2026-011\n",
    to: str = "To: Harlow Print Works Ltd", what: str = "Lift maintenance, February", extra: str = "",
):
    return read_by_rules(name, "invoice", (
        f"{heading}\nInvoice No. {number}\nInvoice date: {date}\nFrom: {supplier}\n{to}\n{reference}{extra}"
        f"{what}: £{amount}\nTotal amount due: £{amount}"
    ))


# ============================================================ silence needs a reason
def test_a_look_alike_supplier_that_leaves_out_the_contract_number_is_still_compared():
    """Found in the second review: "Redbridge Lift Services Ltd" against "Redbridge Lifts Ltd", no reference
    quoted, £58,000 on a £40,000 contract - no finding at all, and "no issues found" in the report."""
    impostor = invoice(supplier="Redbridge Lift Services Ltd", reference="", amount="58,000.00")
    assert compare_names("Redbridge Lift Services Ltd", "Redbridge Lifts Ltd") == "different"  # not the same company ...
    assert may_belong_together("Redbridge Lift Services Ltd", "Redbridge Lifts Ltd")  # ... but the pair to look at
    assert not may_belong_together("Tolland Electrical Ltd", "Redbridge Lifts Ltd")
    assert not may_belong_together("Harlow Cleaning Ltd", "Harlow Lifts Ltd")  # a shared town is not a shared name
    assert kinds([contract(), impostor], basic=True) == [("amount_mismatch", "high"), ("party_mismatch", "high")]


def test_documents_the_person_names_together_are_compared_whatever_the_names_say():
    other = invoice(supplier="Tolland Electrical Ltd", reference="", amount="52,000.00", number="INV-88")
    profiles = [contract(), other]
    # Left alone by default - and the result says so.
    alone = compare(profiles, basic=True)
    assert alone.findings == [] and alone.unrelated == {("inv", "contract")}
    [note] = alone.notes
    assert "inv.pdf was not checked against the contract contract.pdf" in note and "Tolland Electrical Ltd" in note

    question = "Check invoice INV-88 against the maintenance contract MC-2026-011."
    assert named_pairs(question, profiles) == {("inv", "contract")}
    assert named_pairs("Check invoice INV-88.", profiles) == set()  # the contract is not named
    asked = compare(profiles, basic=True, forced=named_pairs(question, profiles))
    assert [(finding.type, finding.severity) for finding in asked.findings] == [("amount_mismatch", "high"), ("party_mismatch", "high")]
    assert asked.notes == [] and asked.unrelated == set()
    # One invoice selected together with one contract says the same without words.
    selected = compare(profiles, basic=True, assume_related=True)
    assert [finding.type for finding in selected.findings] == ["amount_mismatch", "party_mismatch"]


async def test_an_investigation_says_which_invoice_it_left_alone():
    report = await investigate(
        "Does invoice INV-311 match the maintenance contract MC-2026-011?",
        ("Maintenance_Contract.txt", CONTRACT_PAGES), ("Invoice_311.txt", INVOICE_311), ("Invoice_88.txt", other_invoice("£52,000")),
    )
    assert report["contradictions"] == [] and report["risk_level"] == "low"
    assert any("Invoice_88.txt was not checked against the contract Maintenance_Contract.txt" in note for note in report["limitations"])


async def test_an_investigation_compares_the_two_documents_the_question_names():
    report = await investigate(
        "Check invoice INV-88 against the maintenance contract MC-2026-011.",
        ("Maintenance_Contract.txt", CONTRACT_PAGES), ("Invoice_88.txt", other_invoice("£52,000")),
    )
    assert sorted(finding["type"] for finding in report["contradictions"]) == ["amount_mismatch", "party_mismatch"]
    assert report["status"] == "review_required"


# ===================================================================== credit notes
@pytest.mark.parametrize("heading", [
    "INVOICE\nQueries and credit note requests: accounts@castlegate.example",
    "TAX INVOICE\nNo credit note will be issued after 30 days.",
    "INVOICE (credit note requests: accounts@castlegate.example)",
])
def test_the_words_credit_note_near_the_top_do_not_switch_the_checks_off(heading):
    """Found in the second review: two identical £58,000 invoices from the wrong supplier gave five high
    findings without the phrase, and none with it."""
    twice = [invoice(name, heading=heading, supplier="Castlegate Engineering Ltd", amount="58,000.00", number="INV-310") for name in ("a", "b")]
    assert not twice[0].credit_note
    found = kinds([contract(), *twice], basic=True)
    assert found.count(("party_mismatch", "high")) == 2 and found.count(("amount_mismatch", "high")) == 2
    assert ("duplicate_invoice", "high") in found


def test_a_credit_note_is_one_by_its_heading_and_is_still_checked_for_who_sends_it():
    note = read_by_rules("cn", "invoice", (
        "CREDIT NOTE\nCredit Note No. CN-12\nDate: 2 March 2026\nFrom: Castlegate Engineering Ltd\nTo: Harlow Print Works Ltd\n"
        "Contract reference: MC-2026-011\nCredit against Invoice No. INV-311\nLift maintenance, February: -£3,333.33\nTotal credit: -£3,333.33"
    ))
    assert note.credit_note and note.number.text == "CN-12"
    result = compare([contract(), note], basic=True)
    assert [(finding.type, finding.severity) for finding in result.findings] == [("party_mismatch", "high")]
    assert any("calls itself a credit note" in text for text in result.notes)

    # "CREDIT NOTE" over a positive amount that is due is an invoice in disguise.
    disguised = read_by_rules("cn2", "invoice", (
        "CREDIT NOTE\nInvoice No. INV-900\nInvoice date: 20 February 2026\nFrom: Redbridge Lifts Ltd\nTo: Harlow Print Works Ltd\n"
        "Contract reference: MC-2026-011\nLift maintenance: £95,000.00\nTotal amount due: £95,000.00"
    ))
    assert not disguised.credit_note
    assert ("amount_mismatch", "high") in kinds([contract(), disguised], basic=True)


# =========================================================== one company, several names
def test_either_member_of_a_joint_venture_may_send_the_invoice():
    joint = contract(CONTRACT.replace(
        'Redbridge Lifts Ltd (the "Supplier")', '(1) Redbridge Lifts Ltd and (2) Castlegate Engineering Ltd (together the "Contractor")'
    ))
    assert [(party.role, party.text) for party in joint.parties] == [
        ("customer", "Harlow Print Works Ltd"), ("supplier", "Redbridge Lifts Ltd"), ("supplier", "Castlegate Engineering Ltd"),
    ]
    for supplier in ("Redbridge Lifts Ltd", "Castlegate Engineering Ltd"):
        assert kinds([joint, invoice(supplier=supplier)]) == []
        assert kinds([joint, invoice(supplier=supplier, reference="")]) == []
    [stranger] = compare([joint, invoice(supplier="Tolland Electrical Ltd")]).findings
    assert stranger.type == "party_mismatch" and "Redbridge Lifts Ltd and Castlegate Engineering Ltd" in stranger.description

    # The same through a model's reading, which states both suppliers.
    text = CONTRACT.replace('Redbridge Lifts Ltd (the "Supplier")', '(1) Redbridge Lifts Ltd and (2) Castlegate Engineering Ltd (together the "Contractor")')
    by_model = read_by_model("contract", "contract", text, [
        fact("document_number", "Contract No. MC-2026-011", text="MC-2026-011"),
        fact("supplier", "(1) Redbridge Lifts Ltd", text="Redbridge Lifts Ltd"),
        fact("supplier", '(2) Castlegate Engineering Ltd (together the "Contractor")', text="Castlegate Engineering Ltd"),
    ])
    assert len(by_model.parties_with("supplier")) == 2
    assert kinds([by_model, invoice(supplier="Castlegate Engineering Ltd")]) == []


@pytest.mark.parametrize("letterhead", ["Sunshine Cleaning", "ABC Facilities Ltd", "ABC Facilities Ltd t/a Sunshine Cleaning"])
def test_a_company_may_use_its_trading_name(letterhead):
    trading = contract(CONTRACT.replace("Redbridge Lifts Ltd", "ABC Facilities Ltd trading as Sunshine Cleaning"))
    assert trading.supplier.text == "ABC Facilities Ltd trading as Sunshine Cleaning"
    assert compare_names(trading.supplier.text, letterhead) == "same"
    assert kinds([trading, invoice(supplier=letterhead)]) == []
    result = compare([trading, invoice(supplier=letterhead, reference="")])
    assert result.findings == [] and result.linked == {("inv", "contract")}  # recognised as belonging together, not skipped


@pytest.mark.parametrize("to", [
    "To: Estates Department, Harlow Print Works Ltd", "To: Facilities Manager, Harlow Print Works Ltd",
    "To: Mr J. Okafor, Harlow Print Works Ltd", "To: Harlow Print Works Ltd, Accounts Payable",
    "Bill to: Head Office, Harlow Print Works Ltd", "To: Accounts Payable\nHarlow Print Works Ltd",
])
def test_a_department_or_a_person_in_the_address_is_not_the_customer(to):
    addressed = invoice(to=to)
    assert addressed.customer.text == "Harlow Print Works Ltd"
    assert kinds([contract(), addressed]) == []


def test_an_address_that_names_no_company_leaves_the_customer_unread():
    assert invoice(to="To: Head Office, 12 High Street").customer is None
    assert invoice(to="To: Accounts Payable").customer is None


# ============================================================== leniency needs bounds
def courier_invoice():
    amounts = [118.00, 95.00, 143.00, 77.00, 131.00, 88.00, 109.00, 152.00]
    rows = [f"Delivery to site {chr(65 + index)}, consignment {4410 + index}: £{amount:,.2f}" for index, amount in enumerate(amounts)]
    text = (
        "INVOICE\nInvoice No. INV-8801\nInvoice date: 3 March 2026\nFrom: Swiftline Couriers Ltd\nTo: Northwind Facilities Ltd\n"
        + "\n".join(rows) + "\nVAT 20%: £182.60\nTotal due: £1,095.60"
    )
    return text, rows, amounts


def test_an_invoice_whose_lines_are_deliveries_adds_up():
    """Found in the second review: eight "Delivery to site ..." lines were counted as adjustments, the VAT came
    ninth and was no longer tried - so an honest courier's invoice "did not add up"."""
    text, rows, amounts = courier_invoice()
    by_rules = read_by_rules("courier", "invoice", text)
    assert len(by_rules.line_items) == 8 and [(value.text, value.number) for value in by_rules.adjustments] == [("add", 182.6)]
    assert kinds([by_rules]) == []
    by_model = read_by_model("courier", "invoice", text, [
        fact("tax_amount", "VAT 20%: £182.60", number=182.60, unit="GBP"), fact("total_amount", "Total due: £1,095.60", number=1095.60, unit="GBP"),
    ], [item(row.split(":")[0], row, amount=amount) for row, amount in zip(rows, amounts)])
    assert kinds([by_model]) == []


def cartons_invoice(extra: str, extra_amount: float, total: float, footer: str = ""):
    text = (
        "INVOICE INV-7\nInvoice date: 10 March 2026\nFrom: Kestrel Packaging Ltd\nPurchase order: PO-600\n"
        f"Printed cartons  1,000  £1.20  £1,200.00\n{extra}: £{extra_amount:,.2f}\nTotal due: £{total:,.2f}{footer}"
    )
    return read_by_model("inv", "invoice", text, [
        fact("document_number", "INVOICE INV-7", text="INV-7"), fact("purchase_order_reference", "Purchase order: PO-600", text="PO-600"),
        fact("supplier", "From: Kestrel Packaging Ltd", text="Kestrel Packaging Ltd"),
        fact("total_amount", f"Total due: £{total:,.2f}", number=total, unit="GBP"),
    ], [
        item("Printed cartons", "Printed cartons  1,000  £1.20  £1,200.00", 1000, 1.2, 1200),
        item(extra, f"{extra}: £{extra_amount:,.2f}", amount=extra_amount),
    ])


def test_a_charge_explains_the_total_once_not_as_a_line_and_again_as_an_adjustment():
    """Found in the second review: cartons 1,200 + "Handling: £5,000" with a stated total of 11,200 gave no finding."""
    assert kinds([cartons_invoice("Handling", 5000, 6200)]) == []  # the honest version adds up
    assert kinds([cartons_invoice("Handling", 5000, 11200)]) == [("line_items_total_mismatch", "medium")]
    assert kinds([cartons_invoice("Delivery", 45, 1290)]) == [("line_items_total_mismatch", "low")]  # a gap a tax could fill
    freight = read_by_rules("inv", "invoice", "INVOICE\nInvoice No. INV-7\nFrom: Kestrel Packaging Ltd\nPrinted cartons: £1,200.00\nFreight: £800.00\nTotal due: £2,800.00")
    assert kinds([freight]) == [("line_items_total_mismatch", "medium")]


def test_the_tax_is_counted_once_however_often_the_page_states_it():
    """Found in the second review: lines 1,000 + VAT 200 with a total of 1,400 passed on a page with a VAT summary box."""
    def vat_invoice(total: str, box: str = ""):
        text = (
            "INVOICE INV-21\nFrom: Kestrel Packaging Ltd\nCartons  10  £60.00  £600.00\nLabels  20  £20.00  £400.00\n"
            f"VAT 20%: £200.00\nTotal due: £{total}{box}"
        )
        return read_by_model("inv", "invoice", text, [
            fact("tax_amount", "VAT 20%: £200.00", number=200, unit="GBP"), fact("total_amount", f"Total due: £{total}", number=float(total.replace(",", "")), unit="GBP"),
        ], [item("Cartons", "Cartons  10  £60.00  £600.00", 10, 60, 600), item("Labels", "Labels  20  £20.00  £400.00", 20, 20, 400)])

    box = "\nVAT analysis: standard rate 20% on £1,000.00 = £200.00"
    assert kinds([vat_invoice("1,200.00", box)]) == []
    assert kinds([vat_invoice("1,400.00", box)]) == [("line_items_total_mismatch", "medium")]
    # Two different taxes of the same size (CGST and SGST) are still two.
    gst = read_by_rules("gst", "invoice", (
        "TAX INVOICE\nInvoice No. GST-9\nFrom: Kestrel Packaging Pvt Ltd\nCartons: ₹60,000.00\nLabels: ₹40,000.00\n"
        "CGST 9%: ₹9,000.00\nSGST 9%: ₹9,000.00\nGrand total: ₹1,18,000.00"
    ))
    assert kinds([gst]) == []


def test_a_condition_is_not_a_charge():
    """Found in the second review: "Free delivery on orders over £750.00" explained a total that was £750 too high."""
    text = (
        "INVOICE INV-22\nFrom: Kestrel Packaging Ltd\nCartons: £600.00\nLabels: £400.00\nTotal due: £1,750.00\n"
        "Free delivery on orders over £750.00."
    )
    built = read_by_rules("inv", "invoice", text)
    assert built.adjustments == [] and kinds([built]) == [("line_items_total_mismatch", "medium")]


def test_an_incidental_charge_is_only_incidental_while_it_is_small():
    """Found in the second review: "Handling and administration: £5,000" on a £1,200 order passed as a delivery charge."""
    order = read_by_model("po", "purchase_order", (
        "PURCHASE ORDER PO-600\nDate: 2 March 2026\nBuyer: Northwind Facilities Ltd\nSupplier: Kestrel Packaging Ltd\nPrinted cartons  1,000  £1.20\n"
    ), [
        fact("document_number", "PURCHASE ORDER PO-600", text="PO-600"), fact("supplier", "Supplier: Kestrel Packaging Ltd", text="Kestrel Packaging Ltd"),
    ], [item("Printed cartons", "Printed cartons  1,000  £1.20", 1000, 1.2)])
    assert kinds([order, cartons_invoice("Delivery", 45, 1245)]) == []
    assert kinds([order, cartons_invoice("Handling and administration", 5000, 6200)]) == [("unlisted_line_item", "medium")]
    assert kinds([order, cartons_invoice("Transport of consultants and project management", 25000, 26200)]) == [("unlisted_line_item", "medium")]


# ============================================================ units of measure, layouts
def test_a_price_per_thousand_is_not_an_arithmetic_error():
    """Found in the second review: "5,000 labels at £12.00 per thousand: £60.00" was reported as 5,000 x £12 = £60,000."""
    by_rules = read_by_rules("inv", "invoice", (
        "INVOICE\nInvoice No. INV-501\nFrom: Kestrel Packaging Ltd\n5,000 printed labels at £12.00 per thousand: £60.00\n"
        "2,000 swing tags at £30.00 per 1,000: £60.00\n2,500 cable ties at £3.20 per 100: £80.00\n"
        "7.5 hours site supervision at £85.00 per hour: £637.50\nTotal due: £837.50"
    ))
    assert [(line.quantity, line.unit_price, line.amount) for line in by_rules.line_items] == [
        (5000, None, 60), (2000, None, 60), (2500, None, 80), (7.5, 85, 637.5),
    ]
    assert kinds([by_rules]) == []
    row = "Printed labels   Qty 5,000   Price per 1,000 £12.00   £60.00"
    by_model = read_by_model("inv", "invoice", f"INVOICE INV-501\n{row}\nTotal due: £60.00", [
        fact("total_amount", "Total due: £60.00", number=60, unit="GBP"),
    ], [item("Printed labels", row, 5000, 12, 60)])
    assert kinds([by_model]) == []
    # An ordinary line with a wrong amount is still found.
    wrong = read_by_rules("inv", "invoice", "INVOICE\nInvoice No. INV-502\n40 cartons at £5.00 each: £260.00\nTotal due: £260.00")
    assert kinds([wrong]) == [("line_arithmetic_mismatch", "medium")]


def test_boxes_on_the_order_and_pieces_on_the_invoice_are_a_note_not_two_accusations():
    order = read_by_model("po", "purchase_order", (
        "PURCHASE ORDER PO-4421\nSupplier: Kestrel Packaging Ltd\nNitrile gloves, box of 100   Qty 10 boxes   £50.00 per box   £500.00\nTotal: £500.00"
    ), [
        fact("document_number", "PURCHASE ORDER PO-4421", text="PO-4421"), fact("supplier", "Supplier: Kestrel Packaging Ltd", text="Kestrel Packaging Ltd"),
    ], [item("Nitrile gloves, box of 100", "Nitrile gloves, box of 100   Qty 10 boxes   £50.00 per box   £500.00", 10, 50, 500)])
    billed = read_by_model("inv", "invoice", (
        "INVOICE INV-1\nFrom: Kestrel Packaging Ltd\nPurchase order: PO-4421\nNitrile gloves   Qty 1,000 each   £0.50   £500.00\nTotal due: £500.00"
    ), [
        fact("purchase_order_reference", "Purchase order: PO-4421", text="PO-4421"), fact("supplier", "From: Kestrel Packaging Ltd", text="Kestrel Packaging Ltd"),
    ], [item("Nitrile gloves", "Nitrile gloves   Qty 1,000 each   £0.50   £500.00", 1000, 0.5, 500)])
    [note] = compare([order, billed]).findings
    assert (note.type, note.severity) == ("quantity_mismatch", "low") and "different units" in note.description


@pytest.mark.parametrize("body", [
    "Labour: £1,000.00\nMaterials: £500.00\nVAT 20%: £300.00\nLess CIS deduction 20% on labour: £200.00\nTotal due: £1,600.00",
    "Cartons: £3,000.00\nLabels: £2,000.00\nCarried forward: £5,000.00\nBrought forward: £5,000.00\nPallets: £2,000.00\nTotal due: £7,000.00",
    "Contract sum: £50,000.00\nStage 2 (20%): £10,000.00\nPreviously invoiced: £10,000.00\nTotal due: £10,000.00",
    "Cartons: £600.00\nLabels: £400.00\nThis invoice: £1,000.00\nBrought forward from statement: £300.00\nTotal amount due: £1,300.00",
    "2 hours labour at £45.00 per hour: £120.00\nMinimum call-out charge applies.\nTotal due: £120.00",
])
def test_deductions_and_running_totals_are_not_billed_lines(body):
    """Found in the second review: "Less CIS deduction" and "Carried forward" were added up as if they were goods."""
    built = read_by_rules("inv", "invoice", f"INVOICE\nInvoice No. INV-600\nInvoice date: 3 March 2026\nFrom: Kestrel Builders Ltd\n{body}")
    assert kinds([built]) == []


def test_an_amount_in_millions_is_left_unread_by_the_rules_and_understood_by_the_check():
    """Found in the second review: "USD 1.2 million" was read as $1.20 and held against an invoice for $300,000."""
    assert find_amounts("The total contract value is USD 1.2 million for the Term.") == []
    assert number_appears(1_200_000, "USD 1.2 million") and not number_appears(1.2, "USD 1.2 million")
    assert number_appears(40_000, "£40k") and number_appears(1_500_000, "£1.5m") and not number_appears(1_500_000, "1.5m of cable")
    assert [hit.unit for hit in find_amounts("Total amount due: KWD 39,000")] == ["KWD"] and find_amounts("cad 40 drawings") == []


def test_a_contract_that_renews_has_no_last_day_to_be_late_for():
    evergreen = contract(
        'SERVICE AGREEMENT\nAgreement No. SA-2024-07\nThis agreement is made on 12 March 2024 between Harlow Print Works Ltd (the "Client") '
        'and Redbridge Lifts Ltd (the "Supplier").\nThis agreement starts on 1 April 2024 and continues for twelve months, and thereafter renews '
        "automatically for successive periods of twelve months unless either party gives 90 days notice.\nThe annual fee is £40,000."
    )
    later = invoice(date="3 July 2026", reference="Contract reference: SA-2024-07\n")
    assert evergreen.renews and kinds([evergreen, later]) == []
    fixed = contract(CONTRACT.replace("MC-2026-011", "SA-2024-07").replace("1 February 2026", "1 April 2024"))
    assert not fixed.renews and kinds([fixed, later]) == [("date_outside_term", "low")]


# ======================================================================== duplicates
def test_the_same_number_or_the_same_total_is_not_always_the_same_bill():
    """Found in the second review: two sites billed on one day, a numbering that restarts every year,
    and a pro-forma with its final invoice were all reported as duplicates."""
    sites = [
        invoice("site_a", number="INV-201", amount="1,250.00", date="1 March 2026", reference="Purchase order: PO-771\n", what="Lift maintenance March, Harlow site"),
        invoice("site_b", number="INV-202", amount="1,250.00", date="1 March 2026", reference="Purchase order: PO-772\n", what="Lift maintenance March, Stevenage site"),
    ]
    assert kinds(sites) == []
    without_orders = [invoice(name, number=number, amount="1,250.00", date="1 March 2026", reference="", what=what) for name, number, what in (
        ("site_a", "INV-201", "Lift maintenance March, Harlow site"), ("site_b", "INV-202", "Lift maintenance March, Stevenage site"),
    )]
    assert kinds(without_orders) == []  # other lines: other work
    same_work = [invoice(name, number=number, amount="1,250.00", date="1 March 2026", reference="") for name, number in (("a", "INV-201"), ("b", "INV-207"))]
    assert kinds(same_work) == [("duplicate_invoice", "medium")]  # the real thing is still found

    yearly = [invoice("y2025", number="001", amount="1,250.00", date="3 March 2025", reference=""), invoice("y2026", number="001", amount="1,250.00", date="2 March 2026", reference="")]
    [restart] = compare(yearly).findings
    assert restart.severity == "low" and "starts its numbering again" in restart.description
    resubmitted = [invoice("a", number="001", amount="1,250.00", date="3 March 2026", reference=""), invoice("b", number="001", amount="1,250.00", date="3 March 2026", reference="")]
    assert kinds(resubmitted) == [("duplicate_invoice", "high")]

    pro_forma = read_by_rules("proforma", "invoice", (
        "PRO-FORMA INVOICE\nInvoice No. INV-640\nDate: 2 March 2026\nFrom: Kestrel Packaging Ltd\nTo: Northwind Facilities Ltd\n"
        "Payment in advance is requested.\nPrinted cartons: £1,200.00\nTotal due: £1,200.00"
    ))
    final = read_by_rules("final", "invoice", (
        "TAX INVOICE\nInvoice No. INV-640\nInvoice date: 9 March 2026\nFrom: Kestrel Packaging Ltd\nTo: Northwind Facilities Ltd\n"
        "Printed cartons: £1,200.00\nTotal due: £1,200.00"
    ))
    assert pro_forma.pro_forma and not final.pro_forma and kinds([pro_forma, final]) == []

    unknown = [
        read_by_rules("s1", "invoice", "Tolland Electrical Ltd\nINVOICE\nInvoice No. 1001\nInvoice date: 2 March 2026\nEmergency lighting test: £900.00\nTotal due: £900.00"),
        read_by_rules("s2", "invoice", "Kestrel Packaging Ltd\nINVOICE\nInvoice No. 1001\nInvoice date: 9 April 2026\nPrinted cartons: £1,200.00\nTotal due: £1,200.00"),
    ]
    assert kinds(unknown) == []


def test_an_invoice_is_checked_against_the_latest_version_of_its_order_whichever_was_uploaded_first():
    def order(name: str, revision: int, quantity: int):
        row = f"Nitrile gloves box of 100   Qty {quantity}   Unit price £5.00   £{quantity * 5:.2f}"
        text = f"PURCHASE ORDER PO-4421 Revision {revision}\nDate: {2 + revision} March 2026\nSupplier: Kestrel Packaging Ltd\n{row}\nTotal: £{quantity * 5:.2f}"
        return read_by_model(name, "purchase_order", text, [
            fact("document_number", "PURCHASE ORDER PO-4421", text="PO-4421"), fact("supplier", "Supplier: Kestrel Packaging Ltd", text="Kestrel Packaging Ltd"),
            fact("document_date", f"Date: {2 + revision} March 2026", text=f"{2 + revision} March 2026"),
            fact("total_amount", f"Total: £{quantity * 5:.2f}", number=quantity * 5, unit="GBP"),
        ], [item("Nitrile gloves box of 100", row, quantity, 5, quantity * 5)])

    row = "Nitrile gloves box of 100   Qty 150   Unit price £5.00   £750.00"
    billed = read_by_model("inv", "invoice", f"INVOICE INV-9\nInvoice date: 20 March 2026\nFrom: Kestrel Packaging Ltd\nPurchase order: PO-4421\n{row}\nTotal due: £750.00", [
        fact("purchase_order_reference", "Purchase order: PO-4421", text="PO-4421"), fact("supplier", "From: Kestrel Packaging Ltd", text="Kestrel Packaging Ltd"),
        fact("total_amount", "Total due: £750.00", number=750, unit="GBP"),
    ], [item("Nitrile gloves box of 100", row, 150, 5, 750)])
    first, revised = order("rev0", 0, 100), order("rev1", 1, 150)
    for profiles in ([first, revised, billed], [revised, first, billed]):
        result = compare(profiles, basic=True)
        assert result.findings == [] and result.linked == {("inv", "rev1")} and ("inv", "rev0") in result.unrelated


def test_a_reference_number_is_not_matched_inside_a_longer_number():
    assert same_identifier("2026-004", "Contract-2026-004") and same_identifier("4421", "PO-4421")
    assert not same_identifier("1001", "PO-21001")


# ========================================================================= grounding
@pytest.mark.parametrize("quote, value, page", [
    ("Total amount due: £58", 58, "INVOICE INV-1\nTotal amount due: £58,000\n"),
    ("40,000", 40000, "The total contract value is £140,000 for the Term."),
    ("40,000 for the Term", 40000, "The total contract value is £140,000 for the Term."),
    ("Total due: £1,250", 1250, "Total due: £1,250.75"),
    ("Gesamtbetrag: 40.000", 40, "Gesamtbetrag: 40.000,00 EUR"),
    ("Total amount due: ... £58,000", 58000, "INVOICE INV-1\nTotal amount due: see below\nCredit limit on this account: £58,000\nAmount now payable £40,000"),
    ("Total ... £58,000", 58000, "Description Amount\nTotal\nOrder value approved: £58,000\nNet £40,000"),
])
def test_a_quote_cut_out_of_a_longer_number_or_stitched_across_lines_states_nothing(quote, value, page):
    assert not is_grounded(quote, value, page)


@pytest.mark.parametrize("quote, value, page", [
    ("Total contract value ........ £40,000", 40000, "Total contract value ........ £40,000\nPayment terms: 30 days"),
    ("Total contract value ... £40,000", 40000, "Total contract value . . . . . . . £40,000"),
    ("Total … £40,000", 40000, "Total ........ £40,000"),
    ("The total contract value is £40,000", 40000, "The total contract\nvalue is £40,000 for the Term."),
    ("Lift maintenance | ... | £1,000", 1000, "Lift maintenance | 4 | £250 | £1,000"),
    ("USD 1.2 million", 1_200_000, "The total contract value is USD 1.2 million for the Term."),
    ("Total amount due: £58,000", 58000, "INVOICE INV-1\nTotal amount due: £58,000\n"),
])
def test_honest_quotes_are_still_accepted(quote, value, page):
    assert is_grounded(quote, value, page)


def test_a_negative_value_needs_a_quote_that_says_minus():
    assert number_stated(-1200, "Credit: -£1,200") and number_stated(-1200, "Credit: (£1,200)") and number_stated(-1200, "£1,200 CR")
    assert not number_stated(-1200, "Total: £1,200") and number_stated(1200, "Total: £1,200")
    text = "CREDIT NOTE CN-1\nTotal: £1,200.00"
    from app.comparison.llm import ProfileExtraction, ground_extraction
    from app.rag.loaders import PageText

    built, dropped = ground_extraction(
        ProfileExtraction(facts=[fact("total_amount", "Total: £1,200.00", number=-1200, unit="GBP")], line_items=[]),
        document_id="cn", filename="cn.pdf", document_type="invoice", pages=[PageText(1, text)],
    )
    assert dropped == 1 and built.total is None
