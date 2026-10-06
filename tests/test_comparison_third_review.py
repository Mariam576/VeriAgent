"""Document comparison: what a third independent review found.

Three things run through these cases.

1. A guess must give way to knowledge. "An invoice that equals another
   contract's total probably belongs to that contract" is a guess for when
   nothing else is known. Used where the invoice cites its contract, it let an
   over-billed invoice pass.
2. A document's word about itself is a claim. "t/a Redbridge Lifts" on an
   invoice does not make the sender Redbridge Lifts.
3. Money is compared like with like: the amount before tax against a value
   that was agreed before tax, several invoices added up, a ceiling treated
   as the total it limits.
"""

from __future__ import annotations

import pytest

from app.comparison.engine import compare, named_in
from app.comparison.names import compare_claimed, compare_names, same_identifier
from app.numbers import number_stated
from app.patterns import find_amounts
from app.rag.citations import is_grounded
from app.tools.contradiction import find_contradictions
from tests.test_comparison_review import fact, investigate, item, kinds, read_by_model, read_by_rules, stated
from tests.test_comparison_second_review import CONTRACT, contract, invoice


# ============================================================ a guess gives way to knowledge
def test_an_invoice_tied_to_its_contract_is_measured_against_it_whatever_another_contract_totals():
    """Found in the third review: an invoice for £58,000 on a £40,000 contract passed, because a second
    contract in the folder happened to be worth £58,000."""
    maintenance, cleaning = stated("E1", "Maintenance.pdf", "contract", 40000), stated("E2", "Cleaning.pdf", "contract", 58000)
    bill = stated("E3", "Invoice.pdf", "invoice", 58000)
    tied = {("doc:Invoice.pdf", "doc:Maintenance.pdf")}
    apart = {("doc:Invoice.pdf", "doc:Cleaning.pdf")}
    [over] = find_contradictions([maintenance, cleaning, bill], linked=tied, unrelated=apart)
    assert (over.source_a, over.value_a, over.value_b, over.severity) == ("Maintenance.pdf", 40000, 58000, "high")
    # Without any knowledge of what belongs together, the guess still spares the honest pair.
    assert find_contradictions([maintenance, cleaning, bill]) == []


async def test_the_same_in_an_investigation():
    second = [
        'CLEANING CONTRACT Contract No. CC-2026-004. This contract is made on 12 January 2026 between Harlow Print Works Ltd (the "Client") '
        'and Fenwick Cleaning Services Ltd (the "Supplier") for daily office cleaning.',
        "3. Charges. 3.1 The total contract value is £58,000 for the Term, exclusive of VAT.",
    ]
    first = [
        'MAINTENANCE CONTRACT Contract No. MC-2026-011. This contract is made on 5 January 2026 between Harlow Print Works Ltd (the "Client") '
        'and Redbridge Lifts Ltd (the "Supplier").',
        "4. Charges. 4.1 The total contract value is £40,000 for the Term, exclusive of VAT.",
    ]
    bill = [
        "INVOICE Invoice No. INV-311. Invoice date: 20 February 2026. From: Redbridge Lifts Ltd. To: Harlow Print Works Ltd. "
        "Contract reference: MC-2026-011. Lift maintenance, annual: £58,000. Total amount due: £58,000."
    ]
    report = await investigate(
        "Does invoice INV-311 match the maintenance contract MC-2026-011?",
        ("Maintenance_Contract.txt", first), ("Cleaning_Contract.txt", second), ("Invoice_311.txt", bill),
    )
    assert [(finding["type"], finding["severity"]) for finding in report["contradictions"]] == [("amount_mismatch", "high")]


def test_an_amendment_does_not_take_the_place_of_the_contract():
    """Found in the third review: a later paper with the contract's number was taken for its latest version -
    it states no value, so an invoice far above the contract went through."""
    amendment = read_by_rules("amendment", "contract", (
        'AMENDMENT No. 1\nContract No. MC-2026-011\nThis amendment is made on 1 June 2026 between Harlow Print Works Ltd (the "Client") '
        'and Redbridge Lifts Ltd (the "Supplier").\nClause 4.2 is replaced by: Payment terms: 30 days from the date of invoice.'
    ))
    over = compare([contract(), amendment, invoice(amount="58,000.00", date="20 June 2026")], basic=True)
    assert [(finding.type, finding.source_a) for finding in over.findings] == [("amount_mismatch", "contract.pdf")]
    assert compare([contract(), amendment, invoice()], basic=True).findings == []  # an ordinary February invoice


def test_a_contract_without_a_number_is_still_the_contract():
    """Found in the third review: the invoice quoted a number, the contract showed none - the pair was dropped without a word."""
    unnumbered = contract(CONTRACT.replace("Contract No. MC-2026-011\n", ""))
    assert unnumbered.number is None
    assert kinds([unnumbered, invoice(amount="58,000.00")], basic=True) == [("amount_mismatch", "high")]
    assert ("party_mismatch", "high") in kinds([unnumbered, invoice(supplier="Redbridge Lift Services Ltd")])


def test_an_invoice_that_cites_another_contract_is_not_measured_against_this_one():
    elsewhere = invoice(reference="Contract reference: RC-2026-002\n", amount="60,000.00", what="Lift car refurbishment, stage 1")
    result = compare([contract(), elsewhere], basic=True)
    assert [(finding.type, finding.severity) for finding in result.findings] == [("reference_mismatch", "medium")]
    assert result.unrelated == {("inv", "contract")}  # nor at the level of totals from the search


def test_what_was_left_unchecked_is_always_said():
    second = contract(CONTRACT.replace("MC-2026-011", "MC-2026-020").replace("£40,000", "£90,000"), name="contract2")
    unsure = compare([contract(), second, invoice(reference="", amount="58,000.00")], basic=True)
    assert unsure.findings == []
    assert any("could belong to more than one" in note and "contract.pdf, contract2.pdf" in note for note in unsure.notes)

    strangers = [invoice(f"other{index}", supplier=f"Tolland Electrical {name} Ltd", reference="", number=f"INV-9{index}", date=f"{index + 1} March 2026")
                 for index, name in enumerate(["North", "South", "East", "West", "Central", "Coastal"])]
    many = compare([contract(), *strangers])
    assert len(many.notes) == 4 and many.notes[-1].startswith("3 more document(s) were left unchecked")


def test_a_file_called_invoice_is_not_named_by_every_question_about_invoices():
    """Found in the third review: "invoice.txt" counted as named, and another supplier's invoice was forced against the contract."""
    plain = invoice("plain", supplier="Tolland Electrical Ltd", reference="", number="INV-88")
    plain.filename = "invoice.txt"
    numbered = invoice("numbered")
    numbered.filename = "Invoice_311.txt"
    agreement = contract()
    agreement.filename = "contract.txt"
    question = "Does invoice INV-311 match the contract MC-2026-011?"
    assert named_in(question, [agreement, plain, numbered]) == {"contract", "numbered"}
    assert named_in("Compare invoice.txt with contract.txt", [agreement, plain, numbered]) == {"contract", "plain"}
    assert named_in("Look at Invoice_311 please", [agreement, plain, numbered]) == {"numbered"}


# ============================================================ a document's word about itself
def test_a_trading_name_counts_when_the_contract_states_it_not_when_the_invoice_claims_it():
    """Found in the third review: any company passed as the supplier by writing "t/a <supplier>" on its invoice."""
    assert compare_names("Tolland Electrical Ltd t/a Redbridge Lifts", "Redbridge Lifts Ltd") == "same"  # the symmetric question
    assert compare_claimed("Tolland Electrical Ltd t/a Redbridge Lifts", "Redbridge Lifts Ltd") == "similar"  # the claim
    assert compare_claimed("Sunshine Cleaning", "ABC Facilities Ltd trading as Sunshine Cleaning") == "same"
    assert compare_claimed("ABC Facilities Ltd t/a Sunshine Cleaning", "ABC Facilities Ltd trading as Sunshine Cleaning") == "same"
    assert compare_claimed("Tolland Electrical Ltd", "Redbridge Lifts Ltd") == "different"
    for claimed in ("Tolland Electrical Ltd t/a Redbridge Lifts", "Castlegate Holdings Ltd trading as Redbridge Lifts Ltd"):
        [finding] = compare([contract(), invoice(supplier=claimed)]).findings
        assert (finding.type, finding.severity) == ("party_mismatch", "medium")


def test_a_pro_forma_copy_of_an_invoice_is_not_waved_through():
    def copy(name: str, heading: str):
        return read_by_rules(name, "invoice", (
            f"{heading}\nInvoice No. INV-640\nInvoice date: 9 March 2026\nFrom: Kestrel Packaging Ltd\nTo: Northwind Facilities Ltd\n"
            "Printed cartons: £1,200.00\nTotal due: £1,200.00"
        ))

    [finding] = compare([copy("final", "TAX INVOICE"), copy("copy", "PROFORMA INVOICE")]).findings
    assert finding.type == "duplicate_invoice" and "Pay only one of the two" in finding.description


@pytest.mark.parametrize("sentence, renews", [
    ("This contract renews automatically for successive periods of twelve months.", True),
    ("This contract does not renew automatically.", False),
    ("This contract will not automatically renew.", False),
])
def test_does_not_renew_is_not_renews(sentence, renews):
    assert contract(CONTRACT + "\n" + sentence).renews is renews


# ================================================================== money, like with like
def test_an_invoice_for_the_agreed_value_plus_tax_bills_what_was_agreed():
    """Found in the third review: £40,000 "exclusive of VAT" against "VAT 20%: £8,000, total due £48,000" was a high finding."""
    with_vat = invoice(amount="40,000.00", extra="", what="Lift maintenance, annual")
    text = (
        "INVOICE\nInvoice No. INV-311\nInvoice date: 20 February 2026\nFrom: Redbridge Lifts Ltd\nTo: Harlow Print Works Ltd\n"
        "Contract reference: MC-2026-011\nLift maintenance, annual: £{net}\nVAT 20%: £{vat}\nTotal amount due: £{total}"
    )
    honest = read_by_rules("inv", "invoice", text.format(net="40,000.00", vat="8,000.00", total="48,000.00"))
    result = compare([contract(), honest], basic=True)
    assert result.findings == [] and result.net_totals == {"inv": 40000.0}
    assert kinds([contract(), with_vat], basic=True) == []
    # More than the agreed value before tax is still found, and a "tax" of half the bill is not a tax.
    over = read_by_rules("inv", "invoice", text.format(net="45,000.00", vat="9,000.00", total="54,000.00"))
    assert kinds([contract(), over], basic=True) == [("amount_mismatch", "high")]
    absurd = read_by_rules("inv", "invoice", text.format(net="40,000.00", vat="18,000.00", total="58,000.00"))
    assert kinds([contract(), absurd], basic=True) == [("amount_mismatch", "high")]
    # The comparison of totals from the search is told the amount before tax as well.
    agreed, billed = stated("E1", "Contract.pdf", "contract", 40000), stated("E2", "Invoice.pdf", "invoice", 48000)
    assert find_contradictions([agreed, billed], net_totals={"doc:Invoice.pdf": 40000.0}) == []
    assert len(find_contradictions([agreed, billed])) == 1


def test_two_taxes_and_a_carriage_charge_do_not_make_an_order_overbilled():
    order = read_by_rules("po", "purchase_order", (
        "PURCHASE ORDER\nPO No. LG/PO/3321\nDate: 01-03-2026\nBuyer: Lahore Garments Ltd\nSupplier: Indus Textile Mills (Pvt) Ltd\n"
        "Cotton yarn 30/1 carded | 250 | Rs. 1,150.00 | Rs. 2,87,500.00\nPolyester staple fibre | 100 | Rs. 420.00 | Rs. 42,000.00\n"
        "Order value: Rs. 3,29,500.00 plus applicable sales tax"
    ))
    billed = read_by_rules("inv", "invoice", (
        "SALES TAX INVOICE\nInvoice No: STI-2026/0456\nDate: 15-03-2026\nSupplier: Indus Textile Mills (Pvt) Ltd\nBuyer: Lahore Garments Ltd\nPO No. LG/PO/3321\n"
        "Cotton yarn 30/1 carded | 250 | Rs. 1,150.00 | Rs. 2,87,500.00\nPolyester staple fibre | 100 | Rs. 420.00 | Rs. 42,000.00\n"
        "Value excluding sales tax: Rs. 3,29,500.00\nSales tax @ 18%: Rs. 59,310.00\nFurther tax @ 3%: Rs. 9,885.00\nTotal amount payable: Rs. 3,98,695.00/-"
    ))
    assert len(billed.line_items) == 2 and billed.subtotal.number == 329500
    assert kinds([order, billed], basic=True) == []


def test_several_invoices_are_added_up_in_an_investigation_too():
    """Found in the third review: three invoices of £15,000 on a £40,000 contract ended as "no issues found"."""
    three = [invoice(f"inv{index}", number=f"INV-40{index}", amount="15,000.00", date=f"{index + 1} March 2026", what=f"Lift maintenance, stage {index + 1}") for index in range(3)]
    [together] = compare([contract(), *three]).findings  # basic=False: as inside an investigation
    assert together.type == "amount_mismatch" and "together bill £45,000" in together.description
    assert compare([contract(), *three[:2]]).findings == []  # £30,000 of £40,000


def test_an_order_billed_in_full_twice_is_said_quietly_and_a_monthly_one_is_not():
    order = read_by_rules("po", "purchase_order", (
        "PURCHASE ORDER\nPurchase Order No. PO-9102\nDate: 2 March 2026\nBuyer: Orchard Foods Ltd\nSupplier: Kestrel Packaging Ltd\n"
        "Pallet wrapping machine service: £1,350.00\nSpare film rolls: £400.00"
    ))

    def bill(name: str, number: str, date: str):
        return read_by_rules(name, "invoice", (
            f"INVOICE\nInvoice No. {number}\nInvoice date: {date}\nFrom: Kestrel Packaging Ltd\nTo: Orchard Foods Ltd\nPurchase order: PO-9102\n"
            "Pallet wrapping machine service: £1,350.00\nSpare film rolls: £400.00\nTotal amount due: £1,750.00"
        ))

    twice = compare([order, bill("march", "INV-3310", "12 March 2026"), bill("april", "INV-3344", "14 April 2026")])
    assert [(finding.type, finding.severity) for finding in twice.findings] == [("line_amount_mismatch", "low")] * 2
    assert "Unless the purchase order is a standing one" in twice.findings[0].description
    standing = read_by_rules("po", "purchase_order", order_text := (
        "PURCHASE ORDER\nPurchase Order No. PO-9102\nDate: 2 March 2026\nBuyer: Orchard Foods Ltd\nSupplier: Kestrel Packaging Ltd\n"
        "Monthly machine service: £1,350.00"
    ))
    assert order_text and compare([standing, *[
        read_by_rules(name, "invoice", (
            f"INVOICE\nInvoice No. {number}\nInvoice date: {date}\nFrom: Kestrel Packaging Ltd\nPurchase order: PO-9102\n"
            "Monthly machine service: £1,350.00\nTotal amount due: £1,350.00"
        )) for name, number, date in (("march", "INV-1", "12 March 2026"), ("april", "INV-2", "14 April 2026"))
    ]]).findings == []


def test_a_subtotal_that_does_not_lead_to_the_total_is_found():
    """Found in the third review: lines 1,000, "Subtotal: 1,000", "VAT: 200", "Total due: 1,900" passed."""
    def bill(total: str, extra: str = ""):
        return read_by_rules("inv", "invoice", (
            f"INVOICE\nInvoice No. INV-22\nFrom: Kestrel Packaging Ltd\nCartons: £600.00\nLabels: £400.00\nSubtotal: £1,000.00\n"
            f"VAT 20%: £200.00\n{extra}Total due: £{total}"
        ))

    assert kinds([bill("1,200.00")]) == []
    [wrong] = compare([bill("1,900.00")]).findings
    assert wrong.type == "line_items_total_mismatch" and "subtotal of £1,000 and a total of £1,900" in wrong.description
    assert kinds([bill("1,235.00", "Shipping: £35.00\n")]) == []  # a charge after the subtotal is on the page


def test_one_line_against_the_total_is_checked_when_a_model_read_the_document():
    text = "INVOICE INV-311\nFrom: Redbridge Lifts Ltd\nLift maintenance, February: £3,333.33\nTotal amount due: £9,333.33"
    by_model = read_by_model("inv", "invoice", text, [
        fact("total_amount", "Total amount due: £9,333.33", number=9333.33, unit="GBP"),
    ], [item("Lift maintenance, February", "Lift maintenance, February: £3,333.33", amount=3333.33)])
    assert kinds([by_model]) == [("line_items_total_mismatch", "medium")]
    # The rules only read the layouts they know: one line they understood says nothing about a total made of ten.
    assert kinds([read_by_rules("inv", "invoice", text)]) == []


@pytest.mark.parametrize("row, quantity, price, amount, found", [
    ("Cartons 10 £60.00 20% £480.00", 10, 60, 480, False),  # 20% off
    ("Cartons 10 £60.00 20% £720.00", 10, 60, 720, False),  # 20% tax in the line
    ("Cartons 10 £60.00 20% £900.00", 10, 60, 900, True),  # neither
    ("Printed cartons  1,000  £1.20  £120.00  £1,080.00", 1000, 1.2, 1080, False),  # a discount column in money
    ("Printed cartons  1,000  £1.20  £120.00  £1,280.00", 1000, 1.2, 1280, True),
    ("Printed labels   Qty 5,000   Price per 1,000 £12.00   £60.00", 5000, 12, 60, False),
    ("Printed labels   Qty 5,000   Price per 1,000 £12.00   £600.00", 5000, 12, 600, True),
])
def test_what_a_row_says_is_worked_out_not_waved_through(row, quantity, price, amount, found):
    """Found in the third review: a "%" anywhere in a row, or "per thousand", switched the arithmetic check off."""
    built = read_by_model("inv", "invoice", f"INVOICE INV-1\n{row}", [], [item(row.split("  ")[0].split(" 10")[0], row, quantity, price, amount)])
    assert (kinds([built]) == [("line_arithmetic_mismatch", "medium")]) is found


def test_a_minimum_order_value_does_not_excuse_a_line_that_is_billed_too_high():
    over = "INVOICE\nInvoice No. INV-9\n40 cartons at £5.00 each: £2,600.00\nTotal due: £2,600.00\nMinimum order value £50."
    assert kinds([read_by_rules("inv", "invoice", over)]) == [("line_arithmetic_mismatch", "medium")]
    call_out = "INVOICE\nInvoice No. INV-9\n2 hours labour at £45.00 per hour: £120.00\nMinimum call-out charge applies.\nTotal due: £120.00"
    assert kinds([read_by_rules("inv", "invoice", call_out)]) == []


def test_a_line_is_not_counted_again_as_a_charge_when_its_quote_is_shortened():
    text = (
        "INVOICE INV-7\nFrom: Kestrel Packaging Ltd\nPrinted cartons | 1,000 | £1.20 | £1,200.00\n"
        "Handling and administration | 1 | £5,000.00 | £5,000.00\nTotal due: £11,200.00"
    )
    built = read_by_model("inv", "invoice", text, [fact("total_amount", "Total due: £11,200.00", number=11200, unit="GBP")], [
        item("Printed cartons", "Printed cartons | 1,000 | £1.20 | £1,200.00", 1000, 1.2, 1200),
        item("Handling and administration", "Handling and administration | 1 | ... | £5,000.00", 1, None, 5000),
    ])
    assert built.adjustments == [] and kinds([built]) == [("line_items_total_mismatch", "medium")]
    runs = read_by_rules("inv", "invoice", "INVOICE\nInvoice No. INV-8\nPrinted cartons: £1,200.00\n2 pallet delivery runs at £400.00 each\nTotal due: £2,400.00")
    assert runs.adjustments == []  # £400 is the price of a line, not a charge of its own


@pytest.mark.parametrize("body", [
    "Labour: £1,000.00\nMaterials: £500.00\nVAT 20%: £300.00\nCIS tax deducted (20% of labour): £200.00\nTotal due: £1,600.00",
    "Professional services: Rs. 100,000.00\nSales tax 16%: Rs. 16,000.00\nWithholding tax 10%: Rs. 10,000.00\nNet payable: Rs. 106,000.00",
    "Consulting fees: ₹1,00,000.00\nTravel: ₹20,000.00\nGST 18%: ₹21,600.00\nTDS 10%: ₹12,000.00\nTotal amount due: ₹1,29,600.00",
    "Works completed: £20,000.00\nVariation 3: £2,000.00\nRetention released: £1,100.00\nTotal due: £23,100.00",
    "Quarterly preventive maintenance: $9,500.00\n12 filters at $18.50 each: $222.00\nSubtotal: $9,722.00\nSales tax (8.25%): $802.07\nShipping: $35.00\nTotal Due: $10,559.07",
])
def test_taxes_that_are_withheld_and_retentions_that_are_released(body):
    built = read_by_rules("inv", "invoice", f"INVOICE\nInvoice No. INV-600\nInvoice date: 3 March 2026\nFrom: Kestrel Builders Ltd\n{body}")
    assert kinds([built]) == []


def test_a_rate_is_not_a_total_and_a_ceiling_is():
    assert [(hit.value, hit.priority) for hit in find_amounts("The total monthly charge is £3,000, exclusive of VAT.")] == [(3000.0, 0)]
    assert [hit.priority for hit in find_amounts("Total fee: £40,000 per annum.")] == [0]
    for ceiling in ("The total contract value shall not exceed £40,000.", "The total contract price is limited to a maximum of £40,000.", "Total contract value: up to £40,000."):
        assert [(hit.value, hit.priority > 0) for hit in find_amounts(ceiling)] == [(40000.0, True)]
    assert [hit.priority for hit in find_amounts("Invoices with a total over £5,000 need approval.")] == [0]

    monthly = contract(CONTRACT.replace("The total contract value is £40,000 for the Term, exclusive of VAT.", "The total monthly charge is £3,000, exclusive of VAT."))
    months = [invoice(f"m{index}", number=f"INV-4{index}", date=f"5 {month} 2026", amount="3,000.00", what=f"Lift maintenance, {month}") for index, month in enumerate(["March", "April", "May"])]
    assert monthly.total is None and kinds([monthly, *months], basic=True) == []
    capped = contract(CONTRACT.replace("The total contract value is £40,000 for the Term", "The total contract value shall not exceed £40,000"))
    assert kinds([capped, invoice(amount="58,000.00")], basic=True) == [("amount_mismatch", "high")]


# ================================================================== ordinary business
def test_an_annual_fee_invoiced_shortly_before_the_term_is_ordinary():
    assert kinds([contract(), invoice(date="19 January 2026")]) == []
    assert kinds([contract(), invoice(date="19 November 2025")]) == [("date_outside_term", "medium")]


def test_lines_worded_differently_with_the_same_numbers_are_the_same_lines():
    def table(name: str, kind: str, head: str, rows: list[str]):
        text = head + "\nDescription | Qty | Unit price | Amount\n" + "\n".join(rows)
        return read_by_rules(name, kind, text)

    order = table("po", "purchase_order", "PURCHASE ORDER\nPurchase Order No. PO-4421\nSupplier: Kestrel Packaging Ltd", [
        "Nitrile gloves, box of 100, size M | 100 | £5.00 | £500.00", "Hand sanitiser 500ml | 48 | £2.25 | £108.00", "Blue roll 2-ply, case of 6 | 20 | £11.40 | £228.00",
    ])
    billed = table("inv", "invoice", "TAX INVOICE\nInvoice No. K-7801\nFrom: Kestrel Packaging Ltd\nPurchase order: PO-4421", [
        "Nitrile gloves size M (box 100) | 100 | £5.00 | £500.00", "Sanitiser gel 500 ml pump | 48 | £2.25 | £108.00", "Blue centrefeed roll 2ply x6 | 20 | £11.40 | £228.00",
    ])
    assert len(order.line_items) == 3 and kinds([order, billed]) == []
    extra = table("inv", "invoice", "TAX INVOICE\nInvoice No. K-7801\nFrom: Kestrel Packaging Ltd\nPurchase order: PO-4421", [
        "Nitrile gloves size M (box 100) | 100 | £5.00 | £500.00", "Consultancy days | 4 | £650.00 | £2,600.00",
    ])
    assert kinds([order, extra]) == [("unlisted_line_item", "medium")]


def test_the_delivery_address_is_not_the_customer_when_the_billing_address_is():
    delivered = invoice(to="Deliver to: Harlow Works Site Office Ltd\nBill to: Harlow Print Works Ltd")
    assert [party.text for party in delivered.parties_with("customer")] == ["Harlow Works Site Office Ltd", "Harlow Print Works Ltd"]
    assert kinds([contract(), delivered]) == []


def test_reference_numbers_with_a_part_suffix_or_other_zeros():
    assert same_identifier("PO-4421/2", "PO-4421") and same_identifier("MC-2026-011/A", "MC-2026-011")
    assert same_identifier("MC/2026/11", "MC-2026-011") and same_identifier("PO-04421", "PO-4421")
    assert not same_identifier("PO-44212", "PO-4421") and not same_identifier("MC-2026-019", "MC-2026-011")


# ================================================================== totals from the search
def test_an_order_above_its_quotation_and_a_contract_above_its_approval_are_found():
    """Found in the third review: "less than the reference" had silenced both."""
    [order] = find_contradictions([stated("E1", "Quotation.pdf", "quotation", 10000), stated("E2", "Purchase_Order.pdf", "purchase_order", 12000)])
    assert (order.source_a, order.value_b) == ("Quotation.pdf", 12000)
    assert find_contradictions([stated("E1", "Quotation.pdf", "quotation", 10000), stated("E2", "Purchase_Order.pdf", "purchase_order", 8000)]) == []
    [memo] = find_contradictions([stated("E1", "Contract.pdf", "contract", 40000), stated("E2", "Board_Approval.pdf", "other", 35000)])
    assert (memo.value_a, memo.value_b) == (40000, 35000)


def test_a_receipt_is_not_measured_against_the_contract_behind_its_invoice():
    agreed, billed, paid = stated("E1", "Contract.pdf", "contract", 40000), stated("E2", "Invoice.pdf", "invoice", 48000), stated("E3", "Receipt.pdf", "receipt", 48000)
    assert find_contradictions([agreed, billed, paid], net_totals={"doc:Invoice.pdf": 40000.0}) == []


def test_the_same_number_far_apart_is_said_quietly_when_both_bill_the_same_work():
    first = invoice("april", number="INV-880", amount="3,150.00", date="2 April 2026", reference="", what="Lift maintenance, call-out 14 March")
    raised = invoice("july", number="INV-880", amount="3,950.00", date="6 July 2026", reference="", what="Lift maintenance, call-out 14 March")
    other = invoice("july", number="INV-880", amount="3,950.00", date="6 July 2026", reference="", what="Annual inspection of hoist")
    assert kinds([first, raised]) == [("duplicate_invoice", "low")]
    assert kinds([first, other]) == []  # other work under a number that came round again


# ========================================================================= grounding
def test_a_sentence_the_page_wraps_is_one_sentence():
    """Found in the third review: a "..." over a line break in the middle of a sentence was refused - and with
    it the contract's total, so an invoice above it was no longer found."""
    page = "4.1 The total contract value for the maintenance of the three passenger lifts\nat the Harlow site is £40,000 for the Term, exclusive of VAT."
    assert is_grounded("The total contract value ... is £40,000 for the Term", 40000, page)
    stitched = "INVOICE INV-1\nTotal amount due: see below\nCredit limit on this account: £58,000"
    assert not is_grounded("Total amount due: ... £58,000", 58000, stitched)


def test_a_currency_code_written_onto_the_figure_is_still_a_figure():
    assert number_stated(58000, "Total amount due: USD58,000.00") and number_stated(150000, "PKR150,000") and number_stated(5000, "Rs5,000")
    assert not number_stated(58, "INV58") and not number_stated(2, "v2")


@pytest.mark.parametrize("value, text", [
    (-311, "Invoice No. INV-311"), (-200, "Discount band £100-£200"), (-1200, "Total due - £1,200.00"), (-5000, "Paid: £5,000 credit card"),
])
def test_a_hyphen_a_range_and_a_separator_are_not_a_minus_sign(value, text):
    assert not number_stated(value, text)
