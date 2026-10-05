"""Reranking, citations and grounding - no database needed."""

from __future__ import annotations

from app.rag.citations import (
    Citation,
    citations_from_evidence,
    format_citation,
    is_grounded,
    quote_in_text,
    unsupported_citations,
)
from app.numbers import normalize_unit, number_appears, numbers_in_text, parse_number
from app.rag.reranker import lexical_overlap, rerank
from app.schemas.investigation import EvidenceItem, RetrievedChunk


def chunk(chunk_id: str, document_id: str, text: str, score: float, page: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, document_id=document_id, filename=f"{document_id}.pdf", page=page, text=text, score=score
    )


def item(id_: str, chunk_: RetrievedChunk) -> EvidenceItem:
    return EvidenceItem(
        id=id_, claim="c", chunk_id=chunk_.chunk_id, document_id=chunk_.document_id,
        filename=chunk_.filename, page=chunk_.page,
    )


# ---------------------------------------------------------------- reranker
def test_lexical_overlap_counts_query_words_in_text():
    assert lexical_overlap("payment terms", "Payment terms: 30 days") == 1.0
    assert lexical_overlap("payment terms", "The kitchen is cleaned weekly") == 0.0
    assert lexical_overlap("the of and", "anything") == 0.0  # only stop words


def test_exact_identifiers_weigh_more_than_ordinary_words():
    with_id = lexical_overlap("invoice INV-102", "Reference INV-102 enclosed")
    with_word = lexical_overlap("invoice INV-102", "This invoice is enclosed")
    assert with_id > with_word


def test_rerank_keeps_top_n_best_first():
    candidates = [
        chunk("a", "doc1", "Office kitchen rota", 0.50),
        chunk("b", "doc1", "Payment terms: 30 days from invoice", 0.48),
        chunk("c", "doc1", "Payment is due monthly", 0.40),
    ]
    result = rerank("payment terms", candidates, top_n=2)
    assert [c.chunk_id for c in result] == ["b", "c"]
    assert result[0].rerank_score >= result[1].rerank_score


def test_rerank_keeps_at_least_one_chunk_per_document():
    candidates = [
        chunk("c1", "contract", "total contract value 40,000", 0.90),
        chunk("c2", "contract", "contract value and charges", 0.88),
        chunk("c3", "contract", "value of the contract", 0.86),
        chunk("i1", "invoice", "amount due", 0.30),
    ]
    result = rerank("total contract value", candidates, top_n=2)
    assert {c.document_id for c in result} == {"contract", "invoice"}


def test_rerank_edge_cases():
    assert rerank("q", [], top_n=4) == []
    assert rerank("q", [chunk("a", "d", "t", 0.5)], top_n=0) == []
    assert len(rerank("q", [chunk("a", "d", "t", 0.5)], top_n=4)) == 1


# --------------------------------------------------------------- grounding
def test_numbers_in_text():
    assert numbers_in_text("The total contract value is £40,000 payable within 30 days.") == [40000.0, 30.0]
    assert numbers_in_text("Total: 1,234,567.89 for INV-102") == [1234567.89, 102.0]
    assert numbers_in_text("no digits") == []


def test_a_number_must_be_written_in_the_chunk():
    text = "7.1 The total contract value is £40,000 (forty thousand pounds sterling)."
    assert is_grounded("total contract value is £40,000", 40000, text) is True
    assert is_grounded("total contract value is £58,000", 58000, text) is False


def test_a_quote_must_appear_in_the_chunk():
    text = "The Client shall pay each valid\ninvoice within 30 days of the invoice date."
    assert is_grounded("pay each valid invoice within 30 days", None, text) is True  # line break ignored
    assert is_grounded("PAY EACH VALID INVOICE", None, text) is True  # case ignored
    assert is_grounded("the invoice was approved by the Finance Director", None, text) is False
    assert is_grounded("", None, text) is False
    assert is_grounded("pay", None, text) is False  # too short to prove anything


# --------------------------------------------------------------- citations
def test_citations_are_unique_per_document_page():
    contract_p4 = chunk("k1", "contract", "t", 0.9, page=4)
    contract_p4_b = chunk("k2", "contract", "t", 0.9, page=4)
    invoice_p1 = chunk("k3", "invoice", "t", 0.9, page=1)
    citations = citations_from_evidence(
        [item("E1", contract_p4), item("E2", invoice_p1), item("E3", contract_p4_b), item("E4", contract_p4)]
    )
    assert [(c.document, c.page) for c in citations] == [("contract.pdf", 4), ("invoice.pdf", 1)]
    assert citations[0].chunk_ids == ["k1", "k2"]


def test_format_citation():
    assert format_citation("Contract.pdf", 4) == "[Contract.pdf, page 4]"


def test_citation_without_a_retrieved_chunk_is_reported():
    retrieved = [chunk("k1", "contract", "t", 0.9, page=4)]
    good = Citation(document="contract.pdf", page=4, document_id="contract", chunk_ids=["k1"])
    wrong_page = Citation(document="contract.pdf", page=9, document_id="contract", chunk_ids=["k1"])
    invented = Citation(document="approval.pdf", page=1, document_id="nope", chunk_ids=["zzz"])
    no_chunk = Citation(document="contract.pdf", page=4, document_id="contract", chunk_ids=[])
    assert unsupported_citations([good], retrieved) == []
    assert unsupported_citations([good, wrong_page, invented, no_chunk], retrieved) == [wrong_page, invented, no_chunk]


# ------------------------------------------- grounding against real-world text
def test_a_number_elsewhere_in_the_chunk_does_not_support_an_invented_claim():
    text = "7.3 Payment terms: the Client shall pay each valid invoice within 30 days of the invoice date."
    # "30" really is in the chunk, but not in this (invented) quote.
    assert is_grounded("approved by 30 directors", 30, text) is False
    # A real quote that does not contain the number is not enough either.
    assert is_grounded("the Client shall pay each valid invoice", 30, text) is False
    assert is_grounded("pay each valid invoice within 30 days", 30, text) is True


def test_numeric_quotes_may_be_short_and_in_any_number_format():
    assert is_grounded("£40,000", 40000, "The total contract value is £40,000.") is True
    assert is_grounded("Rs. 1,50,000", 150000, "Total amount due: Rs. 1,50,000 only") is True
    assert is_grounded("EUR 40.000,00", 40000, "Gesamtbetrag: EUR 40.000,00") is True


def test_quotes_survive_pdf_line_breaks_hyphenation_dashes_and_ellipses():
    text = "The Supplier shall invoice the Client quarterly in arrears. Pay-\nment is due within 30 days – no exceptions.\nReference INV-\n102 applies."
    assert quote_in_text("Payment is due within 30 days", text)  # word broken across lines
    assert quote_in_text("due within 30 days - no exceptions", text)  # en dash vs hyphen
    assert quote_in_text("Reference INV-102 applies", text)  # code broken across lines
    assert quote_in_text("The Supplier shall invoice ... Payment is due within 30 days", text)  # shortened with "..."
    assert quote_in_text("The Supplier shall invoice the Client… no exceptions", text)
    assert not quote_in_text("Payment is due within 30 days ... The Supplier shall invoice", text)  # wrong order
    assert not quote_in_text("The Supplier shall invoice ... a signed approval exists", text)  # one part invented
    assert not quote_in_text("...", text)


def test_number_helpers_are_shared_with_the_rest_of_the_app():
    assert numbers_in_text("£40,000 within 30 days") == [40000.0, 30.0]
    assert parse_number("1,50,000") == [150000.0]
    assert number_appears(150000, "Rs. 1,50,000")
    assert normalize_unit("pounds sterling") == "GBP"


# ------------------------------------------------------------ hybrid search
def test_keyword_terms_separate_words_from_exact_codes():
    from app.rag.hybrid import keyword_terms, to_websearch_query

    words, identifiers = keyword_terms("What does Clause 7.3 say about invoice INV-102 and PO-4421 in 2026?")
    assert identifiers == ["7.3", "inv-102", "po-4421"]
    assert words == ["clause", "say", "invoice", "2026"]
    assert to_websearch_query(identifiers) == '"7.3" or "inv-102" or "po-4421"'
    assert keyword_terms("the of and is") == ([], [])


def test_keyword_terms_cannot_carry_query_syntax():
    from app.rag.hybrid import keyword_terms, to_websearch_query

    words, identifiers = keyword_terms("payment'); DROP TABLE users; -- \"terms\" | & ! <-> :*")
    assert words == ["payment", "drop", "table", "users", "terms"] and identifiers == []
    assert to_websearch_query(words) == '"payment" or "drop" or "table" or "users" or "terms"'


def test_rank_fusion_rewards_chunks_found_by_both_searches():
    from app.rag.hybrid import reciprocal_rank_fusion

    a, b, c, d = (chunk(name, "doc", "t", 0.5) for name in "abcd")
    fused = reciprocal_rank_fusion(vector_hits=[a, b, c], keyword_hits=[c, d], limit=10)
    assert [item.chunk_id for item in fused][0] == "c"  # rank 3 + rank 1 beats rank 1 alone
    by_id = {item.chunk_id: item for item in fused}
    assert by_id["c"].matched_by == ["vector", "keyword"]
    assert by_id["a"].matched_by == ["vector"] and by_id["d"].matched_by == ["keyword"]
    assert by_id["c"].fused_score > by_id["a"].fused_score > by_id["b"].fused_score
    assert len(reciprocal_rank_fusion([a, b, c], [c, d], limit=2)) == 2
    assert reciprocal_rank_fusion([], [], limit=5) == []
