"""Reranking, citations and grounding - no database needed."""

from __future__ import annotations

from app.rag.citations import (
    Citation,
    citations_from_evidence,
    format_citation,
    is_grounded,
    numbers_in_text,
    unsupported_citations,
)
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
