"""Step 3 of the spec: User -> LLM -> Response.

Checks the configured provider with the same kinds of request the agent
makes, so you know your API key and model work before a full investigation:

    1. a plain structured answer
    2. request analysis        (structured output with enums and lists)
    3. evidence extraction     (structured output with nullable fields, on a tiny document)
    4. tool calling            (the model must ask for the calculator)
    5. embeddings              (one passage and one question, with the configured provider)

    docker compose exec api python -m scripts.check_llm
    (or locally)  python -m scripts.check_llm

Uses 4 small model requests and 2 embedding requests. With LLM_PROVIDER=mock and
EMBEDDING_PROVIDER=hash nothing outside this computer is called.

Run it once with the settings you are going to deploy (see docs/DEPLOYMENT.md): a
hosting platform's free plan gives you no shell to try things out on the server.
"""

from __future__ import annotations

import asyncio
import sys

from app.agents import prompts
from app.config import get_settings
from app.embeddings.base import EmbeddingError
from app.embeddings.factory import get_embedding_provider
from app.llm.base import LLMError, LLMPing
from app.llm.factory import get_llm
from app.security.guardrails import redact_secrets
from app.schemas.investigation import EvidenceExtraction, EvidenceItem, RequestAnalysis, RetrievedChunk
from app.tools.calculator import CALCULATOR_TOOLS

QUESTION = "Compare the invoice total with the contract value."
SAMPLE_CHUNK = RetrievedChunk(
    chunk_id="check-1", document_id="check-doc", filename="Sample_Invoice.pdf", document_type="invoice",
    page=1, ref="C1", text="Invoice INV-7\nTotal amount due: £58,000\nPayment terms: 60 days from invoice date.",
)


def _evidence(id_: str, filename: str, doc_type: str, value: float) -> EvidenceItem:
    return EvidenceItem(
        id=id_, claim=f"{filename} states a total amount of {value:,.0f} GBP.", key="total_amount",
        value_number=value, unit="GBP", chunk_id=id_, document_id=filename, filename=filename,
        document_type=doc_type, page=1,
    )


async def main() -> int:
    settings = get_settings()
    print(f"Provider: {settings.LLM_PROVIDER}   Model: {settings.llm_model_name or '(not set)'}\n")
    try:
        llm = get_llm()

        ping = await llm.structured(
            LLMPing, system="You are a connectivity test. Answer briefly.",
            user="Reply with the single word: pong", purpose="check_ping",
        )
        print(f"1. structured answer    OK   reply={ping.reply!r}")

        analysis = await llm.structured(
            RequestAnalysis, system=prompts.system_prompt(prompts.ANALYZE_TASK),
            user=prompts.build_analysis_prompt(QUESTION, []), purpose="check_analysis",
        )
        print(f"2. request analysis     OK   task_type={analysis.task_type} queries={analysis.search_queries}")

        extraction = await llm.structured(
            EvidenceExtraction, system=prompts.system_prompt(prompts.EXTRACT_TASK),
            user=prompts.build_extraction_prompt(QUESTION, [SAMPLE_CHUNK]), purpose="check_extraction",
        )
        total = next((fact for fact in extraction.facts if fact.key == "total_amount"), None)
        if total is None or total.value_number != 58000:
            print(f"3. evidence extraction  WARN the model did not return the total as a typed fact: {extraction.facts}")
        else:
            print(f"3. evidence extraction  OK   total_amount={total.value_number:,.0f} {total.unit} from {total.chunk_ref}")

        evidence = [_evidence("E1", "Contract.pdf", "contract", 40000), _evidence("E2", "Invoice.pdf", "invoice", 58000)]
        calls = await llm.choose_tools(
            system=prompts.system_prompt(prompts.TOOLS_TASK),
            user=prompts.build_tools_prompt("What is the ratio of the invoice total to the contract total?", evidence, []),
            tools=CALCULATOR_TOOLS, purpose="check_tools",
        )
        if calls:
            print(f"4. tool calling         OK   requested={[(call.name, call.args) for call in calls]}")
        else:
            print("4. tool calling         WARN the model requested no tool (the agent still runs its standard calculations)")
    except LLMError as exc:
        print(f"FAILED: {exc}")
        return 1

    print(f"\nEmbeddings: {settings.EMBEDDING_PROVIDER}   Size: {settings.EMBEDDING_DIM}")
    try:
        embedder = get_embedding_provider()
        passages = [SAMPLE_CHUNK.text, "The cafeteria is closed on public holidays."]
        vectors = await embedder.aembed_documents(passages)
        query = await embedder.aembed_text("What is the total amount due on the invoice?")
        sizes = {len(vector) for vector in [*vectors, query]}
        if sizes != {settings.EMBEDDING_DIM}:
            print(f"FAILED: the vectors have {sorted(sizes)} numbers, EMBEDDING_DIM is {settings.EMBEDDING_DIM}.")
            return 1
        related, unrelated = (sum(a * b for a, b in zip(query, vector, strict=True)) for vector in vectors)
        verdict = "OK  " if related > unrelated else "WARN"
        print(f"5. embeddings           {verdict} model={embedder.model_id} similarity: invoice passage {related:.2f}, unrelated passage {unrelated:.2f}")
    except EmbeddingError as exc:
        print(f"FAILED: {redact_secrets(str(exc), settings)}")
        return 1

    print("\nThe model and the embeddings work. You can run an investigation now.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
