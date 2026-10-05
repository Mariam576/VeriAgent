"""Run the demo investigation from spec section 57 against a running API.

    docker compose exec api python -m scripts.run_demo
    docker compose exec api python -m scripts.run_demo --with-injection
    python -m scripts.run_demo --base-url http://localhost:8000 --question "Compare the payment terms across these documents."

It uploads the sample contract, invoice and policy, asks the question and
prints the report. Uploading the same files again is harmless (duplicates are
detected), so the script can be run as often as you like.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
DEFAULT_QUESTION = "Review this invoice against the contract and tell me whether it should be approved."
SAMPLE_FILES = ["Contract.pdf", "Invoice_102.pdf", "Procurement_Policy.pdf"]


def fail(message: str) -> int:
    print(f"\nERROR: {message}")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the VeriAgent demo investigation.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--user", default="demo-user", help="Value for the X-User-Id header.")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--with-injection", action="store_true", help="Also upload a note containing a prompt-injection attempt.")
    args = parser.parse_args()

    if not (SAMPLES_DIR / "Contract.pdf").exists():
        from scripts.make_sample_docs import main as make_samples

        make_samples()

    headers = {"X-User-Id": args.user}
    files = [SAMPLES_DIR / name for name in SAMPLE_FILES]
    if args.with_injection:
        files.append(SAMPLES_DIR / "adversarial" / "Vendor_Note.txt")

    # An investigation makes several LLM calls, so allow it plenty of time.
    with httpx.Client(base_url=args.base_url, headers=headers, timeout=600) as client:
        try:
            health = client.get("/health").json()
        except httpx.HTTPError as exc:
            return fail(f"Cannot reach the API at {args.base_url} ({exc}). Is 'docker compose up' running?")
        print(f"API: {health['status']}  |  LLM: {health['llm_provider']} ({health['llm_model']})  |  Embeddings: {health['embedding_provider']}")
        if health["llm_provider"] == "mock":
            print("Note: LLM_PROVIDER=mock is a rule-based stand-in. Set LLM_PROVIDER=gemini and GEMINI_API_KEY for real reasoning.")

        print("\nUploading documents (the first upload downloads the embedding model, which can take a minute)...")
        for path in files:
            with path.open("rb") as handle:
                response = client.post("/documents/upload", files={"file": (path.name, handle)})
            if response.status_code != 201:
                return fail(f"Upload of {path.name} failed: {response.status_code} {response.text}")
            document = response.json()
            note = "already uploaded" if document["duplicate"] else f"{document['page_count']} page(s), {document['chunk_count']} chunk(s)"
            print(f"  {document['filename']:<26} type={document['document_type']:<10} {note}")

        print(f"\nQuestion: {args.question}\nInvestigating...\n")
        response = client.post("/investigations", json={"question": args.question})
        if response.status_code != 201:
            return fail(f"Investigation failed: {response.status_code} {response.text}")
        report = response.json()

    print("=" * 78)
    print(report["answer_text"])
    print("=" * 78)
    stats = report["stats"]
    print(
        f"Investigation {report['investigation_id']}\n"
        f"LLM calls: {stats['llm_calls']}  |  tool calls: {stats['tool_calls']}  |  repeat searches: {stats['iterations']}  |  "
        f"chunks read: {stats['retrieved_chunks']}  |  facts kept/discarded: {stats['evidence_items']}/{stats['discarded_facts']}  |  "
        f"{stats['duration_ms'] / 1000:.1f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
