"""Run the demo investigation from spec section 57 against a running API.

    docker compose exec api python -m scripts.run_demo --email you@example.com --register
    docker compose exec api python -m scripts.run_demo --email you@example.com --with-injection
    python -m scripts.run_demo --base-url http://localhost:8000 --question "Compare the payment terms across these documents."

It uploads the sample contract, invoice and policy, asks the question and
prints the report. Uploading the same files again is harmless (duplicates are
detected), so the script can be run as often as you like.

Human approval
    The demo investigation finds a high-risk mismatch, so it pauses and waits
    for a person. The script then asks you to approve, reject or modify. To
    answer without typing:

    python -m scripts.run_demo --decision approve --note "Checked with procurement"
    python -m scripts.run_demo --decision modify --action "Pay 40,000 and dispute the rest"
    python -m scripts.run_demo --decision skip      (leave it waiting)

Progress is shown live through the streaming endpoint; --no-stream waits for
the finished report instead.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from pathlib import Path

import httpx

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
DEFAULT_QUESTION = "Review this invoice against the contract and tell me whether it should be approved."
SAMPLE_FILES = ["Contract.pdf", "Invoice_102.pdf", "Procurement_Policy.pdf"]


def fail(message: str) -> int:
    print(f"\nERROR: {message}")
    return 1


def stream_investigation(client: httpx.Client, question: str) -> tuple[dict | None, str]:
    """Run the investigation through the streaming endpoint and print progress as it arrives.

    Returns (report, "") or (None, what went wrong).
    """
    with client.stream("POST", "/investigations/stream", json={"question": question}) as response:
        if response.status_code != 200:
            response.read()
            return None, f"{response.status_code} {response.text}"
        event_name, report = "", None
        for line in response.iter_lines():
            if line.startswith("event: "):
                event_name = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
                if event_name == "progress":
                    print(f"  - {data['message']}")
                elif event_name == "tool":
                    print(f"      tool {data['tool_name']}: {data['status']}")
                elif event_name == "report":
                    report = data
                elif event_name == "error":
                    return None, data.get("message", "unknown error")
    return (report, "") if report else (None, "the stream ended without a report")


def sign_in(client: httpx.Client, args: argparse.Namespace) -> str:
    """Put the right credentials on the client. Returns "" or what went wrong."""
    config = client.get("/auth/config").json()
    if config["auth_mode"] == "dev":
        client.headers["X-User-Id"] = args.user  # development mode: a name is enough
        return ""
    email = args.email or os.environ.get("VERIAGENT_EMAIL", "")
    password = args.password or os.environ.get("VERIAGENT_PASSWORD", "")
    if not email or not password:
        if not sys.stdin.isatty():
            return "This server needs a sign-in. Pass --email and --password (add --register to create the account)."
        email = email or input("Email: ").strip()
        password = password or getpass.getpass("Password: ")
    credentials = {"email": email, "password": password}
    response = client.post("/auth/login", json=credentials)
    if response.status_code == 401 and args.register:
        response = client.post("/auth/register", json={**credentials, "signup_code": args.signup_code})
    if response.status_code not in (200, 201):
        hint = "" if args.register else " (add --register to create the account)"
        return f"Sign-in failed: {response.json().get('error', {}).get('message', response.text)}{hint}"
    client.headers["Authorization"] = f"Bearer {response.json()['access_token']}"
    print(f"Signed in as {response.json()['user']['email']}")
    return ""


def print_report(report: dict) -> None:
    print("=" * 78)
    print(report["answer_text"])
    print("=" * 78)


def ask_decision(args: argparse.Namespace) -> dict | None:
    """The decision to send, or None to leave the investigation waiting."""
    choice, note, action = args.decision, args.note, args.action
    if choice is None:
        if not sys.stdin.isatty():
            print("\nThis investigation is waiting for a human decision (no terminal to ask in).")
            return None
        print("\nThis investigation is paused and waits for your decision.")
        while choice not in ("approve", "reject", "modify", "skip"):
            choice = input("Decision [approve / reject / modify / skip]: ").strip().lower()
        if choice == "modify" and not action:
            action = input("Recommendation to use instead: ").strip()
        if choice != "skip" and note is None:
            note = input("Note (optional): ").strip() or None
    if choice == "skip":
        return None
    decision = {"decision": choice, "note": note}
    if choice == "modify":
        decision["modified_action"] = action
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the VeriAgent demo investigation.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--user", default="demo-user", help="AUTH_MODE=dev only: value for the X-User-Id header.")
    parser.add_argument("--email", help="Account to sign in with (or set VERIAGENT_EMAIL).")
    parser.add_argument("--password", help="Its password (or set VERIAGENT_PASSWORD; asked for when omitted).")
    parser.add_argument("--register", action="store_true", help="Create the account if it does not exist yet.")
    parser.add_argument("--signup-code", default=None, help="The sign-up code, if the server asks for one.")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--with-injection", action="store_true", help="Also upload a note containing a prompt-injection attempt.")
    parser.add_argument("--no-stream", action="store_true", help="Wait for the result instead of showing live progress.")
    parser.add_argument("--decision", choices=["approve", "reject", "modify", "skip"], help="Answer the approval request without asking.")
    parser.add_argument("--note", default=None, help="Reason for the decision (shown in the final report).")
    parser.add_argument("--action", default=None, help="For --decision modify: the recommendation to use instead.")
    args = parser.parse_args()

    if not (SAMPLES_DIR / "Contract.pdf").exists():
        from scripts.make_sample_docs import main as make_samples

        make_samples()

    files = [SAMPLES_DIR / name for name in SAMPLE_FILES]
    if args.with_injection:
        files.append(SAMPLES_DIR / "adversarial" / "Vendor_Note.txt")

    # An investigation makes several LLM calls, so allow it plenty of time.
    with httpx.Client(base_url=args.base_url, timeout=600) as client:
        try:
            health = client.get("/health").json()
        except httpx.HTTPError as exc:
            return fail(f"Cannot reach the API at {args.base_url} ({exc}). Is 'docker compose up' running?")
        problem = sign_in(client, args)
        if problem:
            return fail(problem)
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
        if args.no_stream:
            response = client.post("/investigations", json={"question": args.question})
            if response.status_code != 201:
                return fail(f"Investigation failed: {response.status_code} {response.text}")
            report = response.json()
        else:
            report, problem = stream_investigation(client, args.question)
            if report is None:
                return fail(f"Investigation failed: {problem}")
            print()

        if report["human_review"]["status"] == "pending":
            print_report(report)
            decision = ask_decision(args)
            if decision is None:
                print(
                    f"\nLeft waiting. Decide later with:\n"
                    f"  POST {args.base_url}/investigations/{report['investigation_id']}/approve"
                )
                return 0
            response = client.post(f"/investigations/{report['investigation_id']}/approve", json=decision)
            if response.status_code != 200:
                return fail(f"The decision was not accepted: {response.status_code} {response.text}")
            report = response.json()
            print("\nFinal report after the decision:\n")

    print_report(report)
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
