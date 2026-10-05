"""VeriAgent - Streamlit interface (spec section 50).

Run it:
    docker compose up --build          then open http://localhost:8501
or, without Docker, next to a running API:
    streamlit run ui/app.py

Layout
    Left panel    your documents, upload, investigation history, memory
    Main panel    the chat, with live progress while the agent works
    Result panel  risk level, confidence, findings as expandable evidence
                  cards, contradictions, missing evidence, calculations,
                  recommendation, sources, and the human decision

Concept - how Streamlit works:
    The whole script runs again from top to bottom every time you click or
    type something. Anything that must survive those reruns (the chat so
    far, the report on screen) is kept in ``st.session_state``.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

import streamlit as st

# "streamlit run ui/app.py" puts ui/ on the path, not the project folder.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ui.api_client import ApiClient, ApiError  # noqa: E402

API_URL = os.environ.get("VERIAGENT_API_URL", "http://localhost:8000")
RISK_COLORS = {"low": "green", "medium": "orange", "high": "red", "critical": "red"}
FINDING_LABELS = {"fact": "Fact", "calculation": "Calculation", "missing_evidence": "Missing evidence", "risk": "Risk"}
DECISION_LABELS = {
    "approve": "Approve - accept the agent's recommendation",
    "reject": "Reject - do not accept it",
    "modify": "Modify - replace the recommendation with my own",
}
PANEL_HEIGHT = 640  # pixels; the chat and the result panel each scroll inside this height
USER_PATTERN = re.compile(r"^[A-Za-z0-9_.@\-]{1,64}$")

st.set_page_config(page_title="VeriAgent", page_icon=":mag:", layout="wide")


# ------------------------------------------------------------------- state
def init_state() -> None:
    defaults: dict[str, Any] = {
        "user": "demo-user",
        "messages": [],  # the chat on screen: {"role", "content"}
        "conversation_id": None,  # lets the API resolve follow-up questions
        "report": None,  # the report shown in the result panel
        "notice": None,  # a one-off message shown after a rerun: (kind, text)
        "uploader_key": 0,  # changing it empties the file uploader
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def new_conversation() -> None:
    st.session_state.messages = []
    st.session_state.conversation_id = None
    st.session_state.report = None


def notify(kind: str, text: str) -> None:
    st.session_state.notice = (kind, text)


def show_notice() -> None:
    notice = st.session_state.notice
    if notice:
        kind, text = notice
        {"success": st.success, "error": st.error, "info": st.info}[kind](text)
        st.session_state.notice = None


def cite(citation: dict[str, Any]) -> str:
    return f"{citation['document']} — Page {citation['page']}"


def amount(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:,.2f}".rstrip("0").rstrip(".")
    return str(value)


# ----------------------------------------------------------------- sidebar
def sidebar(api: ApiClient) -> None:
    with st.sidebar:
        st.title("VeriAgent")
        st.caption("Evidence verification for business documents")

        try:
            health = api.health()
            st.caption(f"API: {health['status']} · model: {health['llm_provider']} ({health['llm_model']})")
            if health["llm_provider"] == "mock":
                st.info("Running with the offline mock model. Set LLM_PROVIDER=gemini and GEMINI_API_KEY in .env for real reasoning.")
        except ApiError as exc:
            st.error(str(exc))
            st.stop()

        st.subheader("Documents")
        try:
            documents = api.documents()
        except ApiError as exc:
            st.error(str(exc))
            documents = []
        if not documents:
            st.caption("Nothing uploaded yet.")
        for document in documents:
            name, delete = st.columns([6, 1], vertical_alignment="center")
            status = "" if document["status"] == "ready" else f" · {document['status']}"
            name.markdown(
                f"**{document['filename']}**  \n"
                f":gray[{document['document_type']} · {document['page_count']} page(s){status}]"
            )
            if delete.button(":material/delete:", key=f"delete-document-{document['id']}", help="Delete this document", type="tertiary"):
                run(lambda document=document: api.delete_document(document["id"]), f"Deleted {document['filename']}.")

        files = st.file_uploader(
            "New upload", type=["pdf", "docx", "txt"], accept_multiple_files=True,
            key=f"uploader-{st.session_state.uploader_key}",
        )
        if files and st.button("Upload", type="primary", width="stretch"):
            upload_files(api, files)

        st.subheader("Investigation history")
        try:
            history = api.investigations()[:15]
        except ApiError as exc:
            st.error(str(exc))
            history = []
        if not history:
            st.caption("No investigations yet.")
        for item in history:
            risk = (item.get("risk_level") or item["status"]).capitalize()
            waiting = " · waiting for a decision" if item["approval_status"] == "pending" else ""
            if st.button(
                f"{risk}{waiting}: {item['question'][:60]}", key=f"history-{item['id']}", width="stretch"
            ):
                open_investigation(api, item["id"])

        memory_panel(api)

        st.divider()
        user = st.text_input("User name", value=st.session_state.user, help="Placeholder sign-in: each name has its own documents and memory.")
        if user != st.session_state.user:
            if USER_PATTERN.match(user):
                st.session_state.user = user
                new_conversation()
                st.rerun()
            else:
                st.error("Letters, digits and . _ @ - only (max 64 characters).")
        if st.button("New conversation", width="stretch"):
            new_conversation()
            st.rerun()


def run(action: Any, success: str) -> None:
    """Do something through the API, then rerun so the page shows the new state."""
    try:
        action()
        notify("success", success)
    except ApiError as exc:
        notify("error", str(exc))
    st.rerun()


def upload_files(api: ApiClient, files: list[Any]) -> None:
    done, problems = [], []
    with st.spinner("Reading, chunking and embedding..."):
        for file in files:
            try:
                document = api.upload(file.name, file.getvalue())
                done.append(f"{document['filename']} (already uploaded)" if document["duplicate"] else document["filename"])
            except ApiError as exc:
                problems.append(f"{file.name}: {exc}")
    st.session_state.uploader_key += 1
    if problems:
        notify("error", " | ".join(problems))
    elif done:
        notify("success", "Uploaded: " + ", ".join(done))
    st.rerun()


def open_investigation(api: ApiClient, investigation_id: str) -> None:
    try:
        detail = api.investigation(investigation_id)
        if detail.get("report"):
            st.session_state.report = detail["report"]
        else:
            notify("info", f"That investigation has no report (status: {detail['status']}).")
    except ApiError as exc:
        notify("error", str(exc))
    st.rerun()


def memory_panel(api: ApiClient) -> None:
    with st.expander("Memory"):
        try:
            memory = api.memory()
        except ApiError as exc:
            st.error(str(exc))
            return
        if not memory["enabled"]:
            st.caption("Long-term memory is switched off (ENABLE_MEMORY=false).")
            return
        st.caption("Preferences shape how answers are written.")
        for preference in memory["preferences"]:
            options = preference["allowed_values"]
            choice = st.selectbox(
                preference["key"].replace("_", " ").capitalize(), options,
                index=options.index(preference["value"]), help=preference["description"],
                # The stored value is part of the key: when it changes on the server (for example
                # because you told the chat "keep answers brief"), the box shows the new value.
                key=f"preference-{preference['key']}-{preference['value']}-{st.session_state.user}",
            )
            if choice != preference["value"]:
                run(lambda key=preference["key"], value=choice: api.set_preference(key, value), "Preference saved.")
        st.caption("Notes from earlier sessions. They are shown as context, never used as evidence.")
        if not memory["notes"]:
            st.caption("No notes yet.")
        for note in memory["notes"]:
            text, delete = st.columns([6, 1], vertical_alignment="center")
            text.markdown(f":gray[{note['created_at'][:10]} · {note['source']}]  \n{note['text']}")
            if delete.button(":material/delete:", key=f"delete-note-{note['id']}", help="Forget this note", type="tertiary"):
                run(lambda note=note: api.delete_note(note["id"]), "Note deleted.")
        new_note = st.text_input("Remember a fact", placeholder="Vendor ABC was flagged for duplicate invoices in 2025.", key="new-note")
        if st.button("Save note", disabled=not new_note.strip()):
            run(lambda: api.add_note(new_note), "Note saved.")


# -------------------------------------------------------------------- chat
def chat_panel(api: ApiClient, prompt: str | None) -> None:
    if not st.session_state.messages and not prompt:
        st.caption(
            "Upload documents on the left, then ask, for example: "
            "*Review this invoice against the contract and tell me whether it should be approved.*"
        )
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if prompt:
        ask(api, prompt)


def ask(api: ApiClient, prompt: str) -> None:
    with st.chat_message("user"):
        st.markdown(prompt)
    answer, failure = None, None
    with st.chat_message("assistant"):
        # The status box shows the safe progress events as they arrive (streaming).
        with st.status("Investigating...", expanded=True) as status:
            try:
                for event, data in api.chat_stream(prompt, st.session_state.conversation_id):
                    if event == "started":
                        st.session_state.conversation_id = data["conversation_id"]
                    elif event == "progress":
                        status.update(label=data["message"])
                        st.write(data["message"])
                    elif event == "tool":
                        st.write(f"Tool `{data['tool_name']}`: {data['status']}")
                    elif event == "report":
                        st.session_state.report = data
                    elif event == "answer":
                        answer = data
                    elif event == "error":
                        failure = data.get("message", "The investigation failed.")
            except ApiError as exc:
                failure = str(exc)
            if failure:
                status.update(label="Failed", state="error", expanded=True)
            else:
                status.update(label="Done", state="complete", expanded=False)

    if failure or answer is None:
        notify("error", failure or "The stream ended without an answer.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": answer["answer"]})
    st.rerun()


# ------------------------------------------------------------- result panel
def result_panel(api: ApiClient) -> None:
    report = st.session_state.report
    if not report:
        st.caption("The report of the latest investigation appears here.")
        return
    if report["status"] == "not_applicable":
        st.info(report["summary"])
        return

    st.caption(report["question"])
    risk, confidence, status = st.columns(3)
    risk.markdown(f"**Risk level**  \n:{RISK_COLORS.get(report['risk_level'], 'gray')}[**{report['risk_level'].upper()}**]")
    confidence.markdown(f"**Confidence**  \n**{round(report['confidence'] * 100)}%**")
    status.markdown(f"**Status**  \n{report['status_label']}")
    st.markdown(report["summary"])

    approval_block(api, report)

    st.markdown("**Recommendation**")
    st.info(report["recommended_action"] or "No recommendation.")
    original = report["human_review"].get("original_action")
    if original:
        st.caption(f"The agent had recommended: {original}")

    if report["contradictions"]:
        st.markdown(f"**Contradictions ({len(report['contradictions'])})**")
        for item in report["contradictions"]:
            with st.expander(f"{item['severity'].upper()} · {item['description']}"):
                st.markdown(f"**Finding:** {item['description']}")
                st.markdown(f"**Risk:** {item['severity'].capitalize()}")
                st.markdown(f"**Values:** {item['source_a']}: {amount(item['value_a'])} · {item['source_b']}: {amount(item['value_b'])}")
                if item.get("difference") is not None:
                    change = f" ({amount(item['percentage_change'])}%)" if item.get("percentage_change") is not None else ""
                    st.markdown(f"**Difference (calculated by a tool):** {amount(item['difference'])}{change}")
                evidence(item["citations"])

    if report["missing_evidence"]:
        st.markdown(f"**Missing evidence ({len(report['missing_evidence'])})**")
        for text in report["missing_evidence"]:
            st.warning(text)

    findings = [finding for finding in report["key_findings"] if finding["type"] != "missing_evidence"]
    if findings:
        st.markdown(f"**Key findings ({len(findings)})**")
        for finding in findings:
            label = FINDING_LABELS.get(finding["type"], finding["type"])
            with st.expander(f"{label} · {finding['statement'][:110]}"):
                st.markdown(finding["statement"])
                evidence(finding["citations"])

    if report["risk_reasons"]:
        with st.expander(f"Risk reasoning ({len(report['risk_reasons'])}) - the agent's inference"):
            for reason in report["risk_reasons"]:
                st.markdown(f"- {reason['statement']}")

    if report["calculations"]:
        with st.expander(f"Calculations ({len(report['calculations'])}) - done by deterministic tools"):
            st.dataframe(
                [
                    {
                        "What": item["description"], "Tool": item["tool_name"],
                        "Inputs": ", ".join(f"{name}={amount(value)}" for name, value in item["arguments"].items()),
                        "Result": amount(item["result"]),
                    }
                    for item in report["calculations"]
                ],
                hide_index=True, width="stretch",
            )

    if report["sources"]:
        with st.expander(f"Sources ({len(report['sources'])})"):
            for source in report["sources"]:
                st.markdown(f"- {cite(source)}")

    recalled = report["memory"]["recalled"]
    if recalled:
        with st.expander(f"From earlier investigations ({len(recalled)}) - memory, not evidence"):
            for note in recalled:
                st.markdown(f"- {note['text']}")
    for text in report["memory"]["saved"]:
        st.caption(text)

    if report["limitations"]:
        with st.expander(f"Limitations ({len(report['limitations'])})"):
            for text in report["limitations"]:
                st.markdown(f"- {text}")

    with st.expander("Full report as text"):
        if report["presentation"]["response_format"] == "markdown":
            st.markdown(report["answer_text"])
        else:
            st.text(report["answer_text"])

    with st.expander("What the agent did (trace)"):
        stats = report["stats"]
        st.caption(
            f"{stats['llm_calls']} LLM call(s) · {stats['tool_calls']} tool call(s) · {stats['retrieval_rounds']} search round(s) · "
            f"{stats['duration_ms'] / 1000:.1f}s · tokens in/out: {stats['input_tokens']}/{stats['output_tokens']}"
        )
        if st.button("Load trace", key=f"trace-{report['investigation_id']}"):
            try:
                trace = api.trace(report["investigation_id"])
                st.dataframe(
                    [
                        {
                            "Run": event["run"], "#": event["seq"], "Kind": event["kind"], "Name": event["name"],
                            "ms": event["latency_ms"], "Details": str(event["data"])[:200],
                        }
                        for event in trace["events"]
                    ],
                    hide_index=True, width="stretch",
                )
            except ApiError as exc:
                st.error(str(exc))


def evidence(citations: list[dict[str, Any]]) -> None:
    """The evidence card body: where the statement comes from, with the exact words."""
    if not citations:
        st.caption("No document evidence (this is a derived statement).")
        return
    st.markdown("**Evidence:**")
    for citation in citations:
        st.markdown(f"- {cite(citation)}")
        if citation.get("quote"):
            st.caption(f"“{citation['quote']}”")


def approval_block(api: ApiClient, report: dict[str, Any]) -> None:
    """Human-in-the-loop: a paused investigation waits here for a person."""
    review = report["human_review"]
    if review["status"] == "pending":
        st.warning("This investigation is paused. A person has to decide before the report is final.")
        with st.form(f"decision-{report['investigation_id']}"):
            decision = st.radio("Your decision", list(DECISION_LABELS), format_func=DECISION_LABELS.get)
            modified_action = st.text_input("Recommendation to use instead (only for Modify)")
            note = st.text_area("Reason (shown in the final report)", height=80)
            if st.form_submit_button("Submit decision", type="primary"):
                if decision == "modify" and not modified_action.strip():
                    st.error("Modify needs the recommendation to use instead.")
                    return
                try:
                    final = api.decide(report["investigation_id"], decision, note.strip(), modified_action.strip())
                except ApiError as exc:
                    st.error(str(exc))
                    return
                st.session_state.report = final
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"Decision recorded: **{final['human_review']['status']}**. The report is final."}
                )
                st.rerun()
    elif review["status"] in ("approved", "rejected", "modified"):
        text = f"Human review: **{review['status']}** by {review.get('reviewer') or 'reviewer'}"
        if review.get("note"):
            text += f" — {review['note']}"
        st.success(text)
    elif review["status"] == "not_requested":
        st.warning("A person should review this result before acting on it.")


# -------------------------------------------------------------------- page
@st.cache_resource
def get_api(base_url: str, user: str) -> ApiClient:
    """One HTTP client per user, reused across reruns instead of opening new connections each time."""
    return ApiClient(base_url, user)


def main() -> None:
    init_state()
    api = get_api(API_URL, st.session_state.user)
    sidebar(api)
    show_notice()
    prompt = st.chat_input("Ask a question about your documents")
    chat_column, result_column = st.columns([5, 4], gap="large")
    # Each panel scrolls on its own, so a long report never pushes the chat off the screen.
    with result_column:
        st.subheader("Result")
        with st.container(height=PANEL_HEIGHT, border=False):
            result_panel(api)
    with chat_column:
        st.subheader("Chat")
        with st.container(height=PANEL_HEIGHT, border=False):
            chat_panel(api, prompt)  # last: a new question reruns the page when it finishes


main()
