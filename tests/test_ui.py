"""The Streamlit interface, driven without a browser against a real API process.

Streamlit's AppTest runs ui/app.py in-process and lets a test click, type and
read what is on the page. The API runs as a separate process (mock LLM, hash
embeddings, the test database), so these tests cover the whole path:
interface -> HTTP -> FastAPI -> LangGraph -> PostgreSQL.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parent.parent
DEMO_QUESTION = "Review this invoice against the contract and tell me whether it should be approved."


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def api_url(database):
    """Start the API as its own process and wait until it answers."""
    port = free_port()
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(port), "--log-level", "warning"],
        cwd=ROOT, env={**os.environ}, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            if process.poll() is not None:
                raise RuntimeError(f"The API process stopped: {process.stderr.read().decode()[-2000:]}")
            try:
                if httpx.get(f"{url}/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.2)
        else:
            raise RuntimeError("The API did not start in time.")
        yield url
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


@pytest.fixture
def user() -> str:
    from tests.conftest import RUN_ID

    return f"test-{RUN_ID}-{uuid.uuid4().hex[:8]}"


def open_app(api_url: str, user: str, monkeypatch) -> AppTest:
    monkeypatch.setenv("VERIAGENT_API_URL", api_url)
    app = AppTest.from_file(str(ROOT / "ui" / "app.py"), default_timeout=60)
    app.session_state["user"] = user
    app.run()
    assert not app.exception, app.exception
    return app


def upload_samples(api_url: str, user: str, samples_dir: Path) -> None:
    with httpx.Client(base_url=api_url, headers={"X-User-Id": user}, timeout=60) as client:
        for name in ("Contract.pdf", "Invoice_102.pdf", "Procurement_Policy.pdf"):
            with (samples_dir / name).open("rb") as handle:
                assert client.post("/documents/upload", files={"file": (name, handle)}).status_code == 201


def page_text(app: AppTest) -> str:
    """Everything a person could read on the page (main area and sidebar)."""
    parts = []
    for kind in ("markdown", "caption", "info", "warning", "success", "error", "text", "subheader", "title"):
        parts += [str(element.value) for element in getattr(app, kind)]
        parts += [str(element.value) for element in getattr(app.sidebar, kind)]
    parts += [str(expander.label) for expander in app.expander]
    return "\n".join(parts)


def test_empty_page_explains_what_to_do(api_url, user, monkeypatch):
    app = open_app(api_url, user, monkeypatch)
    text = page_text(app)
    assert "VeriAgent" in text and "Nothing uploaded yet." in text and "No investigations yet." in text
    assert "The report of the latest investigation appears here." in text
    assert "offline mock model" in text  # honest about which model is answering


def test_investigation_approval_and_history_through_the_interface(api_url, user, samples_dir, monkeypatch):
    upload_samples(api_url, user, samples_dir)
    app = open_app(api_url, user, monkeypatch)
    assert "Contract.pdf" in page_text(app) and "Procurement_Policy.pdf" in page_text(app)

    # ---- ask in the chat: progress is streamed, then the report fills the result panel
    app.chat_input[0].set_value(DEMO_QUESTION).run()
    assert not app.exception, app.exception
    assert [message.name for message in app.chat_message] == ["user", "assistant"]
    text = page_text(app)
    assert "£40,000" in text and "£58,000" in text
    assert "HIGH" in text and "Review Required" in text
    assert "This investigation is paused." in text
    labels = [expander.label for expander in app.expander]
    assert any(label.startswith("HIGH · Contract.pdf states the total amount as £40,000") for label in labels)
    assert any(label.startswith("Fact · ") for label in labels) and any(label.startswith("Sources (") for label in labels)
    assert "Contract.pdf — Page 4" in text and "Invoice_102.pdf — Page 1" in text
    assert "7.1 The total contract value is £40,000" in text  # the quote on the evidence card
    assert "Written approval or a signed variation" in text  # missing evidence

    # ---- decide: modify the recommendation
    decision = next(radio for radio in app.radio if radio.label == "Your decision")
    decision.set_value("modify")
    next(field for field in app.text_input if field.label.startswith("Recommendation to use instead")).set_value(
        "Pay £40,000 now and dispute the rest."
    )
    app.text_area[0].set_value("Agreed with finance.")
    next(button for button in app.button if button.label == "Submit decision").click().run()
    assert not app.exception, app.exception
    text = page_text(app)
    assert "Human review: **modified**" in text and "Agreed with finance." in text
    assert "Pay £40,000 now and dispute the rest." in text
    assert "The agent had recommended:" in text
    assert "This investigation is paused." not in text
    assert "Decision recorded: **modified**" in text

    # ---- the investigation is in the history and its note is in memory
    history = [button for button in app.sidebar.button if button.label.startswith("High")]
    assert len(history) == 1 and "waiting for a decision" not in history[0].label
    assert "Earlier investigation of" in page_text(app) and "Human decision: modified" in page_text(app)

    # ---- a new conversation clears the page; the history brings the report back
    next(button for button in app.sidebar.button if button.label == "New conversation").click().run()
    assert "The report of the latest investigation appears here." in page_text(app)
    next(button for button in app.sidebar.button if button.label.startswith("High")).click().run()
    assert "Pay £40,000 now and dispute the rest." in page_text(app)


def test_a_preference_stated_in_chat_shows_up_in_the_memory_panel(api_url, user, monkeypatch):
    app = open_app(api_url, user, monkeypatch)
    detail = next(box for box in app.sidebar.selectbox if box.label == "Detail level")
    assert detail.value == "standard"

    app.chat_input[0].set_value("From now on keep your answers brief.").run()
    assert not app.exception, app.exception
    assert "Saved your preference: detail level = brief." in page_text(app)
    detail = next(box for box in app.sidebar.selectbox if box.label == "Detail level")
    assert detail.value == "brief"

    # Changing it in the panel is stored through the API.
    detail.set_value("detailed").run()
    with httpx.Client(base_url=api_url, headers={"X-User-Id": user}) as client:
        stored = {item["key"]: item["value"] for item in client.get("/memory").json()["preferences"]}
    assert stored["detail_level"] == "detailed"


def test_documents_can_be_deleted_and_an_unreachable_api_is_explained(api_url, user, samples_dir, monkeypatch):
    upload_samples(api_url, user, samples_dir)
    app = open_app(api_url, user, monkeypatch)
    delete_buttons = [button for button in app.sidebar.button if button.key and button.key.startswith("delete-document-")]
    assert len(delete_buttons) == 3
    delete_buttons[0].click().run()
    assert "Deleted " in page_text(app)
    with httpx.Client(base_url=api_url, headers={"X-User-Id": user}) as client:
        assert len(client.get("/documents").json()["documents"]) == 2

    offline = open_app(f"http://127.0.0.1:{free_port()}", user, monkeypatch)
    assert "Cannot reach the API" in page_text(offline)
