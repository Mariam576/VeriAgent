"""Background jobs and upkeep (Version 3, milestone 20)."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pymupdf
import pytest
from sqlalchemy import select, update

from app.config import get_settings
from app.jobs import BackgroundJobs, get_jobs
from tests.conftest import RUN_ID


# ================================================================ the runner
async def test_jobs_run_after_submission_and_failures_stay_contained(caplog):
    jobs = BackgroundJobs(max_concurrent=2)
    done: list[str] = []

    async def good() -> None:
        done.append("good")

    async def bad() -> None:
        raise RuntimeError("this job fails")

    jobs.submit("bad", bad)
    jobs.submit("good", good)
    assert done == [] and jobs.active == 2  # nothing ran yet: submitting does not wait
    assert await jobs.wait_idle(5)
    assert done == ["good"] and jobs.active == 0
    assert "job_failed name=bad" in caplog.text  # logged, not raised into anyone's request


async def test_no_more_jobs_run_at_once_than_allowed():
    jobs = BackgroundJobs(max_concurrent=2)
    running = peak = 0

    async def work() -> None:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.02)
        running -= 1

    for index in range(6):
        jobs.submit(f"work-{index}", work)
    assert await jobs.wait_idle(5)
    assert peak == 2


async def test_periodic_jobs_repeat_survive_errors_and_stop_at_shutdown():
    jobs = BackgroundJobs()
    rounds = 0

    async def upkeep() -> None:
        nonlocal rounds
        rounds += 1
        if rounds == 1:
            raise RuntimeError("a bad round")

    jobs.every("upkeep", 0.01, upkeep, first_after=0)
    await asyncio.sleep(0.08)
    assert rounds >= 3  # it kept going after the failed first round
    await jobs.shutdown()
    seen = rounds
    await asyncio.sleep(0.05)
    assert rounds == seen


async def test_shutdown_waits_briefly_then_cancels():
    jobs = BackgroundJobs()
    finished: list[str] = []

    async def quick() -> None:
        await asyncio.sleep(0.02)
        finished.append("quick")

    async def endless() -> None:
        await asyncio.sleep(60)
        finished.append("endless")

    jobs.submit("quick", quick)
    jobs.submit("endless", endless)
    await jobs.shutdown(timeout=0.2)
    assert finished == ["quick"] and jobs.active == 0


# ====================================================== uploads in the background
integration = pytest.mark.integration


async def upload(client, headers, path: Path, **params):
    with path.open("rb") as handle:
        return await client.post("/documents/upload", headers=headers, files={"file": (path.name, handle)}, params=params)


@integration
async def test_background_upload_answers_at_once_and_finishes_later(client, user_headers, samples_dir, monkeypatch):
    from app.rag import ingestion

    gate = asyncio.Event()
    real = ingestion.process_document

    async def held_back(*args, **kwargs):
        await gate.wait()  # the slow part does not start until the test says so
        return await real(*args, **kwargs)

    monkeypatch.setattr(ingestion, "process_document", held_back)
    response = await upload(client, user_headers, samples_dir / "Contract.pdf", background="true")
    assert response.status_code == 202, response.text
    document = response.json()
    assert (document["status"], document["chunk_count"], document["duplicate"]) == ("processing", 0, False)

    # While it is being processed: visible, not searchable, and the same file is not started twice.
    assert (await client.get(f"/documents/{document['id']}", headers=user_headers)).json()["status"] == "processing"
    again = await upload(client, user_headers, samples_dir / "Contract.pdf", background="true")
    assert again.status_code == 202 and again.json()["id"] == document["id"] and again.json()["duplicate"] is True
    assert get_jobs().active == 1
    answer = (await client.post("/investigations", headers=user_headers, json={"question": "What is the total contract value?"})).json()
    assert answer["status"] == "insufficient_evidence"

    gate.set()
    assert await get_jobs().wait_idle(30)
    ready = (await client.get(f"/documents/{document['id']}", headers=user_headers)).json()
    assert ready["status"] == "ready" and ready["chunk_count"] >= 5 and ready["page_count"] == 5
    assert ready["document_type"] == "contract" and ready["integrity_level"] == "none"

    # Uploading a file that is already there needs no job at all.
    known = await upload(client, user_headers, samples_dir / "Contract.pdf", background="true")
    assert known.status_code == 201 and known.json()["duplicate"] is True and known.json()["status"] == "ready"
    stranger = {"X-User-Id": f"test-{RUN_ID}-{uuid.uuid4().hex[:8]}"}
    assert (await client.get(f"/documents/{document['id']}", headers=stranger)).status_code == 404


@integration
async def test_a_background_upload_that_fails_says_so_on_the_document(client, user_headers, tmp_path):
    empty = tmp_path / "Scan_without_text.pdf"
    document = pymupdf.open()
    document.new_page()  # a page with nothing on it, like a scan without a text layer
    document.save(empty)
    document.close()

    response = await upload(client, user_headers, empty, background="true")
    assert response.status_code == 202  # the file itself is a valid PDF, so the quick check passes
    assert await get_jobs().wait_idle(30)
    failed = (await client.get(f"/documents/{response.json()['id']}", headers=user_headers)).json()
    assert failed["status"] == "failed" and "No text could be extracted" in failed["error"]

    # A file that is not what its name says is refused at once, background or not.
    fake = tmp_path / "Fake.pdf"
    fake.write_bytes(b"this is not a pdf")
    assert (await upload(client, user_headers, fake, background="true")).status_code in (400, 415, 422)


@integration
async def test_a_document_deleted_while_it_is_processed_does_no_harm(client, user_headers, samples_dir, monkeypatch):
    from app.database.models import Document, DocumentChunk
    from app.database.session import get_session_factory
    from app.rag import ingestion

    gate = asyncio.Event()
    real = ingestion.process_document

    async def held_back(*args, **kwargs):
        await gate.wait()
        return await real(*args, **kwargs)

    monkeypatch.setattr(ingestion, "process_document", held_back)
    document = (await upload(client, user_headers, samples_dir / "Invoice_102.pdf", background="true")).json()
    assert (await client.delete(f"/documents/{document['id']}", headers=user_headers)).status_code == 204
    gate.set()
    assert await get_jobs().wait_idle(30)
    async with get_session_factory()() as session:
        assert await session.get(Document, uuid.UUID(document["id"])) is None
        assert (await session.execute(select(DocumentChunk.id).where(DocumentChunk.document_id == uuid.UUID(document["id"])))).first() is None


@integration
async def test_uploaded_files_need_not_be_kept(client, user_headers, samples_dir, monkeypatch):
    """Where the disk is not permanent: everything the application needs is in the database."""
    settings = get_settings()
    monkeypatch.setattr(settings, "KEEP_UPLOADED_FILES", False)
    document = (await upload(client, user_headers, samples_dir / "forensics" / "Invoice_205_altered.pdf")).json()
    assert document["status"] == "ready" and document["integrity_level"] == "high"
    assert list(Path(settings.UPLOAD_DIR).rglob(f"{document['id']}.*")) == []  # the file is gone ...
    forensics = (await client.get(f"/documents/{document['id']}/forensics", headers=user_headers)).json()
    assert forensics["report"]["level"] == "high"  # ... the inspection result is not
    profile = (await client.get(f"/documents/{document['id']}/profile", headers=user_headers)).json()
    assert profile["profile"]["number"]["text"] == "INV-205"  # ... and neither is the text

    monkeypatch.setattr(settings, "KEEP_UPLOADED_FILES", True)
    kept = (await upload(client, user_headers, samples_dir / "Invoice_102.pdf")).json()
    assert len(list(Path(settings.UPLOAD_DIR).rglob(f"{kept['id']}.*"))) == 1


# ===================================================================== upkeep
@integration
async def test_upkeep_removes_expired_sign_ins_and_closes_stuck_documents(client, user_headers, samples_dir):
    from app.database.models import Document, RefreshToken, User
    from app.database.session import get_session_factory
    from app.maintenance import STUCK_MESSAGE, run_maintenance

    ready = (await upload(client, user_headers, samples_dir / "Invoice_102.pdf")).json()
    now = datetime.now(timezone.utc)
    async with get_session_factory()() as session:
        user = (await session.execute(select(User).where(User.name == user_headers["X-User-Id"]))).scalar_one()
        family = uuid.uuid4()
        session.add_all([
            RefreshToken(user_id=user.id, family_id=family, token_hash=uuid.uuid4().hex, expires_at=now - timedelta(days=1)),
            RefreshToken(user_id=user.id, family_id=family, token_hash=uuid.uuid4().hex, expires_at=now + timedelta(days=1)),
        ])
        stuck = Document(user_id=user.id, filename="Stuck.pdf", file_hash=uuid.uuid4().hex, status="processing", uploaded_at=now - timedelta(hours=2))
        fresh = Document(user_id=user.id, filename="Fresh.pdf", file_hash=uuid.uuid4().hex, status="processing", uploaded_at=now)
        session.add_all([stuck, fresh])
        await session.commit()
        user_id, stuck_id, fresh_id = user.id, stuck.id, fresh.id

    done = await run_maintenance()
    assert done["expired_sign_ins"] >= 1 and done["stuck_documents"] >= 1

    async with get_session_factory()() as session:
        tokens = (await session.execute(select(RefreshToken).where(RefreshToken.user_id == user_id))).scalars().all()
        assert len(tokens) == 1 and tokens[0].expires_at > now
        stuck_row, fresh_row = await session.get(Document, stuck_id), await session.get(Document, fresh_id)
        assert (stuck_row.status, stuck_row.error) == ("failed", STUCK_MESSAGE)
        assert fresh_row.status == "processing"  # still within its time
        assert (await session.get(Document, uuid.UUID(ready["id"]))).status == "ready"
        await session.execute(update(Document).where(Document.id == fresh_id).values(status="failed"))
        await session.commit()
    assert (await run_maintenance())["stuck_documents"] == 0  # nothing left to do


# ============================================================ found in review
@integration
async def test_after_a_restart_an_interrupted_document_fails_at_once_and_can_be_uploaded_again(client, user_headers, samples_dir, monkeypatch):
    """A document caught by a restart said "Reading..." for 15 to 75 minutes, and uploading it again did nothing."""
    from app.maintenance import STUCK_MESSAGE, run_maintenance
    from app.rag import ingestion

    async def lost_in_the_restart(*args, **kwargs):
        return None  # the job vanished with the old process: nothing is ever written

    monkeypatch.setattr(ingestion, "process_document", lost_in_the_restart)
    first = (await upload(client, user_headers, samples_dir / "Contract.pdf", background="true")).json()
    assert await get_jobs().wait_idle(10)
    assert (await client.get(f"/documents/{first['id']}", headers=user_headers)).json()["status"] == "processing"
    monkeypatch.undo()

    # The ordinary round leaves it alone: it was uploaded seconds ago.
    assert (await client.get(f"/documents/{first['id']}", headers=user_headers)).json()["status"] == "processing"
    # The round at start-up knows that no job can be alive.
    assert (await run_maintenance(at_startup=True))["stuck_documents"] >= 1
    failed = (await client.get(f"/documents/{first['id']}", headers=user_headers)).json()
    assert (failed["status"], failed["error"]) == ("failed", STUCK_MESSAGE)

    again = await upload(client, user_headers, samples_dir / "Contract.pdf", background="true")
    assert again.status_code == 202 and again.json()["duplicate"] is False
    assert await get_jobs().wait_idle(30)
    listed = (await client.get("/documents", headers=user_headers)).json()["documents"]
    assert [(item["filename"], item["status"], item["error"]) for item in listed] == [("Contract.pdf", "ready", None)]


@integration
async def test_a_document_whose_job_is_still_waiting_is_not_given_up_on(client, user_headers, samples_dir, monkeypatch):
    """The 15 minutes counted from the upload, so a job that waited in a long queue was declared lost while it was alive."""
    from app.database.models import Document
    from app.database.session import get_session_factory
    from app.maintenance import documents_with_a_job, run_maintenance
    from app.rag import ingestion

    gate = asyncio.Event()
    real = ingestion.process_document

    async def held_back(*args, **kwargs):
        await gate.wait()
        return await real(*args, **kwargs)

    monkeypatch.setattr(ingestion, "process_document", held_back)
    document = (await upload(client, user_headers, samples_dir / "Invoice_102.pdf", background="true")).json()
    document_id = uuid.UUID(document["id"])
    async with get_session_factory()() as session:  # as if it had been uploaded two hours ago
        await session.execute(update(Document).where(Document.id == document_id).values(uploaded_at=datetime.now(timezone.utc) - timedelta(hours=2)))
        await session.commit()
    assert documents_with_a_job() == {document_id}

    await run_maintenance()
    assert (await client.get(f"/documents/{document['id']}", headers=user_headers)).json()["status"] == "processing"

    gate.set()
    assert await get_jobs().wait_idle(30)
    assert documents_with_a_job() == set()
    ready = (await client.get(f"/documents/{document['id']}", headers=user_headers)).json()
    assert (ready["status"], ready["error"]) == ("ready", None)


def test_job_names_are_visible_while_the_job_is_alive():
    async def scenario() -> None:
        jobs = BackgroundJobs(max_concurrent=1)
        gate = asyncio.Event()
        jobs.submit("document:first", gate.wait)
        jobs.submit("document:second", gate.wait)  # waits for its turn
        await asyncio.sleep(0)
        assert jobs.names() == {"document:first", "document:second"}
        gate.set()
        assert await jobs.wait_idle(5) and jobs.names() == set()

    asyncio.run(scenario())
