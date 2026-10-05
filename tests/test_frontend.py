"""The React interface as served by the API: files, security headers, and that streams still stream."""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI, Request
from pydantic import BaseModel

from app.web import CONTENT_SECURITY_POLICY, MAX_JSON_BODY_BYTES, NOT_BUILT, UPLOAD_OVERHEAD_BYTES, mount_interface


def app_with(directory) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    mount_interface(app, str(directory))
    return app


async def get(app: FastAPI, path: str) -> httpx.Response:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        return await client.get(path)


@pytest.fixture
def built(tmp_path):
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<!doctype html><title>VeriAgent</title><div id=root></div>")
    (tmp_path / "assets" / "index-abc123.js").write_text("console.log('interface')")
    return tmp_path


async def test_the_interface_is_served_with_strict_headers(built):
    app = app_with(built)
    page = await get(app, "/")
    assert page.status_code == 200 and "<title>VeriAgent</title>" in page.text
    assert page.headers["content-security-policy"] == CONTENT_SECURITY_POLICY
    assert page.headers["x-content-type-options"] == "nosniff" and page.headers["x-frame-options"] == "DENY"
    assert page.headers["referrer-policy"] == "strict-origin-when-cross-origin" and page.headers["cache-control"] == "no-cache"

    script = await get(app, "/assets/index-abc123.js")
    assert script.status_code == 200 and "javascript" in script.headers["content-type"]
    assert script.headers["cache-control"] == "public, max-age=31536000, immutable"


def test_the_policy_allows_no_inline_script_and_no_foreign_origin_but_googles_button():
    directives = dict(part.split(" ", 1) for part in CONTENT_SECURITY_POLICY.split("; "))
    assert directives["script-src"] == "'self' https://accounts.google.com/gsi/client"
    assert "'unsafe-inline'" not in directives["script-src"] and "'unsafe-eval'" not in CONTENT_SECURITY_POLICY
    assert directives["object-src"] == "'none'" and directives["frame-ancestors"] == "'none'"
    assert directives["connect-src"].startswith("'self'")  # the page talks to its own server


async def test_api_routes_win_over_files_and_unknown_paths_are_404(built):
    app = app_with(built)
    api = await get(app, "/health")
    assert api.json() == {"status": "ok"} and api.headers["x-content-type-options"] == "nosniff"
    assert (await get(app, "/no/such/page")).status_code == 404
    assert (await get(app, "/assets/../index.html")).status_code in (200, 404)  # never a file outside the folder
    assert (await get(app, "/%2e%2e/%2e%2e/etc/passwd")).status_code == 404


async def test_https_is_made_permanent_only_when_the_request_really_came_over_https(built, monkeypatch):
    from app.config import get_settings

    app = app_with(built)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        assert "strict-transport-security" not in (await client.get("/")).headers
        # A header anyone can send is ignored unless a trusted proxy is configured.
        spoofed = await client.get("/", headers={"X-Forwarded-Proto": "https"})
        assert "strict-transport-security" not in spoofed.headers
        monkeypatch.setattr(get_settings(), "TRUST_PROXY_HEADERS", True)
        behind_proxy = await client.get("/", headers={"X-Forwarded-Proto": "https"})
        assert behind_proxy.headers["strict-transport-security"] == "max-age=31536000"
        assert "strict-transport-security" not in (await client.get("/", headers={"X-Forwarded-Proto": "http"})).headers
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://test") as client:
        assert (await client.get("/")).headers["strict-transport-security"] == "max-age=31536000"


async def test_without_a_build_the_root_explains_what_to_do(tmp_path):
    page = await get(app_with(tmp_path / "missing"), "/")
    assert page.status_code == 200 and page.text == NOT_BUILT and "npm run build" in page.text


@pytest.mark.integration
async def test_the_real_app_keeps_streaming_through_the_header_middleware(client, user_headers):
    """Server-Sent Events must still arrive event by event, and the docs page is left without a policy."""
    async with client.stream("POST", "/chat/stream", headers=user_headers, json={"message": "hello"}) as response:
        assert response.status_code == 200 and response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["x-content-type-options"] == "nosniff"
        text = "".join([chunk async for chunk in response.aiter_text()])
    assert "event: started" in text and "event: answer" in text

    docs = await client.get("/docs")
    assert docs.status_code == 200 and "content-security-policy" not in docs.headers
    assert (await client.get("/auth/config")).headers["content-security-policy"] == CONTENT_SECURITY_POLICY


# ------------------------------------------------------------ request size limit
class Login(BaseModel):
    email: str
    password: str


def app_that_reads_bodies(built) -> tuple[FastAPI, list[int]]:
    app = FastAPI()
    seen: list[int] = []

    @app.post("/auth/login")
    async def login(body: Login) -> dict[str, int]:
        seen.append(len(body.password))
        return {"length": len(body.password)}

    @app.post("/documents/upload")
    async def upload(request: Request) -> dict[str, int]:
        size = len(await request.body())
        seen.append(size)
        return {"size": size}

    mount_interface(app, str(built))
    return app, seen


async def test_an_oversized_request_is_refused_before_it_is_read(built):
    """Found in review: one 150 MB sign-in request, from nobody in particular, took the server to 590 MB."""
    app, seen = app_that_reads_bodies(built)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        fine = await client.post("/auth/login", json={"email": "a@b.c", "password": "x" * 1000})
        assert fine.status_code == 200 and fine.json() == {"length": 1000}

        huge = await client.post("/auth/login", json={"email": "a@b.c", "password": "x" * (MAX_JSON_BODY_BYTES + 1)})
        assert huge.status_code == 413 and huge.json()["error"]["code"] == "request_too_large"
        assert huge.headers["x-content-type-options"] == "nosniff"  # the refusal carries the security headers too
        assert seen == [1000]  # the endpoint never saw it


async def test_the_size_is_counted_when_it_is_not_announced(built):
    """Without a Content-Length header (chunked sending) the bytes are counted as they arrive."""
    app, seen = app_that_reads_bodies(built)

    async def pieces():
        yield b'{"email": "a@b.c", "password": "'
        for _ in range(MAX_JSON_BODY_BYTES // 65_536 + 2):
            yield b"x" * 65_536
        yield b'"}'

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        request = client.build_request("POST", "/auth/login", content=pieces(), headers={"content-type": "application/json"})
        assert "content-length" not in request.headers
        response = await client.send(request)
    assert response.status_code == 413 and response.json()["error"]["code"] == "request_too_large"
    assert seen == []


async def test_uploads_have_their_own_larger_limit(built, monkeypatch):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "MAX_UPLOAD_MB", 1)
    app, seen = app_that_reads_bodies(built)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        allowed = await client.post("/documents/upload", content=b"x" * (1024 * 1024))  # far more than a JSON body may have
        assert allowed.status_code == 200 and allowed.json() == {"size": 1024 * 1024}
        too_big = await client.post("/documents/upload", content=b"x" * (1024 * 1024 + UPLOAD_OVERHEAD_BYTES + 1))
        assert too_big.status_code == 413
        assert too_big.json()["error"] == {"code": "file_too_large", "message": "The file is larger than the 1 MB limit."}
        assert (await client.get("/")).status_code == 200  # requests without a body are untouched
    assert seen == [1024 * 1024]


@pytest.mark.integration
async def test_the_real_app_refuses_an_oversized_sign_in(client):
    response = await client.post("/auth/login", json={"email": "someone@example.com", "password": "x" * 300_000})
    assert response.status_code == 413 and response.json()["error"]["code"] == "request_too_large"
