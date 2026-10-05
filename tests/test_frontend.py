"""The React interface as served by the API: files, security headers, and that streams still stream."""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI

from app.web import CONTENT_SECURITY_POLICY, NOT_BUILT, mount_interface


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
    assert page.headers["referrer-policy"] == "no-referrer" and page.headers["cache-control"] == "no-cache"

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
