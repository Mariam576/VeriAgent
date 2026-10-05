"""Serving the React interface from the API itself.

Concept - one origin:
    The interface is a set of static files (HTML, JavaScript, CSS) built from
    ``frontend/``. Serving them from the same address as the API means the
    browser treats pages and API as one "origin". Two things follow:

    * no CORS configuration is needed (and none can be got wrong), and
    * the sign-in cookie can be SameSite=Strict: it is only ever sent by
      pages that came from this very server.

Concept - security headers:
    A few response headers tell the browser to be stricter than its default.
    The Content-Security-Policy is the important one: it lists where scripts
    may come from. Even if some text from a document found a way into the page
    as HTML, the browser would refuse to run a script inside it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings

logger = logging.getLogger(__name__)

# Scripts: only our own files and Google's sign-in button. Styles need 'unsafe-inline' for
# that button and for a few computed widths; no script may be inline.
CONTENT_SECURITY_POLICY = "; ".join([
    "default-src 'self'",
    "script-src 'self' https://accounts.google.com/gsi/client",
    "style-src 'self' 'unsafe-inline' https://accounts.google.com/gsi/style",
    "frame-src https://accounts.google.com/gsi/",
    "connect-src 'self' https://accounts.google.com/gsi/",
    "img-src 'self' data: https://*.googleusercontent.com",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "frame-ancestors 'none'",
])
# The interactive API documentation loads its scripts from a CDN, so the policy is not applied there.
_NO_POLICY = ("/docs", "/redoc", "/openapi.json")

NOT_BUILT = """<!doctype html><meta charset="utf-8"><title>VeriAgent</title>
<body style="font-family: system-ui; max-width: 40rem; margin: 4rem auto; line-height: 1.5">
<h1>VeriAgent API is running</h1>
<p>The web interface has not been built yet. In the project folder run:</p>
<pre>cd frontend
npm install
npm run build</pre>
<p>then reload this page. With Docker (<code>docker compose up --build</code>) this happens by itself.</p>
<p>The API documentation is at <a href="/docs">/docs</a>.</p>"""


class SecurityHeaders:
    """Adds the security and caching headers to every response.

    Written as plain ASGI middleware (not ``BaseHTTPMiddleware``) because it
    must not touch the response body: the progress streams (Server-Sent Events)
    have to reach the browser piece by piece, and a closed connection has to be
    noticed at once.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path: str = scope.get("path", "")
        https = scope.get("scheme") == "https"
        if not https and get_settings().TRUST_PROXY_HEADERS:
            # Behind a hosting platform's proxy the request reaches us as plain HTTP;
            # the proxy says in this header how the browser really connected.
            forwarded = dict(scope.get("headers", [])).get(b"x-forwarded-proto", b"").decode("latin-1")
            https = forwarded.split(",")[0].strip() == "https"

        async def send_with_headers(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                present = {name.lower() for name, _ in headers}

                def add(name: bytes, value: str) -> None:
                    if name not in present:
                        headers.append((name, value.encode()))

                add(b"x-content-type-options", "nosniff")  # never guess a file's type from its content
                add(b"referrer-policy", "no-referrer")
                add(b"x-frame-options", "DENY")  # the page may not be embedded in another site
                if https:
                    # "From now on, only ever talk to this site over HTTPS."
                    add(b"strict-transport-security", "max-age=31536000")
                if not path.startswith(_NO_POLICY):
                    add(b"content-security-policy", CONTENT_SECURITY_POLICY)
                if path.startswith("/assets/"):
                    # File names contain a hash of their content, so they can be cached for good.
                    add(b"cache-control", "public, max-age=31536000, immutable")
                elif path in ("/", "/index.html"):
                    add(b"cache-control", "no-cache")  # always ask whether a newer interface exists
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_headers)


def mount_interface(app: FastAPI, directory: str) -> None:
    """Serve the built interface at "/". Must be called after all API routes are registered."""
    app.add_middleware(SecurityHeaders)
    dist = Path(directory)
    if (dist / "index.html").is_file():
        # Registered last: a request reaches the static files only if no API route matched.
        app.mount("/", StaticFiles(directory=dist, html=True), name="interface")
        logger.info("Serving the web interface from %s", dist)
        return

    @app.get("/", include_in_schema=False, response_class=HTMLResponse)
    async def interface_not_built() -> str:
        return NOT_BUILT
