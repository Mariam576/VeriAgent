"""Sign in with Google: check the ID token Google gave the browser.

How it works
    The page shows Google's own sign-in button. After the person signs in,
    Google hands the page a signed ID token (a JWT) that says "this is
    alice@gmail.com, Google account 1234...". The page sends it to
    ``POST /auth/google``. The token must not simply be believed; this module
    checks that

    * Google really signed it (against Google's public keys),
    * it was issued for THIS application (``aud`` = GOOGLE_CLIENT_ID) - a token
      issued for some other website must not sign anyone in here,
    * it comes from Google (``iss``) and has not expired.

    No client secret is needed for this flow, only the client id.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jwt

from app.errors import ServiceUnavailable, Unauthorized

GOOGLE_KEYS_URL = "https://www.googleapis.com/oauth2/v3/certs"
GOOGLE_ISSUERS = ("https://accounts.google.com", "accounts.google.com")

_key_client: jwt.PyJWKClient | None = None


@dataclass(frozen=True)
class GoogleIdentity:
    sub: str  # Google's permanent id for the account
    email: str
    email_verified: bool
    name: str


def _google_key(credential: str) -> Any:
    """Google's public key for this token (fetched over HTTPS and cached by the library)."""
    global _key_client
    if _key_client is None:
        _key_client = jwt.PyJWKClient(GOOGLE_KEYS_URL, cache_keys=True, lifespan=3600, timeout=10)
    return _key_client.get_signing_key_from_jwt(credential).key


def verify_google_credential(
    credential: str, client_id: str, key_for: Callable[[str], Any] = _google_key
) -> GoogleIdentity:
    """Blocking. ``key_for`` exists so tests can supply their own key instead of calling Google."""
    try:
        key = key_for(credential)
    except jwt.PyJWKClientConnectionError as exc:
        raise ServiceUnavailable("Google sign-in is temporarily unavailable. Try again, or use your password.") from exc
    except jwt.PyJWTError as exc:
        raise Unauthorized("Google sign-in failed. Please try again.") from exc
    try:
        claims = jwt.decode(
            credential, key,
            algorithms=["RS256"],
            audience=client_id,
            options={"require": ["exp", "iat", "sub", "aud", "iss"]},
            leeway=30,  # small clock differences between Google and this server
        )
    except jwt.PyJWTError as exc:
        raise Unauthorized("Google sign-in failed. Please try again.") from exc
    if claims.get("iss") not in GOOGLE_ISSUERS:
        raise Unauthorized("Google sign-in failed. Please try again.")
    email = str(claims.get("email") or "").strip().lower()
    if not email or "@" not in email:
        raise Unauthorized("Your Google account did not share an email address.")
    return GoogleIdentity(
        sub=str(claims["sub"]), email=email, email_verified=bool(claims.get("email_verified")),
        name=str(claims.get("name") or email.split("@")[0])[:120],
    )


async def verify_google_credential_async(credential: str, client_id: str) -> GoogleIdentity:
    # Fetching Google's keys is a blocking HTTP call, so it runs in a worker thread.
    return await asyncio.to_thread(verify_google_credential, credential, client_id)
