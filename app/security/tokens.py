"""Access tokens (JWT) and refresh tokens.

Concept - two tokens:
    * The **access token** is a JWT: a small signed text that says "this is
      user X, valid until 14:30". The API checks the signature and needs no
      database lookup to trust it. It is short-lived (ACCESS_TOKEN_MINUTES),
      so a stolen one is useless soon.
    * The **refresh token** is a long random string that is exchanged for a
      new access token. It lives in an httpOnly cookie, which JavaScript
      cannot read, and only its hash is stored in the database, so it can be
      revoked (sign out) and stolen copies can be detected.

Concept - what a JWT is not:
    The content of a JWT is readable by anyone who has it (it is encoded, not
    encrypted). It proves who issued it, nothing more. So it carries the user
    id and an expiry, and nothing confidential.
"""

from __future__ import annotations

import hashlib
import secrets
import uuid
from datetime import datetime, timedelta, timezone

import jwt
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database.models import AppSecret
from app.errors import Unauthorized

ALGORITHM = "HS256"
ISSUER = "veriagent"
MIN_SECRET_CHARS = 32

_signing_secret: str | None = None


async def signing_secret(session: AsyncSession) -> str:
    """The key that signs access tokens: JWT_SECRET, or one the app created and keeps in the database."""
    global _signing_secret
    if _signing_secret is not None:
        return _signing_secret
    configured = get_settings().JWT_SECRET
    if configured is not None:
        value = configured.get_secret_value()
        if len(value) < MIN_SECRET_CHARS:
            raise RuntimeError(f"JWT_SECRET must have at least {MIN_SECRET_CHARS} characters.")
        _signing_secret = value
        return value
    # Not configured: create one once and share it through the database, so that it
    # survives restarts and is the same for every API process.
    await session.execute(
        pg_insert(AppSecret)
        .values(name="jwt_secret", value=secrets.token_urlsafe(48))
        .on_conflict_do_nothing(index_elements=[AppSecret.name])
    )
    value = await session.scalar(select(AppSecret.value).where(AppSecret.name == "jwt_secret"))
    await session.commit()
    _signing_secret = value
    return value


def reset_signing_secret() -> None:
    """Forget the cached key (tests)."""
    global _signing_secret
    _signing_secret = None


def create_access_token(user_id: uuid.UUID, secret: str, now: datetime | None = None) -> tuple[str, int]:
    """Returns (token, seconds until it expires)."""
    now = now or datetime.now(timezone.utc)
    lifetime = timedelta(minutes=get_settings().ACCESS_TOKEN_MINUTES)
    claims = {"sub": str(user_id), "iat": now, "exp": now + lifetime, "iss": ISSUER, "typ": "access"}
    return jwt.encode(claims, secret, algorithm=ALGORITHM), int(lifetime.total_seconds())


def read_access_token(token: str, secret: str) -> uuid.UUID:
    """The user id inside a valid access token. Raises ``Unauthorized`` for anything else."""
    try:
        claims = jwt.decode(
            token, secret,
            algorithms=[ALGORITHM],  # never trust the algorithm named inside the token
            issuer=ISSUER,
            options={"require": ["exp", "iat", "sub", "iss"]},
        )
        if claims.get("typ") != "access":
            raise Unauthorized("Please sign in again.")
        return uuid.UUID(str(claims["sub"]))
    except jwt.ExpiredSignatureError as exc:
        raise Unauthorized("Your session has expired. Please sign in again.") from exc
    except (jwt.InvalidTokenError, ValueError) as exc:
        raise Unauthorized("Please sign in again.") from exc


def new_refresh_token() -> tuple[str, str]:
    """Returns (the token for the client, its hash for the database)."""
    token = secrets.token_urlsafe(48)
    return token, hash_refresh_token(token)


def hash_refresh_token(token: str) -> str:
    # A fast hash is fine here (unlike for passwords): the token is 48 random
    # bytes, so there is nothing to guess.
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
