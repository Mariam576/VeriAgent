"""Who is making this request?

Concept - authentication against authorisation:
    *Authentication* answers "who are you?" and happens here, once per
    request. *Authorisation* answers "may you touch this?" and happens in
    every query: all of them filter by the current user's id, so one user can
    never read another user's documents, investigations or memory.

Two modes (``AUTH_MODE``):

jwt (the default)
    The request carries ``Authorization: Bearer <access token>``. The token is
    obtained from ``/auth/login``, ``/auth/register`` or ``/auth/google``.

dev
    The ``X-User-Id`` header is trusted, with no proof at all. This exists for
    the automated tests and for experiments on your own computer. Never use it
    on a server that other people can reach.
"""

from __future__ import annotations

import re
import uuid
from typing import Annotated

from fastapi import Depends, Header
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import repositories
from app.database.models import User
from app.database.session import get_session
from app.errors import Unauthorized, ValidationFailed
from app.security import tokens

DEFAULT_USER_NAME = "demo-user"
_USER_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.@\-]{1,64}$")
_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://veriagent.local/users")


def user_uuid_for(name: str) -> uuid.UUID:
    """Stable id for a dev-mode user name, so the same header always maps to the same user."""
    return uuid.uuid5(_NAMESPACE, name)


# Declared as a security scheme so that the "Authorize" button on /docs works:
# paste the access_token from /auth/login there. auto_error=False: a missing
# token is handled below, with our own error message.
_bearer = HTTPBearer(auto_error=False, description="The access_token from /auth/login, /auth/register or /auth/google.")


async def get_current_user(
    session: Annotated[AsyncSession, Depends(get_session)],
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)] = None,
    x_user_id: Annotated[str | None, Header(description="AUTH_MODE=dev only: any short name, e.g. 'alice'.")] = None,
) -> User:
    settings = get_settings()
    token = credentials.credentials.strip() if credentials and credentials.credentials.strip() else None
    if token is not None:
        user_id = tokens.read_access_token(token, await tokens.signing_secret(session))
        user = await repositories.get_user(session, user_id)
        if user is None or not user.is_active:
            # Deleted or disabled after the token was issued.
            raise Unauthorized("Please sign in again.")
        return user
    if settings.AUTH_MODE == "dev":
        name = (x_user_id or DEFAULT_USER_NAME).strip()
        if not _USER_NAME_PATTERN.match(name):
            raise ValidationFailed("X-User-Id may only contain letters, digits and . _ @ - (max 64 characters).")
        return await repositories.get_or_create_user(session, user_uuid_for(name), name)
    raise Unauthorized("Sign in to use VeriAgent.")


CurrentUser = Annotated[User, Depends(get_current_user)]
DbSession = Annotated[AsyncSession, Depends(get_session)]
