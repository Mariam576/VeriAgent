"""Who is making this request?

AUTHENTICATION PLACEHOLDER (spec section 36).
Version 1 trusts an ``X-User-Id`` header so that the rest of the system can be
built around a real "current user" and real ownership checks. Anyone can send
any value, so this is NOT secure. Before exposing the API beyond your own
machine, replace ``get_current_user`` with real authentication (for example
OAuth2 / JWT). Nothing else has to change, because every route and every
query already works with the user this function returns.
"""

from __future__ import annotations

import re
import uuid
from typing import Annotated

from fastapi import Depends, Header
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import repositories
from app.database.models import User
from app.database.session import get_session
from app.errors import ValidationFailed

DEFAULT_USER_NAME = "demo-user"
_USER_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.@\-]{1,64}$")
_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://veriagent.local/users")


def user_uuid_for(name: str) -> uuid.UUID:
    """Stable id for a user name, so the same header always maps to the same user."""
    return uuid.uuid5(_NAMESPACE, name)


async def get_current_user(
    session: Annotated[AsyncSession, Depends(get_session)],
    x_user_id: Annotated[str | None, Header(description="Placeholder auth: any short name, e.g. 'alice'.")] = None,
) -> User:
    name = (x_user_id or DEFAULT_USER_NAME).strip()
    if not _USER_NAME_PATTERN.match(name):
        raise ValidationFailed("X-User-Id may only contain letters, digits and . _ @ - (max 64 characters).")
    return await repositories.get_or_create_user(session, user_uuid_for(name), name)


CurrentUser = Annotated[User, Depends(get_current_user)]
DbSession = Annotated[AsyncSession, Depends(get_session)]
