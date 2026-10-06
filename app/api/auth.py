"""Sign-in endpoints: register, log in, Google, refresh, log out, account.

How a browser session works here

1. ``POST /auth/login`` (or ``/register``, ``/google``) answers with a
   short-lived **access token** in the JSON body and sets a **refresh token**
   in an httpOnly cookie.
2. The page keeps the access token in memory and sends it with every API call
   as ``Authorization: Bearer ...``.
3. When the access token expires, the page calls ``POST /auth/refresh``. The
   browser sends the cookie by itself; the answer is a new access token and a
   new cookie (the old refresh token stops working: rotation).
4. ``POST /auth/logout`` revokes the refresh token and removes the cookie.

Why not keep everything in the cookie, or everything in localStorage?
    A token in localStorage can be read by any script that runs on the page
    (for example after a cross-site-scripting bug). An httpOnly cookie cannot
    be read by scripts at all, but cookies are sent automatically, which is
    what cross-site request forgery abuses. Splitting the two gives each token
    the safer place: the long-lived one is unreadable, and the API only ever
    acts on the bearer header, which no other site can make the browser send.
"""

from __future__ import annotations

import asyncio
import secrets
import shutil
import uuid
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlparse

from fastapi import APIRouter, Request, Response

from app.config import get_settings
from app.database import repositories
from app.database.models import User, utcnow
from app.errors import Conflict, Forbidden, NotFound, TooManyRequests, Unauthorized, ValidationFailed
from app.schemas.auth import (
    AuthConfig,
    ChangePasswordRequest,
    DeleteAccountRequest,
    GoogleSignInRequest,
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserOut,
)
from app.security import tokens
from app.security.google import verify_google_credential_async
from app.security.passwords import MIN_PASSWORD_CHARS, hash_password_async, needs_rehash, validate_password, verify_password_async
from app.security.permissions import CurrentUser, DbSession
from app.security.rate_limit import RateLimiter, check, client_address

router = APIRouter(prefix="/auth", tags=["auth"])

REFRESH_COOKIE = "veriagent_refresh"
COOKIE_PATH = "/auth"
# Two tabs of the same browser may refresh at the same moment with the same cookie.
# Within this many seconds, the second one is a harmless race, not a stolen token.
REFRESH_RACE_SECONDS = 10

attempts_by_address = RateLimiter(limit=20, window_seconds=60)  # any sign-in request, per client address
# Wrong passwords. Two counters, because each alone has a weakness:
#   * per account AND address (5 in 15 minutes): stops guessing, and a stranger's wrong
#     guesses do not lock the owner out, who comes from another address;
#   * per account from anywhere (20 in 15 minutes): an attacker who changes address for every
#     guess is still stopped. Only this one can lock the owner out for a while, and it takes
#     a sustained effort to keep it up. Signing in with Google is never affected.
failed_logins = RateLimiter(limit=5, window_seconds=900)
failed_logins_per_account = RateLimiter(limit=20, window_seconds=900)


def user_out(user: User) -> UserOut:
    out = UserOut.model_validate(user)
    out.has_password = user.password_hash is not None
    out.has_google = user.google_sub is not None
    return out


def _is_https(request: Request) -> bool:
    if get_settings().TRUST_PROXY_HEADERS and request.headers.get("x-forwarded-proto", "").split(",")[0].strip() == "https":
        return True
    return request.url.scheme == "https"


def _require_same_origin(request: Request) -> None:
    """For the endpoints that act on the cookie alone: refuse calls that another website started.

    The cookie is SameSite=Strict, so browsers already keep it out of
    cross-site requests. This is a second check, in case a browser does not.
    """
    origin = request.headers.get("origin")
    if not origin:
        return  # not a browser cross-origin request (curl, same-origin GET navigations, tests)
    host = request.headers.get("host", "")
    if get_settings().TRUST_PROXY_HEADERS:
        host = request.headers.get("x-forwarded-host", host).split(",")[0].strip()
    if urlparse(origin).netloc != host:
        raise Forbidden("This request did not come from the VeriAgent page.")


def _clear_cookie(response: Response) -> None:
    response.delete_cookie(REFRESH_COOKIE, path=COOKIE_PATH)


def _refresh_lifetime() -> timedelta:
    return timedelta(days=get_settings().REFRESH_TOKEN_DAYS)


async def _answer(
    session: DbSession, request: Request, response: Response, user: User, version: int, refresh_token: str
) -> TokenResponse:
    """The answer to a successful sign-in: an access token in the body, the refresh token in a cookie."""
    access_token, expires_in = tokens.create_access_token(user.id, await tokens.signing_secret(session), version=version)
    response.set_cookie(
        REFRESH_COOKIE, refresh_token,
        max_age=int(_refresh_lifetime().total_seconds()),
        httponly=True,  # page scripts cannot read it
        samesite="strict",  # other websites cannot make the browser send it
        secure=_is_https(request),  # HTTPS only, except on http://localhost
        path=COOKIE_PATH,  # sent to the sign-in endpoints only
    )
    return TokenResponse(access_token=access_token, expires_in=expires_in, user=user_out(user))


async def _issue_tokens(session: DbSession, request: Request, response: Response, user: User) -> TokenResponse:
    """Start a session for an account whose password or Google identity was just checked.

    The check was made against the account as it was read a moment ago. If the
    account has changed since (its password was replaced, its sessions were
    ended), the check no longer counts and no session is started - otherwise a
    sign-in with the old password that was already under way would survive
    the password change (found in review).
    """
    version = user.token_version  # as it was when the caller checked the credentials
    refresh_token, refresh_hash = tokens.new_refresh_token()
    stored = await repositories.store_refresh_token(
        session, user.id, refresh_hash, utcnow() + _refresh_lifetime(), expected_version=version
    )
    if stored is None:
        raise Unauthorized("Please sign in again.")
    return await _answer(session, request, response, user, version, refresh_token)


def _check_signup_allowed(signup_code: str | None) -> None:
    settings = get_settings()
    if not settings.ALLOW_REGISTRATION:
        raise Forbidden("New accounts cannot be created on this server.")
    if settings.SIGNUP_CODE is not None:
        expected = settings.SIGNUP_CODE.get_secret_value()
        # compare_digest takes the same time whether the first or the last character is wrong.
        if not secrets.compare_digest((signup_code or "").encode(), expected.encode()):
            raise Forbidden("A valid sign-up code is needed to create an account.")


@router.get("/config", response_model=AuthConfig)
async def auth_config() -> AuthConfig:
    """What the sign-in page needs: which ways of signing in are available."""
    settings = get_settings()
    return AuthConfig(
        auth_mode=settings.AUTH_MODE,
        registration_open=settings.ALLOW_REGISTRATION,
        signup_code_required=settings.SIGNUP_CODE is not None,
        google_client_id=settings.GOOGLE_CLIENT_ID or None,
        min_password_chars=MIN_PASSWORD_CHARS,
    )


@router.post("/register", response_model=TokenResponse, status_code=201)
async def register(body: RegisterRequest, request: Request, response: Response, session: DbSession) -> TokenResponse:
    check(attempts_by_address, client_address(request), "sign-in attempts")
    _check_signup_allowed(body.signup_code)
    email = body.email.strip().lower()
    validate_password(body.password, email)
    password_hash = await hash_password_async(body.password)
    user = await repositories.create_account(
        session, email=email, name=(body.name or "").strip() or email.split("@")[0], password_hash=password_hash
    )
    if user is None:
        raise Conflict("An account with this email address already exists. Sign in instead.")
    return await _issue_tokens(session, request, response, user)


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, request: Request, response: Response, session: DbSession) -> TokenResponse:
    check(attempts_by_address, client_address(request), "sign-in attempts")
    email = body.email.strip().lower()
    here = f"{email}|{client_address(request)}"
    # Every attempt is counted BEFORE the password is looked at, and taken back if it was right.
    # Counting afterwards would let any number of guesses that arrive together all be checked
    # while the counter still shows "under the limit" (found in review).
    counted: list[tuple[RateLimiter, str]] = []
    for limiter, key in ((failed_logins, here), (failed_logins_per_account, email)):
        if not limiter.hit(key):
            for earlier, earlier_key in counted:
                earlier.take_back(earlier_key)
            raise TooManyRequests(
                f"Too many failed sign-in attempts for this account. Try again in {limiter.retry_after(key)} seconds."
            )
        counted.append((limiter, key))
    user = await repositories.get_user_by_email(session, email)
    await session.commit()
    password_ok = await verify_password_async(body.password, user.password_hash if user else None)
    if user is None or not password_ok or not user.is_active:
        # One message for every failure: it must not reveal whether the account exists.
        raise Unauthorized("Email or password is incorrect.")
    failed_logins.forget(here)
    failed_logins_per_account.take_back(email)
    if needs_rehash(user.password_hash):
        user.password_hash = await hash_password_async(body.password)
        await session.commit()
    return await _issue_tokens(session, request, response, user)


@router.post("/google", response_model=TokenResponse)
async def google_sign_in(body: GoogleSignInRequest, request: Request, response: Response, session: DbSession) -> TokenResponse:
    settings = get_settings()
    check(attempts_by_address, client_address(request), "sign-in attempts")
    if not settings.GOOGLE_CLIENT_ID:
        raise NotFound("Google sign-in is not set up on this server.")
    identity = await verify_google_credential_async(body.credential, settings.GOOGLE_CLIENT_ID)
    if not identity.email_verified:
        raise Unauthorized("Your Google account's email address is not verified.")

    user = await repositories.get_user_by_google_sub(session, identity.sub)
    if user is None:
        user = await repositories.get_user_by_email(session, identity.email)
        if user is not None:
            # Same address, first time with Google: connect the two.
            if not user.email_verified and user.password_hash is not None:
                # Nobody ever proved that this address belongs to whoever chose the password
                # (email is not verified at registration). Google has now proved who owns it,
                # so the old password and every existing session are dropped - the refresh
                # tokens and, through the token version, the access tokens that are still
                # running. Otherwise someone could register with your address first, stay
                # signed in, and set a new password a minute after you took the account back.
                # One step in the database: see repositories.replace_password.
                await repositories.replace_password(session, user.id, None, google_sub=identity.sub, email_verified=True)
                await session.refresh(user)
            else:
                user.google_sub, user.email_verified = identity.sub, True
            await session.commit()
        else:
            _check_signup_allowed(body.signup_code)
            user = await repositories.create_account(
                session, email=identity.email, name=identity.name, google_sub=identity.sub, email_verified=True
            )
            if user is None:  # created by a parallel request a moment ago
                user = await repositories.get_user_by_email(session, identity.email)
    await session.commit()
    if user is None or not user.is_active:
        raise Unauthorized("This account cannot sign in.")
    return await _issue_tokens(session, request, response, user)


@router.post("/refresh", response_model=TokenResponse)
async def refresh(request: Request, response: Response, session: DbSession) -> TokenResponse:
    """Exchange the refresh cookie for a new access token. The cookie is replaced (rotation)."""
    _require_same_origin(request)
    presented = request.cookies.get(REFRESH_COOKIE)
    if not presented:
        raise Unauthorized("Please sign in.")
    successor, successor_hash = tokens.new_refresh_token()
    # Spending the old token and storing the new one is one step in the database, with the
    # account's row held: a password change at the same moment cannot fall in between.
    token, ok, user = await repositories.exchange_refresh_token(
        session, tokens.hash_refresh_token(presented), successor_hash, utcnow() + _refresh_lifetime()
    )
    if token is None:
        _clear_cookie(response)
        raise Unauthorized("Please sign in again.")
    if not ok:
        spent_long_ago = token.revoked_at is not None and (utcnow() - token.revoked_at).total_seconds() > REFRESH_RACE_SECONDS
        if spent_long_ago:
            # A refresh token is used once. One that returns after it was replaced was copied:
            # end every session that descends from it, for the thief and for the owner alike.
            await repositories.revoke_token_family(session, token.family_id)
            _clear_cookie(response)
        raise Unauthorized("Please sign in again.")
    if user is None or not user.is_active:
        _clear_cookie(response)
        raise Unauthorized("Please sign in again.")
    return await _answer(session, request, response, user, user.token_version, successor)


@router.post("/logout", status_code=204)
async def logout(request: Request, session: DbSession) -> Response:
    """Sign out on this device: the refresh cookie is spent and the access token stops working now."""
    _require_same_origin(request)
    # Whose access tokens end now? Only someone who proves to be signed in may say so: with a
    # refresh cookie that was still live, or with a valid access token. A cookie that was
    # already spent proves nothing - anyone who ever saw an old one could otherwise sign the
    # owner out again and again (found in review).
    user_id: uuid.UUID | None = None
    presented = request.cookies.get(REFRESH_COOKIE)
    if presented:
        token, was_live = await repositories.use_refresh_token(session, tokens.hash_refresh_token(presented))
        if token is not None and was_live:
            user_id = token.user_id
    scheme, _, bearer = request.headers.get("authorization", "").partition(" ")
    if user_id is None and scheme.lower() == "bearer" and bearer.strip():
        try:
            claimed, version = tokens.read_access_claims(bearer.strip(), await tokens.signing_secret(session))
        except Unauthorized:
            claimed = None
        account = await repositories.get_user(session, claimed) if claimed is not None else None
        if account is not None and account.token_version == version:
            user_id = account.id
    if user_id is not None:
        # Other devices of this account keep their refresh cookie: they fetch a new
        # access token at their next request and notice nothing.
        await repositories.end_access_tokens(session, user_id)
    response = Response(status_code=204)
    _clear_cookie(response)
    return response


@router.get("/me", response_model=UserOut)
async def me(user: CurrentUser) -> UserOut:
    return user_out(user)


@router.post("/password", response_model=TokenResponse)
async def change_password(
    body: ChangePasswordRequest, request: Request, response: Response, user: CurrentUser, session: DbSession
) -> TokenResponse:
    """Change the password (or set the first one, for an account that so far only used Google)."""
    if user.email is None:
        raise ValidationFailed("This account has no email address and cannot have a password.")
    if user.password_hash is not None:
        # Counted before it is checked, like a sign-in: a stolen access token must not give
        # unlimited guesses at the current password.
        if not failed_logins_per_account.hit(user.email):
            raise TooManyRequests(
                "Too many failed password attempts for this account. "
                f"Try again in {failed_logins_per_account.retry_after(user.email)} seconds."
            )
        if not await verify_password_async(body.current_password or "", user.password_hash):
            raise Unauthorized("The current password is incorrect.")
        failed_logins_per_account.take_back(user.email)
    validate_password(body.new_password, user.email)
    new_hash = await hash_password_async(body.new_password)
    # New hash, higher token version and every refresh token revoked - one step in the database.
    # Access tokens issued before this moment stop working, on every device, and this one
    # continues with fresh tokens.
    await repositories.replace_password(session, user.id, new_hash)
    await session.refresh(user)
    return await _issue_tokens(session, request, response, user)


@router.delete("/me", status_code=204)
async def delete_account(body: DeleteAccountRequest, user: CurrentUser, session: DbSession) -> Response:
    """Delete the account and everything that belongs to it: documents, investigations, memory."""
    if body.confirm != "DELETE":
        raise ValidationFailed('To delete the account, send {"confirm": "DELETE"}.')
    user_id = user.id
    await repositories.delete_user(session, user_id)  # the database cascades to all of the user's rows
    await asyncio.to_thread(shutil.rmtree, Path(get_settings().UPLOAD_DIR) / str(user_id), True)
    from app.agents.checkpoint import get_checkpointer, purge_stale_checkpoints

    await get_checkpointer()
    await purge_stale_checkpoints()  # saved graph state of the deleted investigations
    response = Response(status_code=204)
    _clear_cookie(response)
    return response
