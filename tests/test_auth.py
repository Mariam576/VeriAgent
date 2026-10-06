"""Sign-in: accounts, passwords, access and refresh tokens, Google, limits."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timedelta, timezone

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from sqlalchemy import select

from app.api import auth as auth_api
from app.config import get_settings
from app.database.models import AppSecret, RefreshToken, User
from app.database.session import get_session_factory
from app.errors import Unauthorized, ValidationFailed
from app.security import tokens
from app.security.google import verify_google_credential
from app.security.passwords import hash_password, validate_password, verify_password
from app.security.rate_limit import RateLimiter
from tests.conftest import RUN_ID

PASSWORD = "correct horse battery"
GOOGLE_CLIENT_ID = "1234567890-test.apps.googleusercontent.com"


# ================================================================== unit level
def test_passwords_are_hashed_with_a_salt_and_verified():
    first, second = hash_password(PASSWORD), hash_password(PASSWORD)
    assert first != second and PASSWORD not in first and first.startswith("$argon2id$")
    assert verify_password(PASSWORD, first) and verify_password(PASSWORD, second)
    assert not verify_password("wrong password!", first)
    assert not verify_password(PASSWORD, None)  # no such account: still takes the time of a real check
    assert not verify_password(PASSWORD, "not-a-hash")


@pytest.mark.parametrize(
    "password",
    ["short", "password123", "aaaaaaaaaaaa", "alice@example.com", "x" * 200],
)
def test_weak_passwords_are_refused(password):
    with pytest.raises(ValidationFailed):
        validate_password(password, "alice@example.com")
    validate_password(PASSWORD, "alice@example.com")  # a good one passes


def test_access_token_round_trip_and_tampering(settings):
    secret = "s" * 48
    user_id = uuid.uuid4()
    token, expires_in = tokens.create_access_token(user_id, secret)
    assert tokens.read_access_token(token, secret) == user_id
    assert expires_in == settings.ACCESS_TOKEN_MINUTES * 60

    with pytest.raises(Unauthorized):
        tokens.read_access_token(token, "another-secret-" + "x" * 40)  # signed by someone else
    header, payload, signature = token.split(".")
    with pytest.raises(Unauthorized):
        tokens.read_access_token(f"{header}.{payload[:-2]}AA.{signature}", secret)  # payload changed
    long_ago = datetime.now(timezone.utc) - timedelta(days=2)
    expired, _ = tokens.create_access_token(user_id, secret, now=long_ago)
    with pytest.raises(Unauthorized, match="expired"):
        tokens.read_access_token(expired, secret)
    # A token that claims "no signature needed" must not be accepted.
    unsigned = jwt.encode({"sub": str(user_id), "iss": "veriagent", "typ": "access", "iat": 1, "exp": 9999999999}, "", algorithm="none")
    with pytest.raises(Unauthorized):
        tokens.read_access_token(unsigned, secret)
    # A correctly signed token of another kind is not an access token.
    other = jwt.encode({"sub": str(user_id), "iss": "veriagent", "typ": "refresh", "iat": int(time.time()), "exp": int(time.time()) + 60}, secret, algorithm="HS256")
    with pytest.raises(Unauthorized):
        tokens.read_access_token(other, secret)
    for garbage in ("", "abc", "a.b.c"):
        with pytest.raises(Unauthorized):
            tokens.read_access_token(garbage, secret)


def test_refresh_tokens_are_random_and_stored_as_a_hash():
    token, token_hash = tokens.new_refresh_token()
    other, _ = tokens.new_refresh_token()
    assert token != other and len(token) >= 60
    assert token_hash == tokens.hash_refresh_token(token) and token not in token_hash and len(token_hash) == 64


def test_rate_limiter_window():
    limiter = RateLimiter(limit=2, window_seconds=0.05)
    assert limiter.hit("a") and limiter.hit("a")
    assert not limiter.hit("a") and not limiter.allowed("a")
    assert limiter.hit("b")  # another key has its own count
    assert limiter.retry_after("a") >= 1
    time.sleep(0.06)
    assert limiter.allowed("a") and limiter.hit("a")
    limiter.forget("a")
    assert limiter.hit("a") and limiter.hit("a")


# ------------------------------------------------------------- Google (unit)
@pytest.fixture(scope="module")
def google_keys():
    """A key pair that plays the role of Google's signing key."""
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private, private.public_key()


def google_credential(private_key, **overrides) -> str:
    now = int(time.time())
    claims = {
        "iss": "https://accounts.google.com", "aud": GOOGLE_CLIENT_ID, "sub": "g-" + uuid.uuid4().hex,
        "email": f"test-{RUN_ID}-{uuid.uuid4().hex[:8]}@gmail.com", "email_verified": True,
        "name": f"test-{RUN_ID}-google", "iat": now, "exp": now + 600,
    }
    claims.update(overrides)
    return jwt.encode({k: v for k, v in claims.items() if v is not None}, private_key, algorithm="RS256")


def test_google_credential_is_checked_thoroughly(google_keys):
    private, public = google_keys
    key_for = lambda credential: public  # noqa: E731

    identity = verify_google_credential(google_credential(private, email="Alice@Gmail.com", sub="g-1"), GOOGLE_CLIENT_ID, key_for)
    assert (identity.sub, identity.email, identity.email_verified) == ("g-1", "alice@gmail.com", True)

    bad = {
        "issued for another website": google_credential(private, aud="another-app.apps.googleusercontent.com"),
        "not from Google": google_credential(private, iss="https://accounts.evil.example"),
        "expired": google_credential(private, iat=int(time.time()) - 7200, exp=int(time.time()) - 3600),
        "no email": google_credential(private, email=None),
        "no subject": google_credential(private, sub=None),
    }
    for reason, credential in bad.items():
        with pytest.raises(Unauthorized):
            verify_google_credential(credential, GOOGLE_CLIENT_ID, key_for)
            pytest.fail(f"accepted a credential that is {reason}")

    stranger = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    with pytest.raises(Unauthorized):  # signed with a key that is not Google's
        verify_google_credential(google_credential(stranger), GOOGLE_CLIENT_ID, key_for)
    with pytest.raises(Unauthorized):  # an HMAC token signed with the public key as the secret
        forged = jwt.encode({"iss": "https://accounts.google.com", "aud": GOOGLE_CLIENT_ID, "sub": "x", "email": "a@b.c", "iat": 1, "exp": 9999999999}, "public-key-bytes-" * 3, algorithm="HS256")
        verify_google_credential(forged, GOOGLE_CLIENT_ID, key_for)
    assert verify_google_credential(google_credential(private, email_verified=False), GOOGLE_CLIENT_ID, key_for).email_verified is False


# ================================================================== API level


@pytest.fixture
def jwt_mode(monkeypatch):
    """Switch the app to real sign-in for one test."""
    settings = get_settings()
    monkeypatch.setattr(settings, "AUTH_MODE", "jwt")
    monkeypatch.setattr(settings, "GOOGLE_CLIENT_ID", GOOGLE_CLIENT_ID)
    for limiter in (auth_api.attempts_by_address, auth_api.failed_logins, auth_api.failed_logins_per_account):
        limiter.reset()
    yield settings
    for limiter in (auth_api.attempts_by_address, auth_api.failed_logins, auth_api.failed_logins_per_account):
        limiter.reset()


@pytest.fixture
def google(google_keys, monkeypatch):
    """Make the API check Google credentials against the test key instead of calling Google."""
    private, public = google_keys

    async def verify(credential: str, client_id: str):
        return verify_google_credential(credential, client_id, key_for=lambda _: public)

    monkeypatch.setattr(auth_api, "verify_google_credential_async", verify)
    return private


def new_email() -> str:
    return f"test-{RUN_ID}-{uuid.uuid4().hex[:8]}@example.com"


async def register(client, email: str | None = None, password: str = PASSWORD, **extra) -> dict:
    email = email or new_email()
    response = await client.post(
        "/auth/register", json={"email": email, "password": password, "name": f"test-{RUN_ID}-{uuid.uuid4().hex[:6]}", **extra}
    )
    assert response.status_code == 201, response.text
    return {**response.json(), "email": email, "headers": {"Authorization": f"Bearer {response.json()['access_token']}"}}


def other_client() -> httpx.AsyncClient:
    """A second browser: its own cookie jar."""
    from app.main import app

    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


@pytest.mark.integration
async def test_register_then_use_the_api(client, jwt_mode):
    account = await register(client, email=f"Test-{RUN_ID}-Mixed@Example.com".replace("Test", "test"))
    assert account["token_type"] == "bearer" and account["expires_in"] == jwt_mode.ACCESS_TOKEN_MINUTES * 60
    assert account["user"]["email"] == account["email"].lower()
    assert account["user"]["has_password"] is True and account["user"]["has_google"] is False
    assert "password" not in str(account["user"]).replace("has_password", "")

    # The refresh token travels in a cookie that scripts cannot read and other sites cannot send.
    cookie = client.cookies.jar.__iter__().__next__()
    assert cookie.name == "veriagent_refresh" and cookie.path == "/auth"
    assert cookie.has_nonstandard_attr("HttpOnly") and cookie.get_nonstandard_attr("SameSite") == "strict"
    assert account["access_token"] not in cookie.value

    me = await client.get("/auth/me", headers=account["headers"])
    assert me.status_code == 200 and me.json()["id"] == account["user"]["id"]
    assert (await client.get("/documents", headers=account["headers"])).json() == {"documents": []}

    async with get_session_factory()() as session:
        stored = await session.get(User, uuid.UUID(account["user"]["id"]))
    assert stored.password_hash.startswith("$argon2id$") and PASSWORD not in stored.password_hash


@pytest.mark.integration
async def test_registration_rules(client, jwt_mode):
    account = await register(client)
    duplicate = await client.post("/auth/register", json={"email": account["email"].upper(), "password": PASSWORD})
    assert duplicate.status_code == 409 and "already exists" in duplicate.json()["error"]["message"]
    for password, reason in (("short", "at least 10"), ("password123", "easy to guess")):
        weak = await client.post("/auth/register", json={"email": new_email(), "password": password})
        assert weak.status_code == 400 and reason in weak.json()["error"]["message"]
    assert (await client.post("/auth/register", json={"email": "not-an-email", "password": PASSWORD})).status_code == 422
    assert (await client.post("/auth/register", json={"email": new_email()})).status_code == 422


@pytest.mark.integration
async def test_login_and_wrong_credentials_look_the_same(client, jwt_mode):
    account = await register(client)
    good = await client.post("/auth/login", json={"email": account["email"], "password": PASSWORD})
    assert good.status_code == 200 and good.json()["user"]["id"] == account["user"]["id"]

    wrong_password = await client.post("/auth/login", json={"email": account["email"], "password": "wrong password!"})
    unknown_account = await client.post("/auth/login", json={"email": new_email(), "password": PASSWORD})
    assert wrong_password.status_code == unknown_account.status_code == 401
    # The same words for both, so the answer does not reveal which accounts exist.
    assert wrong_password.json() == unknown_account.json()
    assert wrong_password.json()["error"]["message"] == "Email or password is incorrect."


@pytest.mark.integration
async def test_repeated_wrong_passwords_lock_the_account_for_a_while(client, jwt_mode):
    account = await register(client)
    for _ in range(5):
        assert (await client.post("/auth/login", json={"email": account["email"], "password": "guess number x"})).status_code == 401
    locked = await client.post("/auth/login", json={"email": account["email"], "password": PASSWORD})
    assert locked.status_code == 429 and "Try again in" in locked.json()["error"]["message"]
    # Someone else is not affected.
    other = await register(client)
    assert (await client.post("/auth/login", json={"email": other["email"], "password": PASSWORD})).status_code == 200


@pytest.mark.integration
async def test_too_many_sign_in_requests_from_one_address_are_slowed_down(client, jwt_mode, monkeypatch):
    monkeypatch.setattr(auth_api, "attempts_by_address", RateLimiter(limit=3, window_seconds=60))
    statuses = [
        (await client.post("/auth/login", json={"email": new_email(), "password": PASSWORD})).status_code for _ in range(5)
    ]
    assert statuses == [401, 401, 401, 429, 429]


@pytest.mark.integration
async def test_without_a_valid_token_nothing_is_reachable(client, jwt_mode):
    for method, path in (
        ("GET", "/documents"), ("GET", "/investigations"), ("GET", "/memory"), ("GET", "/auth/me"),
        ("POST", "/investigations"), ("POST", "/chat"), ("POST", "/investigations/stream"),
    ):
        response = await client.request(method, path, json={"question": "hello", "message": "hello"} if method == "POST" else None)
        assert response.status_code == 401, (path, response.status_code)
        assert response.json()["error"]["code"] == "unauthorized"

    # The development header proves nothing and is ignored now.
    assert (await client.get("/documents", headers={"X-User-Id": "demo-user"})).status_code == 401
    for bad in ("Bearer", "Bearer abc.def.ghi", "Basic dXNlcjpwYXNz", "bearer "):
        assert (await client.get("/documents", headers={"Authorization": bad})).status_code == 401

    account = await register(client)
    # The refresh token is not an access token.
    cookie = next(iter(client.cookies.jar)).value
    assert (await client.get("/documents", headers={"Authorization": f"Bearer {cookie}"})).status_code == 401
    # A token signed with a different key (for example by another installation).
    forged, _ = tokens.create_access_token(uuid.UUID(account["user"]["id"]), "z" * 48)
    assert (await client.get("/documents", headers={"Authorization": f"Bearer {forged}"})).status_code == 401
    # Health and the sign-in configuration stay public.
    assert (await client.get("/health")).status_code == 200
    config = (await client.get("/auth/config")).json()
    assert config == {
        "auth_mode": "jwt", "registration_open": True, "signup_code_required": False,
        "google_client_id": GOOGLE_CLIENT_ID, "min_password_chars": 10,
    }


@pytest.mark.integration
async def test_each_account_sees_only_its_own_data(client, jwt_mode, samples_dir):
    alice, bob = await register(client), await register(client)
    with (samples_dir / "Contract.pdf").open("rb") as handle:
        uploaded = await client.post("/documents/upload", headers=alice["headers"], files={"file": ("Contract.pdf", handle)})
    assert uploaded.status_code == 201
    assert len((await client.get("/documents", headers=alice["headers"])).json()["documents"]) == 1
    assert (await client.get("/documents", headers=bob["headers"])).json()["documents"] == []
    assert (await client.delete(f"/documents/{uploaded.json()['id']}", headers=bob["headers"])).status_code == 404

    report = (await client.post("/investigations", headers=alice["headers"], json={"question": "What is the total contract value?"})).json()
    assert (await client.get(f"/investigations/{report['investigation_id']}", headers=bob["headers"])).status_code == 404
    assert (await client.get(f"/investigations/{report['investigation_id']}", headers=alice["headers"])).status_code == 200


@pytest.mark.integration
async def test_refresh_rotates_the_token_and_a_reused_one_ends_the_session(client, jwt_mode, monkeypatch):
    account = await register(client)
    first_cookie = next(iter(client.cookies.jar)).value

    refreshed = await client.post("/auth/refresh")
    assert refreshed.status_code == 200
    assert refreshed.json()["user"]["id"] == account["user"]["id"]
    second_cookie = next(iter(client.cookies.jar)).value
    assert second_cookie != first_cookie  # rotation: every use replaces the token
    assert (await client.get("/auth/me", headers={"Authorization": f"Bearer {refreshed.json()['access_token']}"})).status_code == 200

    # Another tab of the same browser refreshes with the old cookie a moment later: a harmless race.
    async with other_client() as second_tab:
        raced = await second_tab.post("/auth/refresh", headers={"Cookie": f"veriagent_refresh={first_cookie}"})
        assert raced.status_code == 401
    assert (await client.post("/auth/refresh")).status_code == 200  # the session lives on
    third_cookie = next(iter(client.cookies.jar)).value

    # Much later the old token shows up again: it was copied. Every session that came from it ends.
    monkeypatch.setattr(auth_api, "REFRESH_RACE_SECONDS", 0)
    async with other_client() as thief:
        stolen = await thief.post("/auth/refresh", headers={"Cookie": f"veriagent_refresh={first_cookie}"})
        assert stolen.status_code == 401
    assert (await client.post("/auth/refresh")).status_code == 401
    async with other_client() as owner_again:
        assert (await owner_again.post("/auth/refresh", headers={"Cookie": f"veriagent_refresh={third_cookie}"})).status_code == 401

    async with get_session_factory()() as session:
        rows = (await session.execute(select(RefreshToken).where(RefreshToken.user_id == uuid.UUID(account["user"]["id"])))).scalars().all()
    assert len(rows) >= 3 and all(row.revoked_at is not None for row in rows)
    assert len({row.family_id for row in rows}) == 1
    assert not any(cookie in row.token_hash for row in rows for cookie in (first_cookie, second_cookie, third_cookie))


@pytest.mark.integration
async def test_refresh_needs_the_cookie_and_the_same_origin(client, jwt_mode):
    async with other_client() as nobody:
        assert (await nobody.post("/auth/refresh")).status_code == 401
        assert (await nobody.post("/auth/refresh", headers={"Cookie": "veriagent_refresh=made-up-value"})).status_code == 401
    await register(client)
    cross_site = await client.post("/auth/refresh", headers={"Origin": "https://evil.example"})
    assert cross_site.status_code == 403
    assert (await client.post("/auth/refresh", headers={"Origin": "http://test"})).status_code == 200


@pytest.mark.integration
async def test_logout_ends_the_session(client, jwt_mode):
    await register(client)
    assert (await client.post("/auth/logout")).status_code == 204
    assert len(list(client.cookies.jar)) == 0  # the browser was told to drop the cookie
    assert (await client.post("/auth/refresh")).status_code == 401
    assert (await client.post("/auth/logout")).status_code == 204  # signing out twice is fine


@pytest.mark.integration
async def test_change_password_signs_out_other_sessions(client, jwt_mode):
    account = await register(client)
    async with other_client() as phone:
        assert (await phone.post("/auth/login", json={"email": account["email"], "password": PASSWORD})).status_code == 200

        wrong = await client.post("/auth/password", headers=account["headers"], json={"current_password": "not it at all", "new_password": "a brand new secret"})
        assert wrong.status_code == 401
        weak = await client.post("/auth/password", headers=account["headers"], json={"current_password": PASSWORD, "new_password": "short"})
        assert weak.status_code == 400
        changed = await client.post("/auth/password", headers=account["headers"], json={"current_password": PASSWORD, "new_password": "a brand new secret"})
        assert changed.status_code == 200

        assert (await phone.post("/auth/refresh")).status_code == 401  # the other device is signed out
    assert (await client.post("/auth/refresh")).status_code == 200  # this one continues
    assert (await client.post("/auth/login", json={"email": account["email"], "password": PASSWORD})).status_code == 401
    assert (await client.post("/auth/login", json={"email": account["email"], "password": "a brand new secret"})).status_code == 200


@pytest.mark.integration
async def test_deleting_an_account_removes_everything(client, jwt_mode, samples_dir):
    account = await register(client)
    with (samples_dir / "Invoice_102.pdf").open("rb") as handle:
        await client.post("/documents/upload", headers=account["headers"], files={"file": ("Invoice_102.pdf", handle)})
    await client.post("/memory/notes", headers=account["headers"], json={"text": "Vendor ABC was flagged in 2025."})

    assert (await client.request("DELETE", "/auth/me", headers=account["headers"], json={"confirm": "yes"})).status_code == 400
    deleted = await client.request("DELETE", "/auth/me", headers=account["headers"], json={"confirm": "DELETE"})
    assert deleted.status_code == 204
    assert (await client.get("/auth/me", headers=account["headers"])).status_code == 401  # the token outlived the account
    assert (await client.post("/auth/login", json={"email": account["email"], "password": PASSWORD})).status_code == 401
    async with get_session_factory()() as session:
        assert await session.get(User, uuid.UUID(account["user"]["id"])) is None
    again = await register(client, email=account["email"])  # the address is free again, and starts empty
    assert (await client.get("/documents", headers=again["headers"])).json() == {"documents": []}
    assert (await client.get("/memory", headers=again["headers"])).json()["notes"] == []


@pytest.mark.integration
async def test_sign_up_code_and_closed_registration(client, jwt_mode, monkeypatch):
    from pydantic import SecretStr

    monkeypatch.setattr(jwt_mode, "SIGNUP_CODE", SecretStr("open-sesame-2026"))
    assert (await client.get("/auth/config")).json()["signup_code_required"] is True
    for code in (None, "", "open-sesame-2025"):
        refused = await client.post("/auth/register", json={"email": new_email(), "password": PASSWORD, "signup_code": code})
        assert refused.status_code == 403
    account = await register(client, signup_code="open-sesame-2026")

    monkeypatch.setattr(jwt_mode, "ALLOW_REGISTRATION", False)
    closed = await client.post("/auth/register", json={"email": new_email(), "password": PASSWORD, "signup_code": "open-sesame-2026"})
    assert closed.status_code == 403
    # People who already have an account can still sign in.
    assert (await client.post("/auth/login", json={"email": account["email"], "password": PASSWORD})).status_code == 200


@pytest.mark.integration
async def test_daily_investigation_limit_protects_the_model_quota(client, jwt_mode, monkeypatch):
    monkeypatch.setattr(jwt_mode, "MAX_INVESTIGATIONS_PER_DAY", 2)
    account = await register(client)
    for _ in range(2):
        assert (await client.post("/investigations", headers=account["headers"], json={"question": "hello"})).status_code == 201
    third = await client.post("/investigations", headers=account["headers"], json={"question": "hello"})
    assert third.status_code == 429 and "limit of 2 investigations" in third.json()["error"]["message"]
    assert (await client.post("/chat", headers=account["headers"], json={"message": "hello"})).status_code == 429
    other = await register(client)  # the limit is per account
    assert (await client.post("/investigations", headers=other["headers"], json={"question": "hello"})).status_code == 201


@pytest.mark.integration
async def test_google_sign_in_creates_and_finds_the_account(client, jwt_mode, google):
    email, sub = f"test-{RUN_ID}-{uuid.uuid4().hex[:8]}@gmail.com", "g-" + uuid.uuid4().hex
    first = await client.post("/auth/google", json={"credential": google_credential(google, email=email, sub=sub)})
    assert first.status_code == 200, first.text
    user = first.json()["user"]
    assert user["email"] == email and user["has_google"] is True and user["has_password"] is False and user["email_verified"] is True
    assert (await client.get("/auth/me", headers={"Authorization": f"Bearer {first.json()['access_token']}"})).status_code == 200

    # Signing in again finds the same account, by Google's id, even if the address changed at Google.
    renamed = await client.post("/auth/google", json={"credential": google_credential(google, email=f"test-{RUN_ID}-renamed-{uuid.uuid4().hex[:6]}@gmail.com", sub=sub)})
    assert renamed.json()["user"]["id"] == user["id"]
    # No password was ever set, so there is nothing to guess.
    assert (await client.post("/auth/login", json={"email": email, "password": PASSWORD})).status_code == 401

    refused = {
        "another website's token": google_credential(google, aud="another-app.apps.googleusercontent.com"),
        "unverified address": google_credential(google, email_verified=False),
        "expired": google_credential(google, iat=int(time.time()) - 7200, exp=int(time.time()) - 3600),
    }
    for reason, credential in refused.items():
        assert (await client.post("/auth/google", json={"credential": credential})).status_code == 401, reason


@pytest.mark.integration
async def test_google_takes_back_an_address_someone_else_registered(client, jwt_mode, google):
    """Email is not verified at registration, so a stranger can register with your address first."""
    victim_email = f"test-{RUN_ID}-{uuid.uuid4().hex[:8]}@gmail.com"
    async with other_client() as attacker:
        squatted = await register(attacker, email=victim_email, password="attacker knows this")
        # The real owner arrives and proves ownership through Google.
        owner = await client.post("/auth/google", json={"credential": google_credential(google, email=victim_email)})
        assert owner.status_code == 200 and owner.json()["user"]["id"] == squatted["user"]["id"]
        assert owner.json()["user"]["has_password"] is False
        # The stranger's password and sessions no longer work.
        assert (await attacker.post("/auth/login", json={"email": victim_email, "password": "attacker knows this"})).status_code == 401
        assert (await attacker.post("/auth/refresh")).status_code == 401
    # The owner can set a password of their own (no current password exists to ask for).
    headers = {"Authorization": f"Bearer {owner.json()['access_token']}"}
    assert (await client.post("/auth/password", headers=headers, json={"new_password": "owner's own secret"})).status_code == 200
    assert (await client.post("/auth/login", json={"email": victim_email, "password": "owner's own secret"})).status_code == 200


@pytest.mark.integration
async def test_google_respects_closed_registration_and_missing_setup(client, jwt_mode, google, monkeypatch):
    monkeypatch.setattr(jwt_mode, "ALLOW_REGISTRATION", False)
    assert (await client.post("/auth/google", json={"credential": google_credential(google)})).status_code == 403
    monkeypatch.setattr(jwt_mode, "GOOGLE_CLIENT_ID", "")
    missing = await client.post("/auth/google", json={"credential": google_credential(google)})
    assert missing.status_code == 404 and "not set up" in missing.json()["error"]["message"]
    assert (await client.get("/auth/config")).json()["google_client_id"] is None


@pytest.mark.integration
async def test_signing_key_is_created_once_and_kept_in_the_database(client, jwt_mode):
    tokens.reset_signing_secret()
    first = await register(client)
    async with get_session_factory()() as session:
        stored = await session.scalar(select(AppSecret.value).where(AppSecret.name == "jwt_secret"))
    assert stored and len(stored) >= 48
    assert tokens.read_access_token(first["access_token"], stored) == uuid.UUID(first["user"]["id"])
    tokens.reset_signing_secret()  # as after a restart: the same key is read back, so sessions survive
    assert (await client.get("/auth/me", headers=first["headers"])).status_code == 200


@pytest.mark.integration
async def test_development_mode_still_accepts_the_header_and_tokens(client, jwt_mode, monkeypatch):
    account = await register(client)
    monkeypatch.setattr(jwt_mode, "AUTH_MODE", "dev")
    assert (await client.get("/documents", headers={"X-User-Id": f"test-{RUN_ID}-devuser"})).status_code == 200
    assert (await client.get("/auth/me", headers=account["headers"])).json()["id"] == account["user"]["id"]
    assert (await client.get("/auth/config")).json()["auth_mode"] == "dev"


@pytest.mark.integration
async def test_old_placeholder_data_can_be_adopted_by_an_account(client, jwt_mode, samples_dir, monkeypatch):
    """Data created before sign-in existed (X-User-Id users) moves to a real account."""
    from scripts.adopt_data import adopt, list_placeholder_users

    old_name = f"test-{RUN_ID}-{uuid.uuid4().hex[:8]}"
    monkeypatch.setattr(jwt_mode, "AUTH_MODE", "dev")
    old_headers = {"X-User-Id": old_name}
    with (samples_dir / "Contract.pdf").open("rb") as handle:
        document = (await client.post("/documents/upload", headers=old_headers, files={"file": ("Contract.pdf", handle)})).json()
    report = (await client.post("/investigations", headers=old_headers, json={"question": "What is the total contract value?"})).json()
    await client.put("/memory/preferences/detail_level", headers=old_headers, json={"value": "brief"})
    await client.post("/memory/notes", headers=old_headers, json={"text": "Vendor ABC was flagged in 2025."})
    assert (old_name, 1, 1) in await list_placeholder_users()

    monkeypatch.setattr(jwt_mode, "AUTH_MODE", "jwt")
    account = await register(client)
    counts = await adopt(old_name, account["email"])
    assert counts == {"documents": 1, "conversations": 0, "investigations": 1, "preferences": 1, "notes": 1}

    documents = (await client.get("/documents", headers=account["headers"])).json()["documents"]
    assert [item["id"] for item in documents] == [document["id"]]
    assert (await client.get(f"/investigations/{report['investigation_id']}", headers=account["headers"])).status_code == 200
    memory = (await client.get("/memory", headers=account["headers"])).json()
    assert len(memory["notes"]) == 1 and {p["key"]: p["value"] for p in memory["preferences"]}["detail_level"] == "brief"
    # The adopted document is searchable, and its stored file moved with it.
    again = (await client.post("/investigations", headers=account["headers"], json={"question": "What is the total contract value?"})).json()
    assert any("£40,000" in finding["statement"] for finding in again["key_findings"])
    assert (await client.delete(f"/documents/{document['id']}", headers=account["headers"])).status_code == 204

    assert await adopt(old_name, account["email"]) == {"documents": 0, "conversations": 0, "investigations": 0, "preferences": 0, "notes": 0}
    with pytest.raises(SystemExit, match="no account"):
        await adopt(old_name, "nobody@example.com")
    with pytest.raises(SystemExit, match="no placeholder user"):
        await adopt("no-such-name", account["email"])


# =============================================================== found in review
def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_a_token_carries_the_account_version_it_was_made_with(settings):
    secret, user_id = "s" * 48, uuid.uuid4()
    assert tokens.read_access_claims(tokens.create_access_token(user_id, secret)[0], secret) == (user_id, 0)
    assert tokens.read_access_claims(tokens.create_access_token(user_id, secret, version=7)[0], secret) == (user_id, 7)
    # A token from before the version existed counts as version 0; nonsense is refused.
    old = jwt.encode({"sub": str(user_id), "iat": 1, "exp": 9999999999, "iss": tokens.ISSUER, "typ": "access"}, secret, algorithm="HS256")
    assert tokens.read_access_claims(old, secret) == (user_id, 0)
    odd = jwt.encode({"sub": str(user_id), "iat": 1, "exp": 9999999999, "iss": tokens.ISSUER, "typ": "access", "ver": "x"}, secret, algorithm="HS256")
    with pytest.raises(Unauthorized):
        tokens.read_access_claims(odd, secret)


@pytest.mark.integration
async def test_taking_an_address_back_ends_the_strangers_running_session(client, jwt_mode, google):
    """The stranger's refresh token was revoked, but the access token in their hands lived on for
    half an hour - long enough to set a new password on the account that had just lost its old one."""
    victim_email = f"test-{RUN_ID}-{uuid.uuid4().hex[:8]}@gmail.com"
    async with other_client() as attacker:
        squatted = await register(attacker, email=victim_email, password="attacker knows this")
        assert (await attacker.get("/auth/me", headers=squatted["headers"])).status_code == 200

        owner = await client.post("/auth/google", json={"credential": google_credential(google, email=victim_email)})
        assert owner.status_code == 200

        assert (await attacker.get("/auth/me", headers=squatted["headers"])).status_code == 401
        assert (await attacker.get("/memory", headers=squatted["headers"])).status_code == 401
        takeover = await attacker.post("/auth/password", headers=squatted["headers"], json={"new_password": "attacker's new one"})
        assert takeover.status_code == 401
        assert (await attacker.post("/auth/login", json={"email": victim_email, "password": "attacker's new one"})).status_code == 401
    # The owner's own, new token works.
    assert (await client.get("/auth/me", headers=bearer(owner.json()["access_token"]))).status_code == 200


@pytest.mark.integration
async def test_signing_out_ends_the_access_token_at_once(client, jwt_mode):
    account = await register(client)
    async with other_client() as phone:
        on_phone = (await phone.post("/auth/login", json={"email": account["email"], "password": PASSWORD})).json()
        assert (await client.post("/auth/logout")).status_code == 204
        # The token that was in the browser's memory is worthless now, not in half an hour.
        assert (await client.get("/auth/me", headers=account["headers"])).status_code == 401
        assert (await client.get("/documents", headers=account["headers"])).status_code == 401
        # The phone was not signed out: its old access token is refused, the page fetches a new one, and carries on.
        assert (await phone.get("/auth/me", headers=bearer(on_phone["access_token"]))).status_code == 401
        renewed = await phone.post("/auth/refresh")
        assert renewed.status_code == 200
        assert (await phone.get("/auth/me", headers=bearer(renewed.json()["access_token"]))).status_code == 200


@pytest.mark.integration
async def test_a_changed_password_ends_access_tokens_on_other_devices(client, jwt_mode):
    account = await register(client)
    async with other_client() as phone:
        on_phone = (await phone.post("/auth/login", json={"email": account["email"], "password": PASSWORD})).json()
        changed = await client.post("/auth/password", headers=account["headers"], json={"current_password": PASSWORD, "new_password": "a brand new secret"})
        assert changed.status_code == 200
        assert (await phone.get("/auth/me", headers=bearer(on_phone["access_token"]))).status_code == 401
        assert (await phone.post("/auth/refresh")).status_code == 401
    assert (await client.get("/auth/me", headers=account["headers"])).status_code == 401  # the token from before the change
    assert (await client.get("/auth/me", headers=bearer(changed.json()["access_token"]))).status_code == 200


@pytest.mark.integration
async def test_a_strangers_wrong_guesses_do_not_lock_the_owner_out(client, jwt_mode, monkeypatch):
    """Five wrong passwords by anyone who knew the email address locked the owner out for 15 minutes, again and again."""
    monkeypatch.setattr(jwt_mode, "TRUST_PROXY_HEADERS", True)
    monkeypatch.setattr(jwt_mode, "CLIENT_ADDRESS_HEADER", "x-forwarded-for")
    account = await register(client)
    stranger, owner = {"X-Forwarded-For": "203.0.113.5"}, {"X-Forwarded-For": "198.51.100.7"}
    for _ in range(5):
        assert (await client.post("/auth/login", headers=stranger, json={"email": account["email"], "password": "guess number x"})).status_code == 401
    # The stranger is stopped, even with the right password ...
    assert (await client.post("/auth/login", headers=stranger, json={"email": account["email"], "password": PASSWORD})).status_code == 429
    # ... the owner, who comes from somewhere else, is not.
    assert (await client.post("/auth/login", headers=owner, json={"email": account["email"], "password": PASSWORD})).status_code == 200


@pytest.mark.integration
async def test_changing_address_for_every_guess_does_not_give_unlimited_guesses(client, jwt_mode, monkeypatch):
    monkeypatch.setattr(jwt_mode, "TRUST_PROXY_HEADERS", True)
    monkeypatch.setattr(jwt_mode, "CLIENT_ADDRESS_HEADER", "x-forwarded-for")
    monkeypatch.setattr(auth_api, "failed_logins_per_account", RateLimiter(limit=8, window_seconds=900))
    account = await register(client)
    statuses = []
    for number in range(10):
        response = await client.post(
            "/auth/login", headers={"X-Forwarded-For": f"203.0.113.{number}"}, json={"email": account["email"], "password": f"guess number {number}"}
        )
        statuses.append(response.status_code)
    assert statuses == [401] * 8 + [429] * 2


def test_password_hashing_fits_a_small_server():
    from app.security import passwords

    # 19 MiB per hash (the library default is 64 MiB): OWASP's setting for memory-constrained servers.
    assert passwords._hasher.memory_cost == 19_456 and passwords._hasher.time_cost == 2 and passwords._hasher.parallelism == 1
    assert hash_password(PASSWORD).startswith("$argon2id$v=19$m=19456,t=2,p=1$")
    # A hash made with other settings still verifies, and is renewed at the next sign-in.
    from argon2 import PasswordHasher

    older = PasswordHasher(time_cost=3, memory_cost=65_536, parallelism=4).hash(PASSWORD)
    assert verify_password(PASSWORD, older) and passwords.needs_rehash(older) and not passwords.needs_rehash(hash_password(PASSWORD))


async def test_only_a_few_passwords_are_hashed_at_the_same_moment(monkeypatch):
    """Found in review: 12 sign-in attempts at once, by nobody in particular, used 590 MB on a 512 MB host."""
    import asyncio
    import threading

    from app.security import passwords

    lock, running, most = threading.Lock(), 0, 0

    def slow_verify(password: str, password_hash: str | None) -> bool:
        nonlocal running, most
        with lock:
            running += 1
            most = max(most, running)
        time.sleep(0.03)
        with lock:
            running -= 1
        return True

    monkeypatch.setattr(passwords, "verify_password", slow_verify)
    results = await asyncio.gather(*(passwords.verify_password_async("x", None) for _ in range(12)))
    assert all(results) and most == passwords.MAX_CONCURRENT_HASHES


# ============================================ found in the second review of Version 3
def bearer_of(access_token: str) -> dict:
    return {"Authorization": f"Bearer {access_token}"}


@pytest.mark.integration
async def test_a_spent_refresh_cookie_cannot_sign_the_owner_out(client, jwt_mode):
    """Anyone who ever saw an old cookie could end the owner's access tokens, again and again."""
    await register(client)
    spent = client.cookies.get("veriagent_refresh")
    fresh = (await client.post("/auth/refresh")).json()  # the cookie is replaced; the old one is spent
    async with other_client() as stranger:
        assert (await stranger.post("/auth/logout", cookies={"veriagent_refresh": spent})).status_code == 204
    assert (await client.get("/auth/me", headers=bearer_of(fresh["access_token"]))).status_code == 200


@pytest.mark.integration
async def test_signing_out_with_the_access_token_alone_ends_it(client, jwt_mode):
    """Two tabs: the other tab has just replaced the cookie. Signing out here still ends the access token."""
    account = await register(client)
    async with other_client() as tab:  # no cookie of its own
        assert (await tab.post("/auth/logout", headers=account["headers"])).status_code == 204
        assert (await tab.post("/auth/logout", headers=bearer_of("not-a-token"))).status_code == 204  # harmless
    assert (await client.get("/auth/me", headers=account["headers"])).status_code == 401


@pytest.mark.integration
async def test_a_password_change_cannot_be_overtaken_by_a_stolen_session(client, jwt_mode, monkeypatch):
    """Found in review: while the new password was being hashed, a thief with a copied cookie fetched a
    token that the change did not end - and could then delete the account. Now the change is one step
    in the database, and whatever was issued before it stops working."""
    account = await register(client)
    new_password = "a brand new secret"
    async with other_client() as thief:
        stolen = (await thief.post("/auth/login", json={"email": account["email"], "password": PASSWORD})).json()
        assert stolen["access_token"]
        taken: dict = {}
        hash_as_usual = auth_api.hash_password_async

        async def hash_while_the_thief_acts(password: str) -> str:
            # The moment the review used: the owner's request is waiting for the slow hash.
            first = (await thief.post("/auth/refresh")).json()
            await thief.post("/auth/logout", headers=bearer_of(first["access_token"]), cookies={})  # raises the version
            second = await thief.post("/auth/refresh")
            taken["access"] = second.json().get("access_token") if second.status_code == 200 else first["access_token"]
            return await hash_as_usual(password)

        monkeypatch.setattr(auth_api, "hash_password_async", hash_while_the_thief_acts)
        changed = await client.post(
            "/auth/password", headers=account["headers"], json={"current_password": PASSWORD, "new_password": new_password}
        )
        assert changed.status_code == 200
        assert (await thief.get("/auth/me", headers=bearer_of(taken["access"]))).status_code == 401
        assert (await thief.post("/auth/refresh")).status_code == 401
    async with other_client() as owner:
        assert (await owner.post("/auth/login", json={"email": account["email"], "password": PASSWORD})).status_code == 401
        assert (await owner.post("/auth/login", json={"email": account["email"], "password": new_password})).status_code == 200


@pytest.mark.integration
async def test_the_database_counts_the_token_version_itself(client, jwt_mode):
    """A version worked out from a stale copy would repeat a number that tokens already carry."""
    from app.database import repositories

    account = await register(client)
    user_id = uuid.UUID(account["user"]["id"])
    async with get_session_factory()() as first, get_session_factory()() as second:
        stale = await first.get(User, user_id)
        seen = stale.token_version
        await repositories.end_access_tokens(second, user_id)  # someone else raises it meanwhile
        version = await repositories.replace_password(first, user_id, hash_password("another long secret"))
        assert version == seen + 2
        live = (await second.execute(select(RefreshToken).where(RefreshToken.user_id == user_id, RefreshToken.revoked_at.is_(None)))).scalars().all()
        assert live == []
        # A sign-in that was checked against the account as it was before no longer starts a session.
        assert await repositories.store_refresh_token(
            first, user_id, tokens.new_refresh_token()[1], datetime.now(timezone.utc) + timedelta(days=1), expected_version=seen
        ) is None


@pytest.mark.integration
async def test_wrong_passwords_that_arrive_together_are_all_counted(client, jwt_mode, monkeypatch):
    """Found in review: 300 guesses sent at the same moment were all checked, because each one
    looked at the counter before any of them had been counted."""
    import asyncio

    monkeypatch.setattr(jwt_mode, "TRUST_PROXY_HEADERS", True)
    monkeypatch.setattr(jwt_mode, "CLIENT_ADDRESS_HEADER", "x-forwarded-for")
    monkeypatch.setattr(auth_api, "failed_logins_per_account", RateLimiter(limit=6, window_seconds=900))
    account = await register(client)
    checked = 0
    verify_as_usual = auth_api.verify_password_async

    async def counting(password, password_hash):
        nonlocal checked
        checked += 1
        return await verify_as_usual(password, password_hash)

    monkeypatch.setattr(auth_api, "verify_password_async", counting)

    async def guess(number: int) -> int:
        async with other_client() as guesser:
            response = await guesser.post(
                "/auth/login", headers={"X-Forwarded-For": f"203.0.113.{number}"},
                json={"email": account["email"], "password": f"guess number {number}"},
            )
            return response.status_code

    statuses = await asyncio.gather(*(guess(number) for number in range(30)))
    assert checked == 6 and sorted(statuses) == [401] * 6 + [429] * 24


@pytest.mark.integration
async def test_successful_sign_ins_do_not_use_up_the_failure_counters(client, jwt_mode, monkeypatch):
    monkeypatch.setattr(jwt_mode, "TRUST_PROXY_HEADERS", True)
    monkeypatch.setattr(jwt_mode, "CLIENT_ADDRESS_HEADER", "x-forwarded-for")
    monkeypatch.setattr(auth_api, "failed_logins_per_account", RateLimiter(limit=4, window_seconds=900))
    account = await register(client)
    for number in range(8):
        response = await client.post(
            "/auth/login", headers={"X-Forwarded-For": f"198.51.100.{number}"}, json={"email": account["email"], "password": PASSWORD}
        )
        assert response.status_code == 200


@pytest.mark.integration
async def test_guessing_the_current_password_with_a_stolen_token_is_limited(client, jwt_mode, monkeypatch):
    """The limit was only looked at after a wrong guess, so a right one always went through."""
    monkeypatch.setattr(auth_api, "failed_logins_per_account", RateLimiter(limit=3, window_seconds=900))
    account = await register(client)
    for number in range(3):
        wrong = await client.post("/auth/password", headers=account["headers"], json={"current_password": f"guess number {number}", "new_password": "a brand new secret"})
        assert wrong.status_code == 401
    right = await client.post("/auth/password", headers=account["headers"], json={"current_password": PASSWORD, "new_password": "a brand new secret"})
    assert right.status_code == 429
    assert (await client.get("/auth/me", headers=account["headers"])).status_code == 200  # nothing was changed


async def test_the_queue_for_password_hashing_has_an_end(monkeypatch):
    """A burst of sign-in attempts is told to come back; it does not keep every later visitor waiting."""
    import asyncio
    import time as clock

    from app.errors import TooManyRequests
    from app.security import passwords

    monkeypatch.setattr(passwords, "MAX_WAITING_HASHES", 5)
    monkeypatch.setattr(passwords, "verify_password", lambda password, password_hash: clock.sleep(0.05) or True)
    results = await asyncio.gather(*(passwords.verify_password_async("x", "y") for _ in range(12)), return_exceptions=True)
    assert results.count(True) == 5 and sum(isinstance(result, TooManyRequests) for result in results) == 7
    assert passwords._waiting == 0 and await passwords.verify_password_async("x", "y") is True


# ============================================ found in the third review of Version 3
@pytest.mark.integration
async def test_of_two_password_changes_at_the_same_moment_exactly_one_counts(client, jwt_mode, monkeypatch):
    """Found in the third review: both answered 200, and the password that ended up in the database was not
    the one of whoever was told last that it had worked."""
    account = await register(client)
    async with other_client() as second:
        other_session = (await second.post("/auth/login", json={"email": account["email"], "password": PASSWORD})).json()
        hash_as_usual, running = auth_api.hash_password_async, {"inner": None}

        async def the_other_change_overtakes(password: str) -> str:
            if running["inner"] is None:  # only while the FIRST change is hashing
                running["inner"] = "started"
                running["inner"] = await second.post(
                    "/auth/password", headers=bearer_of(other_session["access_token"]),
                    json={"current_password": PASSWORD, "new_password": "the other new secret"},
                )
            return await hash_as_usual(password)

        monkeypatch.setattr(auth_api, "hash_password_async", the_other_change_overtakes)
        first = await client.post("/auth/password", headers=account["headers"], json={"current_password": PASSWORD, "new_password": "the first new secret"})
        assert (running["inner"].status_code, first.status_code) == (200, 409)
    async with other_client() as later:
        assert (await later.post("/auth/login", json={"email": account["email"], "password": "the first new secret"})).status_code == 401
        assert (await later.post("/auth/login", json={"email": account["email"], "password": "the other new secret"})).status_code == 200


@pytest.mark.integration
async def test_a_password_change_that_was_under_way_does_not_undo_taking_the_address_back(client, jwt_mode, google, monkeypatch):
    """Found in the third review: a stranger who registered your address kept changing "his" password; one of
    those changes landed after you had taken the account back with Google - and he was in again."""
    email = new_email()
    stranger = await register(client, email=email)
    hash_as_usual, taken_back = auth_api.hash_password_async, {}

    async def the_owner_arrives(password: str) -> str:
        if not taken_back:
            async with other_client() as owner:
                taken_back["response"] = await owner.post("/auth/google", json={"credential": google_credential(google, email=email)})
        return await hash_as_usual(password)

    monkeypatch.setattr(auth_api, "hash_password_async", the_owner_arrives)
    change = await client.post("/auth/password", headers=stranger["headers"], json={"current_password": PASSWORD, "new_password": "the stranger stays in"})
    owner = taken_back["response"]
    assert owner.status_code == 200 and owner.json()["user"]["has_password"] is False
    assert change.status_code in (401, 409)
    async with other_client() as again:
        for password in (PASSWORD, "the stranger stays in"):
            assert (await again.post("/auth/login", json={"email": email, "password": password})).status_code == 401
    assert (await client.get("/auth/me", headers=bearer_of(owner.json()["access_token"]))).status_code == 200


@pytest.mark.integration
async def test_renewing_an_old_hash_at_sign_in_does_not_put_an_old_password_back(client, jwt_mode, monkeypatch):
    from argon2 import PasswordHasher

    account = await register(client)
    async with get_session_factory()() as session:
        user = await session.get(User, uuid.UUID(account["user"]["id"]))
        user.password_hash = PasswordHasher(time_cost=3, memory_cost=65_536, parallelism=4).hash(PASSWORD)  # made with older settings
        await session.commit()
    hash_as_usual, changed = auth_api.hash_password_async, {}

    async def the_owner_changes_the_password(password: str) -> str:
        if not changed:  # while the racing sign-in is renewing the old hash
            changed["started"] = True
            changed["response"] = await client.post(
                "/auth/password", headers=account["headers"], json={"current_password": PASSWORD, "new_password": "a brand new secret"}
            )
        return await hash_as_usual(password)

    monkeypatch.setattr(auth_api, "hash_password_async", the_owner_changes_the_password)
    async with other_client() as old_device:
        await old_device.post("/auth/login", json={"email": account["email"], "password": PASSWORD})
    assert changed["response"].status_code == 200
    async with other_client() as later:
        assert (await later.post("/auth/login", json={"email": account["email"], "password": PASSWORD})).status_code == 401
        assert (await later.post("/auth/login", json={"email": account["email"], "password": "a brand new secret"})).status_code == 200


@pytest.mark.integration
async def test_a_copied_cookie_that_is_noticed_ends_everything_the_copy_obtained(client, jwt_mode, monkeypatch):
    """Found in the third review: the reuse was noticed, but the thief's access token went on working -
    long enough to delete the account."""
    monkeypatch.setattr(auth_api, "REFRESH_RACE_SECONDS", -1)  # as if the original came back much later
    account = await register(client)
    original = client.cookies.get("veriagent_refresh")
    async with other_client() as thief:
        stolen = await thief.post("/auth/refresh", cookies={"veriagent_refresh": original})
        assert stolen.status_code == 200
        assert (await client.post("/auth/refresh")).status_code == 401  # the owner's cookie is spent: the copy is noticed
        assert (await thief.get("/auth/me", headers=bearer_of(stolen.json()["access_token"]))).status_code == 401
        assert (await thief.post("/auth/refresh")).status_code == 401
        deleted = await thief.request("DELETE", "/auth/me", headers=bearer_of(stolen.json()["access_token"]), json={"confirm": "DELETE"})
        assert deleted.status_code == 401
    async with other_client() as owner:
        assert (await owner.post("/auth/login", json={"email": account["email"], "password": PASSWORD})).status_code == 200


@pytest.mark.integration
async def test_signing_out_ends_the_session_even_if_its_cookie_was_already_exchanged_elsewhere(client, jwt_mode):
    """Found in the third review: someone exchanged a copy of the cookie; the owner then signed out with the
    spent one - and the copy's successor went on working."""
    account = await register(client)
    original = client.cookies.get("veriagent_refresh")
    async with other_client() as thief:
        assert (await thief.post("/auth/refresh", cookies={"veriagent_refresh": original})).status_code == 200
        assert (await client.post("/auth/logout", headers=account["headers"])).status_code == 204
        assert (await thief.post("/auth/refresh")).status_code == 401
    async with get_session_factory()() as session:
        live = (await session.execute(select(RefreshToken).where(
            RefreshToken.user_id == uuid.UUID(account["user"]["id"]), RefreshToken.revoked_at.is_(None)
        ))).scalars().all()
    assert live == []


@pytest.mark.integration
async def test_another_google_identity_does_not_take_over_a_linked_account(client, jwt_mode, google):
    email = new_email()
    first = await client.post("/auth/google", json={"credential": google_credential(google, email=email, sub="g-first")})
    assert first.status_code == 200
    async with other_client() as other:
        second = await other.post("/auth/google", json={"credential": google_credential(google, email=email, sub="g-second")})
        assert second.status_code == 401 and "different Google account" in second.json()["error"]["message"]
    assert (await client.get("/auth/me", headers=bearer_of(first.json()["access_token"]))).status_code == 200


@pytest.mark.integration
async def test_a_sign_in_turned_away_by_a_busy_server_is_not_a_wrong_password(client, jwt_mode, monkeypatch):
    """Found in the third review: a flood of sign-ins for other accounts got an uninvolved user locked out for
    "too many failed attempts" - with the right password every time."""
    from app.errors import TooManyRequests

    account = await register(client)
    verify_as_usual = auth_api.verify_password_async

    async def busy(password, password_hash):
        raise TooManyRequests("The server is busy with other sign-ins. Try again in a few seconds.")

    monkeypatch.setattr(auth_api, "verify_password_async", busy)
    for _ in range(30):
        assert (await client.post("/auth/login", json={"email": account["email"], "password": PASSWORD})).status_code == 429
        auth_api.attempts_by_address.reset()  # the per-address limit is not what is being tested
    monkeypatch.setattr(auth_api, "verify_password_async", verify_as_usual)
    assert (await client.post("/auth/login", json={"email": account["email"], "password": PASSWORD})).status_code == 200


@pytest.mark.integration
async def test_an_invalid_request_is_answered_without_repeating_what_was_sent(client, jwt_mode):
    """FastAPI's own answer quotes the input - a password that is too long, for example - and fails with 500
    when the input cannot be written as text at all."""
    secret = "S3cret-" + "x" * 300
    too_long = await client.post("/auth/login", json={"email": "someone@example.com", "password": secret})
    assert too_long.status_code == 422 and secret not in too_long.text and "S3cret" not in too_long.text
    assert too_long.json()["error"]["code"] == "invalid_request" and "password" in too_long.json()["error"]["message"]
    broken = await client.post(
        "/auth/login", content=b'{"email": "someone@example.com", "password": "\\ud800"}', headers={"content-type": "application/json"}
    )
    assert broken.status_code == 422
    named = await client.post("/auth/register", json={"email": new_email(), "password": PASSWORD, "name": "a\x00b"})
    assert named.status_code == 201 and named.json()["user"]["name"] == "ab"
    for path in ("/auth/refresh", "/auth/logout"):
        assert (await client.post(path, headers={"Origin": "http://["})).status_code == 403
