"""Password hashing and the password rules.

Concept - never store a password:
    The database keeps a *hash*: the result of a slow one-way function. To
    check a login, the entered password is hashed the same way and compared.
    Someone who steals the database gets hashes, and turning a good hash back
    into a password is impractical.

    Argon2id is the current recommendation for password hashing. It is slow
    and memory-hungry on purpose, so guessing billions of passwords is
    expensive, and it adds a random "salt" so two users with the same
    password get different hashes.
"""

from __future__ import annotations

import asyncio
import weakref

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

from app.errors import ValidationFailed

MIN_PASSWORD_CHARS = 10
MAX_PASSWORD_CHARS = 128

# Argon2id with 19 MiB of memory, 2 passes, 1 thread: one of the settings OWASP recommends.
# The library's default takes 64 MiB per hash. On a host with 512 MB, a handful of sign-in
# attempts arriving together would then use up the machine (found in review, see below).
_hasher = PasswordHasher(time_cost=2, memory_cost=19_456, parallelism=1)
# How many passwords may be hashed at the same moment. Hashing is deliberately expensive,
# and signing in needs no account, so without a limit anyone could make the server spend
# all its memory and CPU on it. Further attempts wait their turn.
MAX_CONCURRENT_HASHES = 2
# Passwords that attackers try first. A real deployment would check a breach list.
_COMMON = {
    "password", "password1", "password123", "1234567890", "12345678910", "qwertyuiop", "qwerty12345",
    "iloveyou123", "letmein123", "administrator", "welcome123", "veriagent", "veriagent123", "changeme123",
}


def validate_password(password: str, email: str = "") -> None:
    """Raise ``ValidationFailed`` with the reason when a new password is not acceptable."""
    if len(password) < MIN_PASSWORD_CHARS:
        raise ValidationFailed(f"The password must have at least {MIN_PASSWORD_CHARS} characters.")
    if len(password) > MAX_PASSWORD_CHARS:
        raise ValidationFailed(f"The password must have at most {MAX_PASSWORD_CHARS} characters.")
    lowered = password.lower()
    if lowered in _COMMON or len(set(password)) < 4:
        raise ValidationFailed("That password is too easy to guess. Choose a longer or less common one.")
    local_part = email.split("@")[0].lower() if email else ""
    if email and (lowered == email.lower() or (len(local_part) >= 4 and lowered == local_part)):
        raise ValidationFailed("The password must not be your email address.")


def hash_password(password: str) -> str:
    return _hasher.hash(password)


# Verified against when the account does not exist, so a wrong email takes as
# long as a wrong password and the response time does not reveal which it was.
_DUMMY_HASH = _hasher.hash("no-such-account-placeholder")


def verify_password(password: str, password_hash: str | None) -> bool:
    try:
        return _hasher.verify(password_hash or _DUMMY_HASH, password) and password_hash is not None
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


_turns: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = weakref.WeakKeyDictionary()


def _turn() -> asyncio.Semaphore:
    """One semaphore per event loop (the application has one; every test has its own)."""
    loop = asyncio.get_running_loop()
    if loop not in _turns:
        _turns[loop] = asyncio.Semaphore(MAX_CONCURRENT_HASHES)
    return _turns[loop]


async def hash_password_async(password: str) -> str:
    """Hash in a worker thread (it is slow CPU work), at most ``MAX_CONCURRENT_HASHES`` at a time."""
    async with _turn():
        return await asyncio.to_thread(hash_password, password)


async def verify_password_async(password: str, password_hash: str | None) -> bool:
    async with _turn():
        return await asyncio.to_thread(verify_password, password, password_hash)


def needs_rehash(password_hash: str) -> bool:
    """True when the hash was made with older settings and should be renewed at the next login."""
    return _hasher.check_needs_rehash(password_hash)
