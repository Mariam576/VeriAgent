"""One DATABASE_URL, two database drivers.

VeriAgent talks to PostgreSQL through two libraries:

* SQLAlchemy with **asyncpg**, for everything the application stores, and
* **psycopg**, used by the LangGraph checkpointer.

Hosted databases hand out one connection string written for libpq (the
library behind psycopg), for example Neon's:

    postgresql://user:password@ep-cool-name.eu-central-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require

psycopg understands all of it. asyncpg does not know ``sslmode`` or
``channel_binding``; it wants ``ssl=...`` instead. This module translates, so
the same string from the provider's dashboard works for both.

Two more things are handled here:

* A "pooled" address (Neon's ``-pooler`` host, or PgBouncer on port 6543) hands
  each transaction to whatever server connection is free. Prepared statements
  live on one server connection, so they must be switched off in that case.
* ``channel_binding=require`` protects the password exchange against a
  man-in-the-middle. asyncpg cannot do channel binding, so for asyncpg the
  server's certificate is verified instead, which gives the same protection.

The translation never asks for *less* protection than the string did. If it
cannot be read at all, the error says how to write it and does not repeat any
part of the string (it contains the password).
"""

from __future__ import annotations

import ssl
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Literal
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlsplit, urlunsplit

_SCHEMES = ("postgresql+asyncpg", "postgresql+psycopg", "postgresql", "postgres")
# Words in the query string that are not passed on to libpq (asyncpg's spelling, or our own).
_NOT_FOR_LIBPQ = ("ssl", "pgbouncer", "prepared_statement_cache_size")
_VERIFYING = ("verify-ca", "verify-full")
_UNREADABLE = (
    "DATABASE_URL could not be read. It has the form postgresql://user:password@host:port/database. "
    "If the password contains one of / ? # @ : or %, write that character percent-encoded (for example %40 for @)."
)

# How the connection is protected:
#   off         not encrypted
#   default     the driver's default: encrypted if the server offers it, never verified
#   unverified  always encrypted, but the server's certificate is not checked
#   verified    always encrypted, and the certificate is checked against trusted authorities
Tls = Literal["off", "default", "unverified", "verified"]


@dataclass(frozen=True)
class DatabaseTarget:
    async_url: str  # for SQLAlchemy + asyncpg
    connect_args: dict[str, Any] = field(default_factory=dict)  # extra arguments for asyncpg
    psycopg_url: str = ""  # for psycopg (the checkpointer)
    pooled: bool = False  # the address is a transaction pooler: no prepared statements
    host: str = ""
    tls: Tls = "default"

    @property
    def psycopg_prepare_threshold(self) -> int | None:
        """0 = prepare every statement at once (fast); None = never prepare (needed behind a pooler)."""
        return None if self.pooled else 0

    @property
    def is_local(self) -> bool:
        """This computer, or a neighbouring container (Docker Compose service names have no dot)."""
        return self.host in ("localhost", "127.0.0.1", "::1", "") or "." not in self.host and ":" not in self.host


def is_pooled(host: str, port: int | None, query: dict[str, str]) -> bool:
    return "-pooler." in host or port == 6543 or query.get("pgbouncer", "").lower() == "true"


@lru_cache
def _verifying_context(check_hostname: bool, cafile: str | None) -> ssl.SSLContext:
    """TLS that checks the server's certificate against the authorities this computer trusts.

    asyncpg also accepts the *word* "verify-full", but then it insists on a
    certificate file in ``~/.postgresql/root.crt`` and never looks at the
    system's list of authorities - the list every hosted database's certificate
    is signed against. A context built here uses that list.
    """
    context = ssl.create_default_context(cafile=cafile)
    context.check_hostname = check_hostname
    return context


def _tls_wanted(query: dict[str, str]) -> tuple[Tls, bool]:
    """What the connection string asks for: (protection, also check the host name)."""
    mode = (query.get("sslmode") or "").lower()
    if not mode and "ssl" in query:  # written the asyncpg way
        mode = {"true": "require", "1": "require", "false": "disable", "0": "disable"}.get(query["ssl"].lower(), query["ssl"].lower())
    if query.get("channel_binding", "").lower() == "require":
        # Channel binding protects the password exchange against a man in the middle. asyncpg
        # cannot do it, so the certificate is verified instead - whatever sslmode says, because
        # libpq too refuses to connect unprotected when channel binding is required.
        return "verified", mode != "verify-ca"
    if mode in _VERIFYING:
        return "verified", mode == "verify-full"
    if mode == "require":
        if query.get("sslrootcert"):
            # libpq's own rule: "require" together with a certificate file means "check against that file".
            return "verified", False
        return "unverified", False
    if mode == "disable":
        return "off", False
    return "default", False


def _netloc(netloc: str) -> str:
    """The "user:password@host:port" part with user and password percent-encoded once, properly.

    A password with a space or a lone "%" is accepted by one driver and refused
    by the other - and the one that refuses quotes the password in its error
    message, which ends up in the log (found in review). Writing both parts in
    the one spelling every driver reads the same way avoids both.
    """
    credentials, at, address = netloc.rpartition("@")
    if not at:
        return netloc
    user, colon, password = credentials.partition(":")
    encoded = quote(unquote(user), safe="")
    if colon:
        encoded += ":" + quote(unquote(password), safe="")
    return f"{encoded}@{address}"


def parse_database_url(url: str) -> DatabaseTarget:
    try:
        parts = urlsplit(url.strip())
        host, port = parts.hostname or "", parts.port
    except ValueError:
        raise ValueError(_UNREADABLE) from None  # "from None": the original message quotes part of the password
    if parts.scheme not in _SCHEMES:
        raise ValueError("DATABASE_URL must start with postgresql:// (or postgres://).")
    if not host:
        raise ValueError(_UNREADABLE)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    pooled = is_pooled(host, port, query)
    tls, check_hostname = _tls_wanted(query)

    # ---- asyncpg: nothing from the query string is passed through as it is, because asyncpg
    # rejects every word it does not know. What matters is translated into typed arguments.
    connect_args: dict[str, Any] = {}
    if tls == "verified":
        cafile = query.get("sslrootcert") or None
        try:
            connect_args["ssl"] = _verifying_context(check_hostname, None if cafile == "system" else cafile)
        except (OSError, ssl.SSLError):
            raise ValueError(
                "DATABASE_URL names a certificate file (sslrootcert) that cannot be read on this computer. "
                "Leave sslrootcert out to use the certificate authorities the system trusts."
            ) from None
    elif tls == "unverified":
        connect_args["ssl"] = "require"
    elif tls == "off":
        connect_args["ssl"] = False
    if query.get("connect_timeout", "").isdigit() and int(query["connect_timeout"]) > 0:
        connect_args["timeout"] = float(query["connect_timeout"])
    if query.get("application_name"):
        connect_args["server_settings"] = {"application_name": query["application_name"]}
    for_asyncpg: dict[str, str] = {}
    if pooled:
        connect_args["statement_cache_size"] = 0
        # Each prepared statement gets a name nobody else can have, in case the driver prepares one anyway.
        connect_args["prepared_statement_name_func"] = lambda: f"__asyncpg_{uuid.uuid4()}__"
        for_asyncpg["prepared_statement_cache_size"] = "0"
    netloc = _netloc(parts.netloc)
    async_url = urlunsplit(("postgresql+asyncpg", netloc, parts.path, urlencode(for_asyncpg), ""))

    # ---- psycopg (libpq): the string as the provider wrote it, minus what only asyncpg or we understand
    libpq = {key: value for key, value in query.items() if key not in _NOT_FOR_LIBPQ}
    if "sslmode" not in libpq and "ssl" in query:
        libpq["sslmode"] = {"unverified": "require", "off": "disable"}.get(tls, query["ssl"].lower() if tls == "verified" else "prefer")
    # quote_via=quote: libpq reads %20 as a space but does not read "+" as one.
    psycopg_url = urlunsplit(("postgresql", netloc, parts.path, urlencode(libpq, quote_via=quote), ""))
    return DatabaseTarget(async_url=async_url, connect_args=connect_args, psycopg_url=psycopg_url, pooled=pooled, host=host, tls=tls)
