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
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# Parameters that only libpq understands. They are removed from the asyncpg URL.
_LIBPQ_ONLY = {
    "sslmode", "channel_binding", "sslrootcert", "sslcert", "sslkey", "sslcrl", "gssencmode", "target_session_attrs",
    "options", "connect_timeout", "application_name",
}
_SCHEMES = ("postgresql+asyncpg", "postgresql+psycopg", "postgresql", "postgres")


@dataclass(frozen=True)
class DatabaseTarget:
    async_url: str  # for SQLAlchemy + asyncpg
    connect_args: dict[str, Any] = field(default_factory=dict)  # extra arguments for asyncpg
    psycopg_url: str = ""  # for psycopg (the checkpointer)
    pooled: bool = False  # the address is a transaction pooler: no prepared statements

    @property
    def psycopg_prepare_threshold(self) -> int | None:
        """0 = prepare every statement at once (fast); None = never prepare (needed behind a pooler)."""
        return None if self.pooled else 0


def is_pooled(host: str, port: int | None, query: dict[str, str]) -> bool:
    return "-pooler." in host or port == 6543 or query.get("pgbouncer", "").lower() == "true"


def parse_database_url(url: str) -> DatabaseTarget:
    parts = urlsplit(url.strip())
    if parts.scheme not in _SCHEMES:
        raise ValueError("DATABASE_URL must start with postgresql:// (or postgres://).")
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    pooled = is_pooled(parts.hostname or "", parts.port, query)

    # ---- asyncpg
    connect_args: dict[str, Any] = {}
    mode = query.get("sslmode", "").lower()
    if mode in ("require", "verify-ca", "verify-full"):
        wants_binding = query.get("channel_binding", "").lower() == "require"
        connect_args["ssl"] = "verify-full" if wants_binding else mode
    elif mode == "disable":
        connect_args["ssl"] = False
    elif "ssl" in query:  # already written for asyncpg
        connect_args["ssl"] = query["ssl"]
    if pooled:
        connect_args["statement_cache_size"] = 0
        # Each prepared statement gets a name nobody else can have, in case the driver prepares one anyway.
        connect_args["prepared_statement_name_func"] = lambda: f"__asyncpg_{uuid.uuid4()}__"
    kept = {key: value for key, value in query.items() if key not in _LIBPQ_ONLY and key not in ("ssl", "pgbouncer")}
    if pooled:
        kept["prepared_statement_cache_size"] = "0"
    async_url = urlunsplit(("postgresql+asyncpg", parts.netloc, parts.path, urlencode(kept), ""))

    # ---- psycopg (libpq): the string as the provider wrote it, minus what only asyncpg or we understand
    libpq = {key: value for key, value in query.items() if key not in ("ssl", "pgbouncer", "prepared_statement_cache_size")}
    if "ssl" in query and "sslmode" not in libpq and query["ssl"].lower() in ("require", "verify-ca", "verify-full", "true"):
        libpq["sslmode"] = "require" if query["ssl"].lower() == "true" else query["ssl"].lower()
    psycopg_url = urlunsplit(("postgresql", parts.netloc, parts.path, urlencode(libpq), ""))
    return DatabaseTarget(async_url=async_url, connect_args=connect_args, psycopg_url=psycopg_url, pooled=pooled)
