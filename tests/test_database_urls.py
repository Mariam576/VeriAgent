"""One DATABASE_URL for two drivers, and connections that survive a database that goes to sleep."""

from __future__ import annotations

import os

import psycopg
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.database.urls import parse_database_url

NEON = "postgresql://owner:s3cret@ep-cool-name-123456.eu-central-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"


def test_a_local_url_is_passed_on_unchanged():
    target = parse_database_url("postgresql://postgres:postgres@localhost:5432/veriagent")
    assert target.async_url == "postgresql+asyncpg://postgres:postgres@localhost:5432/veriagent"
    assert target.psycopg_url == "postgresql://postgres:postgres@localhost:5432/veriagent"
    assert target.connect_args == {} and not target.pooled and target.psycopg_prepare_threshold == 0
    assert parse_database_url("postgres://u:p@db/x").async_url == "postgresql+asyncpg://u:p@db/x"  # the short scheme


def test_a_neon_url_works_for_both_drivers():
    target = parse_database_url(NEON)
    # asyncpg: no libpq-only parameters in the URL; TLS with a verified certificate instead of channel binding.
    assert target.async_url == "postgresql+asyncpg://owner:s3cret@ep-cool-name-123456.eu-central-1.aws.neon.tech/neondb"
    assert target.connect_args == {"ssl": "verify-full"}
    # psycopg: exactly what the provider wrote.
    assert target.psycopg_url == NEON
    assert not target.pooled


def test_a_pooled_address_switches_prepared_statements_off():
    pooled = parse_database_url(NEON.replace("ep-cool-name-123456.", "ep-cool-name-123456-pooler."))
    assert pooled.pooled and pooled.psycopg_prepare_threshold is None
    assert pooled.connect_args["statement_cache_size"] == 0 and pooled.async_url.endswith("?prepared_statement_cache_size=0")
    first, second = pooled.connect_args["prepared_statement_name_func"](), pooled.connect_args["prepared_statement_name_func"]()
    assert first != second
    assert parse_database_url("postgresql://u:p@db.example.com:6543/postgres").pooled  # PgBouncer's usual port
    flagged = parse_database_url("postgresql://u:p@db.example.com/postgres?pgbouncer=true")
    assert flagged.pooled and "pgbouncer" not in flagged.async_url and "pgbouncer" not in flagged.psycopg_url


def test_ssl_modes_are_translated_faithfully():
    assert parse_database_url("postgresql://u:p@h/d?sslmode=require").connect_args == {"ssl": "require"}
    assert parse_database_url("postgresql://u:p@h/d?sslmode=verify-full&sslrootcert=system").connect_args == {"ssl": "verify-full"}
    assert parse_database_url("postgresql://u:p@h/d?sslmode=disable").connect_args == {"ssl": False}
    assert parse_database_url("postgresql://u:p@h/d?sslmode=prefer").connect_args == {}
    # Written the asyncpg way: psycopg gets the libpq spelling.
    written_for_asyncpg = parse_database_url("postgresql+asyncpg://u:p@h/d?ssl=require")
    assert written_for_asyncpg.connect_args == {"ssl": "require"} and written_for_asyncpg.psycopg_url == "postgresql://u:p@h/d?sslmode=require"
    assert "sslrootcert" not in parse_database_url("postgresql://u:p@h/d?sslmode=verify-full&sslrootcert=system").async_url
    with pytest.raises(ValueError, match="must start with postgresql://"):
        parse_database_url("mysql://u:p@h/d")


def local_url(extra: str) -> str:
    base = os.environ["DATABASE_URL"]
    return f"{base}{'&' if '?' in base else '?'}{extra}"


@pytest.mark.integration
async def test_both_drivers_connect_with_tls_from_one_url(database):
    target = parse_database_url(local_url("sslmode=require"))
    engine = create_async_engine(target.async_url, connect_args=target.connect_args)
    try:
        async with engine.connect() as conn:
            assert (await conn.execute(text("SELECT ssl FROM pg_stat_ssl WHERE pid = pg_backend_pid()"))).scalar() is True
    finally:
        await engine.dispose()
    async with await psycopg.AsyncConnection.connect(target.psycopg_url) as conn:
        row = await (await conn.execute("SELECT ssl FROM pg_stat_ssl WHERE pid = pg_backend_pid()")).fetchone()
        assert row[0] is True


@pytest.mark.integration
async def test_the_settings_for_a_pooled_address_work_on_a_real_server(database):
    target = parse_database_url(local_url("pgbouncer=true"))
    engine = create_async_engine(target.async_url, connect_args=target.connect_args)
    try:
        async with engine.connect() as conn:
            for number in (1, 2, 3):  # the same statement several times: nothing may rely on a cached plan
                assert (await conn.execute(text("SELECT :n + 1"), {"n": number})).scalar() == number + 1
    finally:
        await engine.dispose()


@pytest.mark.integration
async def test_connections_closed_by_the_database_are_replaced(client, user_headers):
    """A database that scales to zero drops every idle connection. The next request must simply work."""
    from app.agents.checkpoint import get_checkpointer
    from app.database.session import get_engine

    assert (await client.get("/documents", headers=user_headers)).status_code == 200
    saver = await get_checkpointer()
    config = {"configurable": {"thread_id": "no-such-thread"}}
    assert await saver.aget_tuple(config) is None

    # Close every other connection to this database, as the provider would.
    async with get_engine().connect() as conn:
        closed = (await conn.execute(text(
            "SELECT count(pg_terminate_backend(pid)) FROM pg_stat_activity "
            "WHERE datname = current_database() AND pid <> pg_backend_pid()"
        ))).scalar()
    assert closed >= 1

    assert (await client.get("/documents", headers=user_headers)).status_code == 200  # the SQLAlchemy pool recovered
    assert await saver.aget_tuple(config) is None  # and so did the checkpointer's pool
