"""One DATABASE_URL for two drivers, and connections that survive a database that goes to sleep."""

from __future__ import annotations

import os
import ssl

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
    assert set(target.connect_args) == {"ssl"} and verifies(target.connect_args["ssl"], hostname=True)
    assert target.tls == "verified" and not target.is_local
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
    assert parse_database_url("postgresql://u:p@h/d?sslmode=disable").connect_args == {"ssl": False}
    assert parse_database_url("postgresql://u:p@h/d?sslmode=prefer").connect_args == {}
    full = parse_database_url("postgresql://u:p@h/d?sslmode=verify-full&sslrootcert=system")
    assert verifies(full.connect_args["ssl"], hostname=True) and full.tls == "verified" and "sslrootcert" not in full.async_url
    ca_only = parse_database_url("postgresql://u:p@h/d?sslmode=verify-ca")
    assert verifies(ca_only.connect_args["ssl"], hostname=False) and ca_only.tls == "verified"
    # Written the asyncpg way: psycopg gets the libpq spelling.
    for written, libpq in (("ssl=require", "sslmode=require"), ("ssl=true", "sslmode=require"), ("ssl=verify-full", "sslmode=verify-full")):
        target = parse_database_url(f"postgresql+asyncpg://u:p@h/d?{written}")
        assert target.psycopg_url == f"postgresql://u:p@h/d?{libpq}" and target.async_url == "postgresql+asyncpg://u:p@h/d"
    assert parse_database_url("postgresql://u:p@h/d?ssl=true").connect_args == {"ssl": "require"}
    with pytest.raises(ValueError, match="must start with postgresql://"):
        parse_database_url("mysql://u:p@h/d")


def test_the_translation_never_asks_for_less_protection_than_the_string():
    """Found in review: "disable" together with channel binding connected in plain text through asyncpg."""
    for query in ("sslmode=disable&channel_binding=require", "channel_binding=require", "sslmode=prefer&channel_binding=require"):
        target = parse_database_url(f"postgresql://u:p@db.example.com/d?{query}")
        assert target.tls == "verified" and verifies(target.connect_args["ssl"], hostname=True), query
    assert parse_database_url("postgresql://u:p@db.example.com/d?sslmode=require").tls == "unverified"
    assert parse_database_url("postgresql://u:p@db.example.com/d").tls == "default"


def test_words_only_libpq_knows_never_reach_asyncpg():
    """Found in review: keepalives=1 reached asyncpg and made every connection fail with a TypeError."""
    url = "postgresql://u:p@h/d?sslmode=require&keepalives=1&options=-c%20statement_timeout%3D5000&connect_timeout=10&application_name=veri"
    target = parse_database_url(url)
    assert target.async_url == "postgresql+asyncpg://u:p@h/d"
    assert target.connect_args == {"ssl": "require", "timeout": 10.0, "server_settings": {"application_name": "veri"}}
    # libpq gets them all, and a space stays %20 (libpq does not read "+" as a space).
    assert target.psycopg_url == url


def test_special_characters_in_the_password():
    encoded = parse_database_url("postgresql://owner:p%40ss%3Aw%2Frd%25%3F@db.example.com:5433/d")
    assert encoded.host == "db.example.com" and "p%40ss%3Aw%2Frd%25%3F@db.example.com:5433" in encoded.async_url
    assert parse_database_url("postgresql://u:p@[2001:db8::1]:5432/d").host == "2001:db8::1"
    # Not encoded: the address cannot be told from the password. The message explains, and repeats nothing.
    for broken in ("postgresql://owner:pa/ss-w0rd@db.example.com/d", "postgresql://owner:pa?ss-w0rd@db.example.com/d", "postgresql://"):
        with pytest.raises(ValueError) as raised:
            parse_database_url(broken)
        assert "percent-encoded" in str(raised.value) and "ss-w0rd" not in str(raised.value)
        # The library's own message ("Port could not be cast ... 'pa'") quotes the password: it must not be attached.
        assert raised.value.__cause__ is None and (raised.value.__context__ is None or raised.value.__suppress_context__)


def test_which_hosts_count_as_local():
    assert parse_database_url("postgresql://u:p@localhost:5432/d").is_local
    assert parse_database_url("postgresql://u:p@db:5432/d").is_local  # a Docker Compose service
    assert parse_database_url("postgresql://u:p@127.0.0.1/d").is_local
    assert not parse_database_url("postgresql://u:p@10.0.0.5/d").is_local
    assert not parse_database_url(NEON).is_local


def verifies(context: object, hostname: bool) -> bool:
    return isinstance(context, ssl.SSLContext) and context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname is hostname


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
async def test_the_string_a_hosted_database_hands_out_really_connects(database):
    """Found in review: the translated value was only compared, never used. With asyncpg's own
    "verify-full" the connection failed before it reached the network, for want of ~/.postgresql/root.crt."""
    probe = parse_database_url(local_url("sslmode=require"))
    engine = create_async_engine(probe.async_url, connect_args={"ssl": ssl.create_default_context()})
    try:
        async with engine.connect():
            pass
    except Exception:  # noqa: BLE001
        pytest.skip("this PostgreSQL has no certificate that the system trusts for its host name")
    finally:
        await engine.dispose()

    target = parse_database_url(local_url("sslmode=require&channel_binding=require"))
    assert target.tls == "verified"
    engine = create_async_engine(target.async_url, connect_args=target.connect_args)
    try:
        async with engine.connect() as conn:
            assert (await conn.execute(text("SELECT ssl FROM pg_stat_ssl WHERE pid = pg_backend_pid()"))).scalar() is True
    finally:
        await engine.dispose()


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
