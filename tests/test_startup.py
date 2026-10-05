"""The checks made before the application starts, and who counts as "the caller"."""

from __future__ import annotations

import logging

import pytest
from starlette.requests import Request

from app import startup
from app.config import Settings, get_settings
from app.security.rate_limit import client_address
from app.startup import ConfigurationError, check_configuration, configuration_problems, configuration_warnings


def settings(**values: object) -> Settings:
    """Settings built from the given values only (the .env file of the project is not read)."""
    base = {
        "LLM_PROVIDER": "mock", "EMBEDDING_PROVIDER": "hash", "AUTH_MODE": "jwt",
        "DATABASE_URL": "postgresql://postgres:postgres@localhost:5432/veriagent",
    }
    return Settings(_env_file=None, **{**base, **values})


def request_with(headers: dict[str, str], peer: str = "10.0.0.7") -> Request:
    raw = [(name.lower().encode(), value.encode()) for name, value in headers.items()]
    return Request({"type": "http", "method": "GET", "path": "/", "headers": raw, "client": (peer, 5000)})


# ------------------------------------------------------------------ problems
def test_sensible_settings_have_no_problems():
    assert configuration_problems(settings()) == []
    check_configuration(settings())  # does not raise


def test_local_embeddings_without_the_library_stop_the_start(monkeypatch):
    monkeypatch.setattr(startup, "_installed", lambda module: False)
    problems = configuration_problems(settings(EMBEDDING_PROVIDER="local"))
    assert len(problems) == 1
    assert "EMBEDDING_PROVIDER=gemini" in problems[0] and "LOCAL_EMBEDDINGS=true" in problems[0]
    with pytest.raises(ConfigurationError, match="sentence-transformers"):
        check_configuration(settings(EMBEDDING_PROVIDER="local"))


def test_local_embeddings_with_the_library_are_fine(monkeypatch):
    monkeypatch.setattr(startup, "_installed", lambda module: True)
    assert configuration_problems(settings(EMBEDDING_PROVIDER="local")) == []


def test_no_sign_in_on_a_public_server_stops_the_start():
    with pytest.raises(ConfigurationError, match="AUTH_MODE=jwt"):
        check_configuration(settings(AUTH_MODE="dev", TRUST_PROXY_HEADERS=True))
    # On your own PC development mode stays allowed.
    assert configuration_problems(settings(AUTH_MODE="dev")) == []


def test_a_signing_key_that_is_too_short_stops_the_start():
    """Found in review: the server started, and then every sign-in answered "500 Something went wrong"."""
    with pytest.raises(ConfigurationError, match="JWT_SECRET must have at least 32 characters"):
        check_configuration(settings(JWT_SECRET="my-secret"))
    assert configuration_problems(settings(JWT_SECRET="x" * 32)) == []
    assert configuration_problems(settings(JWT_SECRET="")) == []  # empty = VeriAgent creates one itself


def test_an_unreadable_database_address_stops_the_start_without_repeating_it():
    with pytest.raises(ConfigurationError) as raised:
        check_configuration(settings(DATABASE_URL="postgresql://owner:pa/ss-w0rd@db.example.com/d"))
    assert "percent-encoded" in str(raised.value) and "ss-w0rd" not in str(raised.value)


def test_a_remote_database_without_a_checked_certificate_is_pointed_out():
    remote = "postgresql://u:p@db.example.com/d"
    assert any("certificate is not checked" in warning for warning in configuration_warnings(settings(DATABASE_URL=remote + "?sslmode=require")))
    assert any("certificate is not checked" in warning for warning in configuration_warnings(settings(DATABASE_URL=remote)))
    for checked in ("?sslmode=verify-full", "?sslmode=require&channel_binding=require"):
        assert configuration_warnings(settings(DATABASE_URL=remote + checked)) == []
    # The database of Docker Compose or of your own PC is not "another machine".
    assert configuration_warnings(settings(DATABASE_URL="postgresql://postgres:postgres@db:5432/veriagent")) == []


# ------------------------------------------------------------------ warnings
def test_a_missing_key_is_a_warning_not_a_refusal(caplog):
    chosen = settings(LLM_PROVIDER="gemini", GEMINI_API_KEY="")
    assert configuration_problems(chosen) == []
    with caplog.at_level(logging.WARNING, logger="app.startup"):
        check_configuration(chosen)
    assert "GEMINI_API_KEY is empty" in caplog.text
    assert not configuration_warnings(settings(LLM_PROVIDER="gemini", GEMINI_API_KEY="a-key"))


def test_a_reviewer_from_another_provider_needs_its_key_too():
    warnings = configuration_warnings(settings(REVIEWER_LLM_PROVIDER="grok"))
    assert any("GROK_API_KEY" in warning for warning in warnings)


def test_an_open_public_server_is_pointed_out():
    public = {"TRUST_PROXY_HEADERS": True, "LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "a-key"}
    warnings = configuration_warnings(settings(**public))
    assert any("SIGNUP_CODE" in warning for warning in warnings)
    assert any("MAX_INVESTIGATIONS_PER_DAY" in warning for warning in warnings)
    closed = settings(**public, SIGNUP_CODE="let-me-in", MAX_INVESTIGATIONS_PER_DAY=25)
    assert configuration_warnings(closed) == []
    assert configuration_warnings(settings(**public, ALLOW_REGISTRATION=False, MAX_INVESTIGATIONS_PER_DAY=25)) == []


def test_incomplete_github_settings_are_pointed_out():
    assert any("GITHUB_TOKEN" in warning for warning in configuration_warnings(settings(ACTION_SERVER="github")))
    complete = settings(ACTION_SERVER="github", GITHUB_TOKEN="a-token", GITHUB_REPOSITORY="someone/project")
    assert configuration_warnings(complete) == []
    # "GITHUB_TOKEN=" in the .env file means "not set", like the other keys.
    assert settings(ACTION_SERVER="github", GITHUB_TOKEN="", GITHUB_REPOSITORY="someone/project").GITHUB_TOKEN is None


def test_no_message_contains_a_secret(caplog):
    chosen = settings(
        TRUST_PROXY_HEADERS=True, AUTH_MODE="dev", LLM_PROVIDER="gemini",
        GEMINI_API_KEY="very-secret-key", SIGNUP_CODE="very-secret-code", GITHUB_TOKEN="very-secret-token", ACTION_SERVER="github",
    )
    with caplog.at_level(logging.WARNING, logger="app.startup"), pytest.raises(ConfigurationError) as raised:
        check_configuration(chosen)
    assert "very-secret" not in caplog.text + str(raised.value)


# ------------------------------------------------------------ client address
def test_without_a_trusted_proxy_headers_are_ignored(monkeypatch):
    monkeypatch.setattr(get_settings(), "TRUST_PROXY_HEADERS", False)
    request = request_with({"X-Forwarded-For": "203.0.113.9", "CF-Connecting-IP": "203.0.113.9"})
    assert client_address(request) == "10.0.0.7"


def test_behind_a_proxy_the_forwarded_address_counts(monkeypatch):
    monkeypatch.setattr(get_settings(), "TRUST_PROXY_HEADERS", True)
    monkeypatch.setattr(get_settings(), "CLIENT_ADDRESS_HEADER", "x-forwarded-for")
    assert client_address(request_with({"X-Forwarded-For": "203.0.113.9, 10.1.1.1"})) == "203.0.113.9"
    assert client_address(request_with({})) == "10.0.0.7"


def test_a_header_the_proxy_overwrites_beats_the_forwarded_list(monkeypatch):
    monkeypatch.setattr(get_settings(), "TRUST_PROXY_HEADERS", True)
    monkeypatch.setattr(get_settings(), "CLIENT_ADDRESS_HEADER", "CF-Connecting-IP")
    # The first entry of X-Forwarded-For was written by the client; the proxy's own header was not.
    spoofed = request_with({"X-Forwarded-For": "1.2.3.4, 198.51.100.20", "CF-Connecting-IP": "198.51.100.20"})
    assert client_address(spoofed) == "198.51.100.20"
    # A platform that does not send the chosen header: fall back to the list.
    assert client_address(request_with({"X-Forwarded-For": "198.51.100.20"})) == "198.51.100.20"
    monkeypatch.setattr(get_settings(), "CLIENT_ADDRESS_HEADER", "")
    assert client_address(request_with({"X-Forwarded-For": "198.51.100.20"})) == "198.51.100.20"


# ---------------------------------------------------------- the Render blueprint
def test_the_render_blueprint_only_uses_settings_that_exist(monkeypatch):
    """render.yaml cannot be tried out here, so at least its names and values are checked."""
    from pathlib import Path

    import yaml

    blueprint = yaml.safe_load((Path(__file__).parent.parent / "render.yaml").read_text(encoding="utf-8"))
    (service,) = blueprint["services"]
    assert (service["type"], service["runtime"], service["plan"]) == ("web", "docker", "free")
    assert service["healthCheckPath"] == "/health"

    entries = {entry["key"]: entry for entry in service["envVars"]}
    build_only = {"LOCAL_EMBEDDINGS"}  # read by the Dockerfile, not by the application
    assert set(entries) - build_only <= set(Settings.model_fields)

    # Secrets are asked for by Render or generated by it; none is written in the file.
    for name in ("DATABASE_URL", "GEMINI_API_KEY", "SIGNUP_CODE"):
        assert entries[name] == {"key": name, "sync": False}
    assert entries["JWT_SECRET"] == {"key": "JWT_SECRET", "generateValue": True}

    # The values form settings the application accepts, on the image without PyTorch.
    monkeypatch.setattr(startup, "_installed", lambda module: module != "sentence_transformers")
    values = {name: entry["value"] for name, entry in entries.items() if "value" in entry and name not in build_only}
    assert all(isinstance(value, str) for value in values.values())  # YAML would turn a bare false into a boolean
    chosen = settings(
        **values, GEMINI_API_KEY="a-key", SIGNUP_CODE="a-code", JWT_SECRET="x" * 44,
        DATABASE_URL="postgresql://owner:pw@ep-cool-name-123456.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require",
    )
    assert configuration_problems(chosen) == []
    assert configuration_warnings(chosen) == []
    assert entries["LOCAL_EMBEDDINGS"]["value"] == "false" and chosen.EMBEDDING_PROVIDER == "gemini"
    assert chosen.AUTH_MODE == "jwt" and chosen.TRUST_PROXY_HEADERS and not chosen.KEEP_UPLOADED_FILES


def test_every_setting_is_explained_in_the_example_file():
    """A setting that exists in the code but not in .env.example is one nobody will find."""
    import re
    from pathlib import Path

    example = (Path(__file__).parent.parent / ".env.example").read_text(encoding="utf-8")
    written = set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]+)=", example, re.MULTILINE))
    fixed_in_code = {"APP_NAME", "APP_VERSION"}
    assert set(Settings.model_fields) - written - fixed_in_code == set()
    only_for_docker_or_langsmith = {"API_PORT", "DB_PORT", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT", "LANGSMITH_TRACING"}
    assert written - set(Settings.model_fields) == only_for_docker_or_langsmith
