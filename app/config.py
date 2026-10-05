"""Application settings.

Every tunable value and every secret is read from environment variables (or a
local ``.env`` file). Nothing in the code base hardcodes a model name or a key.

Usage::

    from app.config import get_settings
    settings = get_settings()
    settings.RAG_TOP_K
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    APP_NAME: str = "VeriAgent"
    APP_VERSION: str = "3.0.0"

    # ------------------------------------------------------------------ LLM
    LLM_PROVIDER: Literal["gemini", "grok", "mock"] = "gemini"
    GEMINI_API_KEY: SecretStr | None = None
    GEMINI_MODEL: str = ""
    GROK_API_KEY: SecretStr | None = None
    GROK_MODEL: str = ""
    GROK_BASE_URL: str = "https://api.x.ai/v1"
    LLM_FALLBACK_PROVIDER: Literal["gemini", "grok", "mock", ""] = ""
    LLM_TIMEOUT_SECONDS: float = Field(default=60.0, gt=0)
    LLM_MAX_RETRIES: int = Field(default=2, ge=0, le=5)
    LLM_TEMPERATURE: float | None = None
    LLM_REQUESTS_PER_MINUTE: int = Field(default=0, ge=0)

    # A second model that checks the first one's reading of the documents (see app/agents/reviewer.py).
    ENABLE_CLAIM_REVIEW: bool = True
    # "" = the same provider as LLM_PROVIDER. Another provider gives a truly independent second opinion.
    REVIEWER_LLM_PROVIDER: Literal["gemini", "grok", "mock", ""] = ""
    # "" = that provider's normal model (GEMINI_MODEL / GROK_MODEL).
    REVIEWER_MODEL: str = ""

    # ------------------------------------------------------------- Database
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/veriagent"

    # ----------------------------------------------------------- Embeddings
    EMBEDDING_PROVIDER: Literal["local", "gemini", "hash"] = "local"
    LOCAL_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-001"
    EMBEDDING_DIM: int = Field(default=384, gt=0, le=2000)

    # ------------------------------------------------------------------ RAG
    CHUNK_SIZE: int = Field(default=800, ge=100, le=8000)
    CHUNK_OVERLAP: int = Field(default=120, ge=0)
    RAG_TOP_K: int = Field(default=8, ge=1, le=50)
    RERANK_TOP_K: int = Field(default=4, ge=1, le=50)
    MAX_RETRIEVED_CHUNKS: int = Field(default=12, ge=1, le=60)
    # Hybrid search: combine vector similarity with keyword matching.
    HYBRID_SEARCH: bool = True
    KEYWORD_TOP_K: int = Field(default=8, ge=1, le=50)

    # ---------------------------------------------------------------- Agent
    MAX_AGENT_ITERATIONS: int = Field(default=6, ge=1, le=20)
    ENABLE_HUMAN_APPROVAL: bool = True
    # Checkpoints of a finished investigation are deleted unless this is on (useful for debugging).
    KEEP_CHECKPOINTS: bool = False
    # A run with no progress for this long is treated as interrupted and may be resumed.
    STALE_RUN_SECONDS: int = Field(default=300, ge=0)
    # An investigation still unfinished after this many days (never decided, never resumed)
    # is closed as failed and its saved state is deleted. 0 = keep waiting forever.
    CHECKPOINT_TTL_DAYS: int = Field(default=30, ge=0)
    MAX_QUESTION_CHARS: int = Field(default=2000, ge=10)

    # ------------------------------------------------------------- Sign-in
    # jwt = people sign in (email and password, or Google) and send a bearer token.
    # dev = the X-User-Id header is trusted. For tests and experiments on your own PC only.
    AUTH_MODE: Literal["jwt", "dev"] = "jwt"
    # Signs the access tokens. Leave empty and the app creates one and keeps it in the database.
    JWT_SECRET: SecretStr | None = None
    ACCESS_TOKEN_MINUTES: int = Field(default=30, ge=1, le=1440)
    REFRESH_TOKEN_DAYS: int = Field(default=30, ge=1, le=365)
    # Google sign-in is offered when this is set (an OAuth client id from Google Cloud).
    GOOGLE_CLIENT_ID: str = ""
    ALLOW_REGISTRATION: bool = True
    # If set, a new account can only be created by someone who knows this code.
    SIGNUP_CODE: SecretStr | None = None
    # Protects your LLM quota on a public deployment. 0 = no limit.
    MAX_INVESTIGATIONS_PER_DAY: int = Field(default=0, ge=0)
    # true only behind a proxy you trust (Render, a load balancer): the client's address is
    # then read from X-Forwarded-For. Without such a proxy a client could fake that header.
    TRUST_PROXY_HEADERS: bool = False

    # ---------------------------------------------------------- Comparison
    # Compare parties, reference numbers, dates and line items between documents (one extra
    # model call per document, the first time it takes part in a comparison).
    ENABLE_DOCUMENT_COMPARISON: bool = True

    # -------------------------------------------------------------- Actions
    # Where a follow-up issue is created after a person approved it, through MCP (app/actions).
    #   ""        nothing is ever created
    #   tickets   VeriAgent's own small ticket list (a file); needs no account
    #   github    an issue in GITHUB_REPOSITORY, through GitHub's MCP server
    ACTION_SERVER: Literal["", "tickets", "github"] = ""
    TICKETS_FILE: str = "data/tickets.jsonl"
    GITHUB_MCP_URL: str = "https://api.githubcopilot.com/mcp/"
    # A personal access token that may write issues in that one repository, and nothing else.
    GITHUB_TOKEN: SecretStr | None = None
    GITHUB_REPOSITORY: str = ""  # owner/repository
    ACTION_TIMEOUT_SECONDS: float = Field(default=30.0, gt=0, le=120)

    # ------------------------------------------------------------ Forensics
    # Inspect every uploaded file for signs of editing (hidden or covered text, odd fonts,
    # file history). No model call; the result is a risk indicator, never proof.
    ENABLE_FORENSICS: bool = True

    # --------------------------------------------------------------- Memory
    ENABLE_MEMORY: bool = True
    MEMORY_TOP_K: int = Field(default=3, ge=1, le=10)
    # Notes less similar to the question than this are not recalled (cosine similarity, 0..1).
    MEMORY_MIN_SIMILARITY: float = Field(default=0.3, ge=0.0, le=1.0)

    # ----------------------------------------------------- Background work
    # How many background jobs (document processing) may run at the same time.
    MAX_BACKGROUND_JOBS: int = Field(default=2, ge=1, le=16)
    # A document still "processing" after this long is marked failed (its job was lost in a restart).
    DOCUMENT_PROCESSING_TIMEOUT_MINUTES: int = Field(default=15, ge=1)
    # How often the upkeep runs (expired sign-ins, stale saved state, stuck documents). 0 = only at startup.
    MAINTENANCE_INTERVAL_MINUTES: int = Field(default=60, ge=0)

    # -------------------------------------------------------------- Uploads
    # false = delete the uploaded file once its text, vectors and inspection result are stored.
    # Use false where the disk is not permanent (most free hosting): nothing then depends on the file.
    KEEP_UPLOADED_FILES: bool = True
    MAX_UPLOAD_MB: int = Field(default=20, ge=1, le=200)
    ALLOWED_EXTENSIONS: str = ".pdf,.docx,.txt"
    UPLOAD_DIR: str = "uploads"

    LOG_LEVEL: str = "INFO"

    # An empty string in .env (``LLM_TEMPERATURE=``) means "not set".
    @field_validator("LLM_TEMPERATURE", mode="before")
    @classmethod
    def _empty_temperature_is_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("GEMINI_API_KEY", "GROK_API_KEY", "JWT_SECRET", "SIGNUP_CODE", mode="before")
    @classmethod
    def _empty_secret_is_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("CHUNK_OVERLAP")
    @classmethod
    def _overlap_smaller_than_chunk(cls, value: int, info) -> int:
        chunk_size = info.data.get("CHUNK_SIZE", 800)
        if value >= chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return value

    # ------------------------------------------------------------ Derived
    @property
    def async_database_url(self) -> str:
        """SQLAlchemy needs the driver in the URL: postgresql+asyncpg://..."""
        url = self.DATABASE_URL
        for prefix in ("postgresql://", "postgres://"):
            if url.startswith(prefix):
                return "postgresql+asyncpg://" + url[len(prefix):]
        return url

    @property
    def allowed_extensions(self) -> set[str]:
        return {
            ext.strip().lower() if ext.strip().startswith(".") else "." + ext.strip().lower()
            for ext in self.ALLOWED_EXTENSIONS.split(",")
            if ext.strip()
        }

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_MB * 1024 * 1024

    @property
    def llm_model_name(self) -> str:
        """Model name of the active provider, for display in /health and logs."""
        return {
            "gemini": self.GEMINI_MODEL,
            "grok": self.GROK_MODEL,
            "mock": "rule-based-mock",
        }[self.LLM_PROVIDER]


@lru_cache
def get_settings() -> Settings:
    """Settings are read once and cached. Tests call ``get_settings.cache_clear()``."""
    return Settings()
