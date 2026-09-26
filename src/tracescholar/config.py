"""Centralized application configuration for TraceScholar."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """TraceScholar settings loaded from environment variables or a .env file.

    Environment variables use the ``TRACESCHOLAR_`` prefix. For example,
    ``openai_api_key`` is read from ``TRACESCHOLAR_OPENAI_API_KEY``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="TRACESCHOLAR_",
        env_ignore_empty=True,
        case_sensitive=False,
        extra="ignore",
        frozen=True,
    )

    app_env: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    data_dir: Path = Path("data")

    # Reserved for the persistence layer.
    database_url: SecretStr | None = None

    # Reserved for OpenAI-backed workflows.
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-5.6-terra"

    # One LLM entry point. Override these for an OpenAI-compatible third party.
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: SecretStr | None = None
    llm_model: str | None = None
    llm_structured_mode: Literal["json_schema", "json_object"] = "json_schema"
    llm_timeout_seconds: float = Field(default=60.0, gt=0, le=300)
    fulltext_screening_llm_timeout_seconds: float = Field(default=180.0, gt=0, le=600)
    evidence_llm_timeout_seconds: float = Field(default=120.0, gt=0, le=600)
    synthesis_llm_timeout_seconds: float = Field(default=180.0, gt=0, le=600)

    # OpenAlex paper-source adapter.
    openalex_base_url: str = "https://api.openalex.org"
    openalex_api_key: SecretStr | None = None
    openalex_timeout_seconds: float = Field(default=30.0, gt=0, le=120)

    # Crossref's public REST API; email opts into its polite pool.
    crossref_base_url: str = "https://api.crossref.org"
    crossref_email: str | None = None
    crossref_timeout_seconds: float = Field(default=30.0, gt=0, le=120)

    # Open-access full-text acquisition. Files are stored below data_dir.
    fulltext_timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    fulltext_max_bytes: int = Field(default=20_000_000, ge=100_000, le=100_000_000)
    fulltext_unavailable_retry_hours: int = Field(default=24, ge=1, le=720)

    # Separate cloud Embeddings endpoint. Never reuse the LLM key implicitly.
    embedding_base_url: str | None = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_api_key: SecretStr | None = None
    embedding_model: str | None = "qwen3.7-text-embedding"
    embedding_model_version: str | None = None
    embedding_dimensions: int | None = Field(default=1024, ge=1, le=2000)
    embedding_batch_size: int = Field(default=20, ge=1, le=20)
    embedding_timeout_seconds: float = Field(default=60.0, gt=0, le=300)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide immutable settings instance."""
    return Settings()
