"""Provider-neutral contract and independent OpenAI-compatible cloud adapter."""

from __future__ import annotations

import hashlib
from importlib.metadata import version
from typing import Protocol, Sequence
from urllib.parse import urlsplit

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, OpenAIError

from tracescholar.config import Settings, get_settings


class EmbeddingError(RuntimeError):
    def __init__(self, code: str, detail: str):
        self.code = code
        super().__init__(detail)


class EmbeddingEncoder(Protocol):
    provider: str
    model_name: str
    model_revision: str
    source_revision: str
    endpoint_url: str | None
    encoder_version: str
    dimensions: int

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, query: str) -> list[float]: ...


class OpenAICompatibleEmbeddings:
    """Call a separate cloud Embeddings API without using the chat-model key."""

    provider = "openai-compatible"

    def __init__(self, *, settings: Settings | None = None, client: OpenAI | None = None):
        active = settings or get_settings()
        base = (active.embedding_base_url or "").strip().rstrip("/")
        model = (active.embedding_model or "").strip()
        if not base or not model or active.embedding_dimensions is None:
            raise EmbeddingError(
                "not_configured", "TRACESCHOLAR_EMBEDDING_BASE_URL, MODEL, and DIMENSIONS are required."
            )
        parsed = urlsplit(base)
        if parsed.scheme not in {"https", "http"} or not parsed.hostname:
            raise EmbeddingError("invalid_url", "Embedding Base URL must be an HTTP(S) URL.")
        if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise EmbeddingError("invalid_url", "Remote Embedding endpoints must use HTTPS.")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise EmbeddingError("invalid_url", "Embedding Base URL must not contain credentials or query parameters.")
        if active.embedding_api_key is None and client is None:
            raise EmbeddingError("not_configured", "TRACESCHOLAR_EMBEDDING_API_KEY is required.")
        self.endpoint_url = base
        self.model_name = model
        self.dimensions = active.embedding_dimensions
        self.source_revision = (active.embedding_model_version or model).strip()
        if not self.source_revision:
            raise EmbeddingError("not_configured", "Embedding model version must not be blank.")
        # Namespace same-named models hosted at different endpoints.
        self.model_revision = hashlib.sha256(
            f"{base}|{model}|{self.source_revision}|{self.dimensions}".encode("utf-8")
        ).hexdigest()
        self.encoder_version = f"openai-{version('openai')}"
        self._client = client or OpenAI(
            api_key=active.embedding_api_key.get_secret_value(),
            base_url=base, timeout=active.embedding_timeout_seconds,
            max_retries=2,
        )

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts or any(not text.strip() for text in texts):
            raise EmbeddingError("empty_input", "Embedding inputs must contain nonempty text.")
        try:
            response = self._client.embeddings.create(
                model=self.model_name, input=list(texts),
                dimensions=self.dimensions, encoding_format="float",
            )
        except APITimeoutError as error:
            raise EmbeddingError("timeout", "Embedding request timed out.") from error
        except APIConnectionError as error:
            raise EmbeddingError("connection_failed", "Embedding service connection failed.") from error
        except APIStatusError as error:
            code = "rate_limited" if error.status_code == 429 else "provider_error"
            raise EmbeddingError(code, f"Embedding service returned HTTP {error.status_code}.") from error
        except OpenAIError as error:
            raise EmbeddingError("provider_error", "Embedding service request failed.") from error
        rows = sorted(response.data, key=lambda item: item.index)
        if len(rows) != len(texts) or [item.index for item in rows] != list(range(len(texts))):
            raise EmbeddingError("invalid_response", "Embedding service returned missing or duplicate indices.")
        return [list(item.embedding) for item in rows]

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, query: str) -> list[float]:
        return self._embed([query])[0]
