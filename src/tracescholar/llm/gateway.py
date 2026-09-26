"""A single typed LLM boundary backed by an OpenAI-compatible API."""

from __future__ import annotations

from typing import Protocol, TypeVar
from urllib.parse import urlsplit

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, OpenAIError
from pydantic import BaseModel, ValidationError

from tracescholar.config import Settings, get_settings


T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """The LLM failed or did not return a valid structured response."""


class LLMConfigurationError(LLMError):
    """The configured LLM endpoint is not ready to call."""


class StructuredLLM(Protocol):
    """Reusable typed output contract for Planner and later agent stages."""

    @property
    def model_name(self) -> str: ...

    def generate(
        self,
        schema: type[T],
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> T: ...


class OpenAICompatibleLLM:
    """Call a configurable Chat Completions endpoint and validate locally."""

    def __init__(self, *, settings: Settings | None = None, client: OpenAI | None = None,
                 max_retries: int = 1) -> None:
        self._settings = settings or get_settings()
        self._client = client
        self._max_retries = max_retries

    @property
    def model_name(self) -> str:
        if self._settings.llm_model is not None:
            return self._settings.llm_model.strip()
        if self._settings.llm_base_url.rstrip("/") == "https://api.openai.com/v1":
            return self._settings.openai_model.strip()
        return ""

    def _get_client(self) -> OpenAI:
        if self._client is not None:
            return self._client
        base_url = self._settings.llm_base_url.strip()
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"https", "http"} or not parsed.hostname:
            raise LLMConfigurationError("TRACESCHOLAR_LLM_BASE_URL must be an HTTP(S) URL.")
        if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise LLMConfigurationError("Remote LLM endpoints must use HTTPS to protect API keys.")
        if parsed.username or parsed.password:
            raise LLMConfigurationError("Do not embed credentials in TRACESCHOLAR_LLM_BASE_URL.")
        key = self._settings.llm_api_key
        if key is None and base_url.rstrip("/") == "https://api.openai.com/v1":
            key = self._settings.openai_api_key
        if key is None:
            raise LLMConfigurationError("TRACESCHOLAR_LLM_API_KEY is required for LLM calls.")
        self._client = OpenAI(
            api_key=key.get_secret_value(),
            base_url=base_url,
            timeout=self._settings.llm_timeout_seconds,
            max_retries=self._max_retries,
        )
        return self._client

    def generate(
        self,
        schema: type[T],
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> T:
        if not self.model_name:
            raise LLMConfigurationError("TRACESCHOLAR_LLM_MODEL must not be blank.")
        mode = self._settings.llm_structured_mode
        response_format = (
            {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__.lower(),
                    "strict": True,
                    "schema": schema.model_json_schema(),
                },
            }
            if mode == "json_schema"
            else {"type": "json_object"}
        )
        try:
            response = self._get_client().chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt + "\nReturn a JSON object only."},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=response_format,
            )
        except APITimeoutError as error:
            raise LLMError("LLM request timed out.") from error
        except APIStatusError as error:
            raise LLMError(f"LLM provider returned HTTP {error.status_code}.") from error
        except APIConnectionError as error:
            raise LLMError("LLM connection failed.") from error
        except OpenAIError as error:
            raise LLMError("LLM request failed.") from error

        if not response.choices:
            raise LLMError("LLM response contained no choices.")
        choice = response.choices[0]
        if choice.finish_reason != "stop":
            raise LLMError(f"LLM response did not finish normally ({choice.finish_reason}).")
        if choice.message.refusal:
            raise LLMError("LLM refused the structured request.")
        content = choice.message.content
        if not isinstance(content, str) or not content.strip():
            raise LLMError("LLM response contained no structured content.")
        try:
            return schema.model_validate_json(content)
        except ValidationError as error:
            locations = ", ".join(
                f"{'.'.join(map(str, item['loc'])) or '<root>'}:{item['type']}:"
                f"{str(item['msg'])[:120]}"
                for item in error.errors(include_input=False)[:3]
            )
            raise LLMError(f"LLM response failed structured validation ({locations}).") from error
