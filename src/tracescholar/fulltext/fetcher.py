"""Bounded PDF downloader that never follows a link outside approved OA hosts."""

from __future__ import annotations

import hashlib
from contextlib import nullcontext
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

import httpx

from tracescholar.config import Settings, get_settings
from tracescholar.fulltext.locator import FullTextLocation, safe_pdf_url


class FetchError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


@dataclass(frozen=True, slots=True)
class FetchedPDF:
    content: bytes
    content_hash: str
    content_type: str | None


class DocumentFetcher:
    """Fetch one public OA PDF, rejecting paywalls, oversized or invalid bodies."""

    def __init__(self, *, settings: Settings | None = None, client: httpx.Client | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = client

    def fetch(self, location: FullTextLocation) -> FetchedPDF:
        if not safe_pdf_url(location.url):
            raise FetchError("unsafe_url", "PDF URL is not an approved HTTPS OA host.")
        url = location.url
        context = nullcontext(self.client) if self.client is not None else httpx.Client()
        try:
            with context as client:
                for _ in range(4):
                    params = None
                    if urlsplit(url).hostname == "content.openalex.org":
                        key = self.settings.openalex_api_key
                        if key is None:
                            raise FetchError("api_key_missing", "OpenAlex content needs an API key.")
                        params = {"api_key": key.get_secret_value()}
                    with client.stream(
                        "GET", url, params=params, headers={"Accept": "application/pdf"},
                        timeout=self.settings.fulltext_timeout_seconds, follow_redirects=False,
                    ) as response:
                        if 300 <= response.status_code < 400:
                            redirect = response.headers.get("Location")
                            if not redirect:
                                raise FetchError("bad_redirect", "PDF response redirected without a location.")
                            url = urljoin(url, redirect)
                            if not safe_pdf_url(url):
                                raise FetchError("unsafe_redirect", "PDF redirect left approved OA hosts.")
                            continue
                        if response.status_code in (401, 403):
                            raise FetchError("access_denied", f"OA PDF returned HTTP {response.status_code}; no paywall bypass attempted.")
                        if response.status_code == 404:
                            raise FetchError("not_found", "OA PDF URL returned HTTP 404.")
                        if response.status_code != 200:
                            raise FetchError("http_error", f"OA PDF returned HTTP {response.status_code}.")
                        length = response.headers.get("Content-Length")
                        if length and length.isdigit() and int(length) > self.settings.fulltext_max_bytes:
                            raise FetchError("too_large", "PDF exceeds the configured maximum size.")
                        body = bytearray()
                        for chunk in response.iter_bytes(chunk_size=65536):
                            body.extend(chunk)
                            if len(body) > self.settings.fulltext_max_bytes:
                                raise FetchError("too_large", "PDF exceeded the configured maximum size while streaming.")
                        media_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().casefold()
                        if media_type and media_type not in {
                            "application/pdf", "application/x-pdf", "application/octet-stream",
                            "binary/octet-stream",
                        }:
                            raise FetchError("not_pdf", f"OA URL returned {media_type}, not a PDF.")
                        if not body.startswith(b"%PDF-") or b"%%EOF" not in body[-8192:]:
                            raise FetchError("not_pdf", "OA URL returned invalid or incomplete PDF content.")
                        content = bytes(body)
                        return FetchedPDF(
                            content=content, content_hash=hashlib.sha256(content).hexdigest(),
                            content_type=media_type or None,
                        )
                raise FetchError("redirect_loop", "PDF URL redirected too many times.")
        except httpx.TimeoutException as error:
            raise FetchError("timeout", "PDF download timed out.") from error
        except httpx.RequestError as error:
            raise FetchError("request_error", "PDF download request failed.") from error
