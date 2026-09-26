"""Crossref REST API keyword-search adapter."""

from __future__ import annotations

import re
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any, Callable

import httpx

from tracescholar.config import Settings, get_settings
from tracescholar.sources.base import PaperHit, PaperMetadata, SearchBatch, SearchScope, SourceError


_DOI = re.compile(r"10\.\d{4,9}/\S+", re.IGNORECASE)


def _string(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _first_string(value: Any) -> str | None:
    if isinstance(value, list):
        return next((text for item in value if (text := _string(item))), None)
    return _string(value)


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _plain_text(value: Any) -> str | None:
    raw = _string(value)
    if raw is None:
        return None
    extractor = _TextExtractor()
    extractor.feed(raw)
    return " ".join(" ".join(extractor.parts).split()) or None


def _published_year(item: dict[str, Any]) -> int | None:
    for field in ("published", "issued"):
        date = item.get(field)
        if not isinstance(date, dict):
            continue
        parts = date.get("date-parts")
        if not isinstance(parts, list) or not parts or not isinstance(parts[0], list) or not parts[0]:
            continue
        year = parts[0][0]
        if isinstance(year, int) and not isinstance(year, bool) and 1000 <= year <= 9999:
            return year
    return None


def _authors(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    names: list[str] = []
    for author in value:
        if not isinstance(author, dict):
            continue
        name = _string(author.get("name"))
        if name is None:
            name = " ".join(
                part for key in ("given", "family") if (part := _string(author.get(key)))
            )
        if name:
            names.append(name)
    return tuple(names)


def _normalize_item(item: Any) -> PaperHit | None:
    """Discard malformed rows and expose only shared paper metadata."""
    if not isinstance(item, dict):
        return None
    doi = _string(item.get("DOI"))
    title = _plain_text(_first_string(item.get("title")))
    if doi is None or title is None or not _DOI.fullmatch(doi):
        return None
    source_url = _string(item.get("URL")) or f"https://doi.org/{doi}"
    return PaperHit(
        paper=PaperMetadata(
            title=title,
            doi=doi,
            year=_published_year(item),
            venue=_plain_text(_first_string(item.get("container-title"))),
            authors=_authors(item.get("author")),
            abstract=_plain_text(item.get("abstract")),
            language=_string(item.get("language")),
        ),
        source_record_id=doi.casefold(),
        source_url=source_url,
    )


class CrossrefSource:
    """Search Crossref works through the provider-neutral PaperSource contract."""

    name = "crossref"

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        client: httpx.Client | None = None,
        max_retries: int = 1,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not 0 <= max_retries <= 3:
            raise ValueError("max_retries must be between 0 and 3.")
        self._settings = settings or get_settings()
        self._client = client
        self._max_retries = max_retries
        self._sleep = sleep

    def search(
        self, query: str, *, limit: int = 20, scope: SearchScope | None = None
    ) -> SearchBatch:
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("Search query must not be blank.")
        if not 1 <= limit <= 100:
            raise ValueError("Crossref limit must be between 1 and 100.")

        params: dict[str, str | int] = {
            "query.bibliographic": clean_query,
            "rows": limit,
        }
        filters: list[str] = []
        if scope is not None:
            if scope.year_from is not None:
                filters.append(f"from-pub-date:{scope.year_from}-01-01")
            if scope.year_to is not None:
                filters.append(f"until-pub-date:{scope.year_to}-12-31")
        if filters:
            params["filter"] = ",".join(filters)
        email = _string(self._settings.crossref_email)
        if email:
            params["mailto"] = email
        agent = "TraceScholar/0.1 (research metadata client"
        if email:
            agent += f"; mailto:{email}"
        headers = {"User-Agent": agent + ")"}
        url = f"{self._settings.crossref_base_url.rstrip('/')}/works"

        client_context = nullcontext(self._client) if self._client is not None else httpx.Client()
        try:
            with client_context as client:
                for attempt in range(self._max_retries + 1):
                    response = client.get(
                        url,
                        params=params,
                        headers=headers,
                        timeout=self._settings.crossref_timeout_seconds,
                    )
                    if response.status_code in {429, 500, 502, 503, 504} and attempt < self._max_retries:
                        retry_after = response.headers.get("Retry-After")
                        delay = float(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
                        self._sleep(min(delay, 45.0))
                        continue
                    break
            response.raise_for_status()
        except httpx.TimeoutException as error:
            raise SourceError("Crossref request timed out.") from error
        except httpx.HTTPStatusError as error:
            raise SourceError(f"Crossref returned HTTP {error.response.status_code}.") from error
        except httpx.RequestError as error:
            raise SourceError("Crossref request failed.") from error

        try:
            payload = response.json()
        except ValueError as error:
            raise SourceError("Crossref returned invalid JSON.") from error
        message = payload.get("message") if isinstance(payload, dict) else None
        if not isinstance(message, dict) or not isinstance(message.get("items"), list):
            raise SourceError("Crossref returned an invalid works envelope.")

        raw_results = message["items"]
        valid = tuple(hit for item in raw_results if (hit := _normalize_item(item)) is not None)
        normalized = tuple(hit for hit in valid if scope is None or scope.matches(hit.paper))
        total_count = message.get("total-results")
        if isinstance(total_count, bool) or not isinstance(total_count, int):
            total_count = None
        return SearchBatch(
            query=clean_query,
            source=self.name,
            results=normalized,
            returned_count=len(raw_results),
            skipped_count=len(raw_results) - len(valid),
            executed_at=datetime.now(timezone.utc),
            filters={
                "rows": limit,
                **({"provider_filter": params.get("filter"), "scope": scope.to_dict()} if scope else {}),
            },
            total_count=total_count,
            scope_filtered_count=len(valid) - len(normalized),
        )
