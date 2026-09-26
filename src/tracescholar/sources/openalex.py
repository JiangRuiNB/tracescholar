"""OpenAlex keyword-search adapter."""

from __future__ import annotations

import re
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import Any, Callable
from urllib.parse import urlsplit

import httpx

from tracescholar.config import Settings, get_settings
from tracescholar.sources.base import PaperHit, PaperMetadata, SearchBatch, SearchScope, SourceError


_WORK_ID = re.compile(r"W\d+")
_SELECT_FIELDS = (
    "id,title,doi,publication_year,language,authorships,primary_location,"
    "abstract_inverted_index,locations"
)


def _string(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _abstract_from_index(index: Any) -> str | None:
    """Reconstruct OpenAlex's position-indexed abstract when available."""
    if not isinstance(index, dict):
        return None
    positioned_words: dict[int, str] = {}
    for word, positions in index.items():
        if not isinstance(word, str) or not isinstance(positions, list):
            continue
        for position in positions:
            if isinstance(position, int) and not isinstance(position, bool) and 0 <= position < 10000:
                positioned_words.setdefault(position, word)
    if not positioned_words:
        return None
    return " ".join(word for _, word in sorted(positioned_words.items()))


def _arxiv_location(work: dict[str, Any]) -> str | None:
    locations = work.get("locations")
    if not isinstance(locations, list):
        return None
    for location in locations:
        if not isinstance(location, dict):
            continue
        for key in ("landing_page_url", "pdf_url"):
            candidate = _string(location.get(key))
            if candidate is None:
                continue
            parsed = urlsplit(candidate)
            if parsed.hostname in {"arxiv.org", "www.arxiv.org", "export.arxiv.org"} and re.match(
                r"^/(?:abs|pdf)/", parsed.path, flags=re.IGNORECASE
            ):
                return candidate
    return None


def _normalize_work(work: Any) -> PaperHit | None:
    """Map one OpenAlex work to the shared paper structure."""
    if not isinstance(work, dict):
        return None
    raw_id = _string(work.get("id"))
    title = _string(work.get("title")) or _string(work.get("display_name"))
    if raw_id is None or title is None:
        return None
    work_id = raw_id.rsplit("/", 1)[-1]
    if not _WORK_ID.fullmatch(work_id):
        return None
    if raw_id != work_id:
        parsed = urlsplit(raw_id)
        if parsed.hostname != "openalex.org" or parsed.path != f"/{work_id}":
            return None

    authorships = work.get("authorships")
    authors: list[str] = []
    if isinstance(authorships, list):
        for authorship in authorships:
            if not isinstance(authorship, dict):
                continue
            author = authorship.get("author")
            if isinstance(author, dict):
                name = _string(author.get("display_name"))
                if name:
                    authors.append(name)

    primary_location = work.get("primary_location")
    venue = None
    if isinstance(primary_location, dict):
        source = primary_location.get("source")
        if isinstance(source, dict):
            venue = _string(source.get("display_name"))
        venue = venue or _string(primary_location.get("raw_source_name"))

    year = work.get("publication_year")
    if isinstance(year, bool) or not isinstance(year, int) or not 1000 <= year <= 9999:
        year = None

    return PaperHit(
        paper=PaperMetadata(
            title=title,
            doi=_string(work.get("doi")),
            arxiv_id=_arxiv_location(work),
            year=year,
            venue=venue,
            authors=tuple(authors),
            abstract=_abstract_from_index(work.get("abstract_inverted_index")),
            language=_string(work.get("language")),
        ),
        source_record_id=work_id,
        source_url=f"https://openalex.org/{work_id}",
    )


class OpenAlexSource:
    """Fetch OpenAlex works and expose only provider-neutral paper metadata."""

    name = "openalex"

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
            raise ValueError("OpenAlex limit must be between 1 and 100.")

        params: dict[str, str | int] = {
            "search": clean_query, "per_page": limit, "page": 1, "select": _SELECT_FIELDS
        }
        filters: list[str] = []
        if scope is not None:
            if scope.year_from is not None:
                filters.append(f"from_publication_date:{scope.year_from}-01-01")
            if scope.year_to is not None:
                filters.append(f"to_publication_date:{scope.year_to}-12-31")
            if scope.languages:
                filters.append(f"language:{'|'.join(scope.languages)}")
        if filters:
            params["filter"] = ",".join(filters)
        headers = {"User-Agent": "TraceScholar/0.1 (research metadata client)"}
        if self._settings.openalex_api_key is not None:
            headers["Authorization"] = (
                f"Bearer {self._settings.openalex_api_key.get_secret_value()}"
            )
        url = f"{self._settings.openalex_base_url.rstrip('/')}/works"

        client_context = nullcontext(self._client) if self._client is not None else httpx.Client()
        try:
            with client_context as client:
                for attempt in range(self._max_retries + 1):
                    response = client.get(
                        url,
                        params=params,
                        headers=headers,
                        timeout=self._settings.openalex_timeout_seconds,
                    )
                    if response.status_code in {429, 500, 502, 503, 504} and attempt < self._max_retries:
                        retry_after = response.headers.get("Retry-After")
                        delay = float(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
                        self._sleep(min(delay, 45.0))
                        continue
                    break
            response.raise_for_status()
        except httpx.TimeoutException as error:
            raise SourceError("OpenAlex request timed out.") from error
        except httpx.HTTPStatusError as error:
            status = error.response.status_code
            guidance = " Configure TRACESCHOLAR_OPENALEX_API_KEY or retry later." if status == 429 else ""
            raise SourceError(f"OpenAlex returned HTTP {status}.{guidance}") from error
        except httpx.RequestError as error:
            raise SourceError("OpenAlex request failed.") from error

        try:
            payload = response.json()
        except ValueError as error:
            raise SourceError("OpenAlex returned invalid JSON.") from error
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise SourceError("OpenAlex returned an invalid results envelope.")

        raw_results = payload["results"]
        valid = tuple(hit for item in raw_results if (hit := _normalize_work(item)) is not None)
        normalized = tuple(hit for hit in valid if scope is None or scope.matches(hit.paper))
        meta = payload.get("meta")
        total_count = meta.get("count") if isinstance(meta, dict) else None
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
                "page": 1, "per_page": limit,
                **({"provider_filter": params["filter"], "scope": scope.to_dict()} if scope else {}),
            },
            total_count=total_count,
            scope_filtered_count=len(valid) - len(normalized),
        )
