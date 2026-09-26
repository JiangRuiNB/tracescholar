"""Find only verified open-access PDF locations for a canonical paper."""

from __future__ import annotations

import re
import uuid
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urljoin, urlsplit

import httpx

from tracescholar.config import Settings, get_settings


_WORK_ID = re.compile(r"W\d+")
_ARXIV_ID = re.compile(r"(?:\d{4}\.\d{4,5}|[a-z.\-]+/\d{7})", re.IGNORECASE)
_PDF_HOSTS = frozenset({
    "content.openalex.org", "arxiv.org", "www.arxiv.org", "export.arxiv.org",
    "aclanthology.org", "www.aclanthology.org", "openreview.net",
    "proceedings.mlr.press", "pmc.ncbi.nlm.nih.gov", "europepmc.org",
    "www.frontiersin.org", "www.mdpi.com", "papers.nips.cc", "neurips.cc",
})
_SELECT = "id,doi,open_access,best_oa_location,locations,has_content,content_urls"


class LocateError(RuntimeError):
    """Open-access metadata lookup failed; this is not the same as unavailable."""


@dataclass(frozen=True, slots=True)
class FullTextPaper:
    id: uuid.UUID
    title: str
    doi: str | None
    arxiv_id: str | None
    openalex_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FullTextLocation:
    url: str
    source_name: str
    license: str | None
    version_label: str | None


class FullTextLocator(Protocol):
    def locate(self, paper: FullTextPaper) -> tuple[FullTextLocation, ...]: ...


def safe_pdf_url(url: str) -> bool:
    """Constrain third-party URLs to known OA hosts; reject credentials and tokens."""
    try:
        parsed = urlsplit(url)
        return bool(
            parsed.scheme == "https" and parsed.hostname in _PDF_HOSTS
            and parsed.port in (None, 443) and not parsed.username and not parsed.password
            and not parsed.query and not parsed.fragment
        )
    except ValueError:
        return False


def _clean(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


class OpenAlexOALocator:
    """Use OpenAlex OA evidence, with a verified arXiv fallback."""

    def __init__(self, *, settings: Settings | None = None, client: httpx.Client | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = client

    def _get_work(self, paper: FullTextPaper) -> dict[str, Any] | None:
        lookup_ids = [item for item in paper.openalex_ids if _WORK_ID.fullmatch(item)]
        if paper.doi:
            lookup_ids.append(f"doi:{paper.doi}")
        if not lookup_ids:
            return None
        headers = {"User-Agent": "TraceScholar/0.1 (open-access locator)"}
        if self.settings.openalex_api_key is not None:
            headers["Authorization"] = f"Bearer {self.settings.openalex_api_key.get_secret_value()}"
        base = self.settings.openalex_base_url.rstrip("/")
        context = nullcontext(self.client) if self.client is not None else httpx.Client()
        try:
            with context as client:
                for lookup_id in lookup_ids:
                    url = f"{base}/works/{lookup_id}"
                    for _ in range(3):
                        response = client.get(
                            url, params={"select": _SELECT}, headers=headers,
                            timeout=self.settings.openalex_timeout_seconds,
                            follow_redirects=False,
                        )
                        if response.status_code not in (301, 302, 307, 308):
                            break
                        redirected = urljoin(url, response.headers.get("Location", ""))
                        if urlsplit(redirected).hostname != urlsplit(base).hostname \
                                or not urlsplit(redirected).path.startswith("/works/"):
                            raise LocateError("OpenAlex work redirect left the API host.")
                        url = redirected
                    else:
                        raise LocateError("OpenAlex work redirected too many times.")
                    if response.status_code == 404:
                        continue
                    response.raise_for_status()
                    payload = response.json()
                    if not isinstance(payload, dict):
                        raise LocateError("OpenAlex returned an invalid work record.")
                    observed_doi = _clean(payload.get("doi"))
                    if paper.doi and observed_doi:
                        normalized = observed_doi.casefold().removeprefix("https://doi.org/")
                        if normalized != paper.doi.casefold():
                            raise LocateError("OpenAlex work DOI does not match the canonical paper.")
                    return payload
        except httpx.TimeoutException as error:
            raise LocateError("OpenAlex full-text lookup timed out.") from error
        except httpx.HTTPStatusError as error:
            raise LocateError(f"OpenAlex full-text lookup returned HTTP {error.response.status_code}.") from error
        except httpx.RequestError as error:
            raise LocateError("OpenAlex full-text lookup request failed.") from error
        except ValueError as error:
            raise LocateError("OpenAlex returned invalid full-text metadata JSON.") from error
        return None

    def locate(self, paper: FullTextPaper) -> tuple[FullTextLocation, ...]:
        work = self._get_work(paper)
        candidates: list[FullTextLocation] = []
        if work is not None:
            oa = work.get("open_access")
            if isinstance(oa, dict) and oa.get("is_oa") is True:
                best = work.get("best_oa_location")
                locations = [best] if isinstance(best, dict) else []
                others = work.get("locations")
                if isinstance(others, list):
                    locations.extend(item for item in others if isinstance(item, dict))
                for location in locations:
                    if location.get("is_oa") is not True:
                        continue
                    url = _clean(location.get("pdf_url"))
                    if url is None or not safe_pdf_url(url):
                        continue
                    source = location.get("source")
                    name = _clean(source.get("display_name")) if isinstance(source, dict) else None
                    candidates.append(FullTextLocation(
                        url=url, source_name=(name or urlsplit(url).hostname or "oa_repository")[:128],
                        license=_clean(location.get("license")),
                        version_label=_clean(location.get("version")),
                    ))
                has_content = work.get("has_content")
                content_urls = work.get("content_urls")
                work_id = (_clean(work.get("id")) or "").rsplit("/", 1)[-1]
                if (
                    self.settings.openalex_api_key is not None
                    and _WORK_ID.fullmatch(work_id)
                    and isinstance(has_content, dict) and has_content.get("pdf") is True
                    and isinstance(content_urls, dict) and _clean(content_urls.get("pdf"))
                ):
                    license_value = _clean(best.get("license")) if isinstance(best, dict) else None
                    candidates.append(FullTextLocation(
                        url=f"https://content.openalex.org/works/{work_id}.pdf",
                        source_name="OpenAlex OA content", license=license_value,
                        version_label=_clean(best.get("version")) if isinstance(best, dict) else None,
                    ))
        if paper.arxiv_id and _ARXIV_ID.fullmatch(paper.arxiv_id):
            candidates.append(FullTextLocation(
                url=f"https://arxiv.org/pdf/{paper.arxiv_id}", source_name="arXiv",
                license=None, version_label="arxiv_latest",
            ))
        distinct: dict[str, FullTextLocation] = {}
        for item in candidates:
            distinct.setdefault(item.url, item)
        return tuple(distinct.values())
