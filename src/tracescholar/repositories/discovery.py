"""Persistence helpers for executed searches and canonical papers."""

from __future__ import annotations

import re
import unicodedata
import uuid
from datetime import datetime
from typing import Any
from urllib.parse import unquote, urlsplit

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from tracescholar.models import Paper, PlannedQuery, ResearchRun, SearchQuery, SearchResult


class PaperIdentityConflict(ValueError):
    """Raised when incoming identifiers point to incompatible paper records."""


def normalize_title(title: str) -> str:
    """Create a conservative title key for candidate deduplication."""
    normalized = unicodedata.normalize("NFKC", title).casefold()
    return " ".join(re.sub(r"[\W_]+", " ", normalized).split())


def normalize_doi(doi: str | None) -> str | None:
    """Strip common DOI URL/prefix forms and normalize case."""
    if doi is None or not doi.strip():
        return None
    value = doi.strip()
    if value.lower().startswith(("http://", "https://")):
        parsed = urlsplit(value)
        if parsed.netloc.lower() not in {"doi.org", "www.doi.org", "dx.doi.org"}:
            raise ValueError("DOI URL must use doi.org.")
        value = unquote(parsed.path.lstrip("/"))
    else:
        value = re.sub(r"^doi:\s*", "", value, flags=re.IGNORECASE)
    value = value.strip().casefold()
    if not re.fullmatch(r"10\.\d{4,9}/\S+", value):
        raise ValueError("Invalid DOI format.")
    return value


def normalize_arxiv_id(arxiv_id: str | None) -> str | None:
    """Normalize an arXiv identifier, collapsing paper-version suffixes."""
    if arxiv_id is None or not arxiv_id.strip():
        return None
    value = arxiv_id.strip()
    if value.lower().startswith(("http://", "https://")):
        parsed = urlsplit(value)
        if parsed.netloc.lower() not in {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}:
            raise ValueError("arXiv URL must use arxiv.org.")
        value = re.sub(r"^/(?:abs|pdf)/", "", parsed.path, flags=re.IGNORECASE)
    else:
        value = re.sub(r"^arxiv:\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\.pdf$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"v\d+$", "", value, flags=re.IGNORECASE).casefold()
    if not re.fullmatch(r"(?:\d{4}\.\d{4,5}|[a-z.\-]+/\d{7})", value):
        raise ValueError("Invalid arXiv ID format.")
    return value


def create_search_query(
    session: Session,
    *,
    run_id: uuid.UUID,
    query: str,
    source: str,
    filters: dict[str, Any] | None = None,
    executed_at: datetime | None = None,
    planned_query_id: uuid.UUID | None = None,
    returned_count: int = 0,
    skipped_count: int = 0,
    scope_filtered_count: int = 0,
) -> SearchQuery:
    """Store one query that has been executed against a source."""
    normalized_query = query.strip()
    normalized_source = source.strip().casefold()
    if not normalized_query or not normalized_source:
        raise ValueError("Search query and source must not be blank.")
    if session.get(ResearchRun, run_id) is None:
        raise LookupError(f"ResearchRun {run_id} does not exist.")
    if planned_query_id is not None:
        planned = session.get(PlannedQuery, planned_query_id)
        if planned is None or planned.run_id != run_id or planned.query != normalized_query:
            raise ValueError("SearchQuery does not match its PlannedQuery.")
    if min(returned_count, skipped_count, scope_filtered_count) < 0:
        raise ValueError("Search result counts must not be negative.")
    search_query = SearchQuery(
        run_id=run_id,
        planned_query_id=planned_query_id,
        query=normalized_query,
        source=normalized_source,
        filters=filters or {},
        returned_count=returned_count,
        skipped_count=skipped_count,
        scope_filtered_count=scope_filtered_count,
    )
    if executed_at is not None:
        if executed_at.tzinfo is None or executed_at.utcoffset() is None:
            raise ValueError("executed_at must include a timezone.")
        search_query.executed_at = executed_at
    session.add(search_query)
    session.flush()
    return search_query


def upsert_paper(
    session: Session,
    *,
    title: str,
    doi: str | None = None,
    arxiv_id: str | None = None,
    year: int | None = None,
    venue: str | None = None,
    authors: list[str] | None = None,
    abstract: str | None = None,
    language: str | None = None,
) -> Paper:
    """Find or create a canonical paper using identifiers, then title and year."""
    clean_title = title.strip()
    title_key = normalize_title(clean_title)
    if not title_key:
        raise ValueError("Paper title must not be blank.")
    if year is not None and not 1000 <= year <= 9999:
        raise ValueError("Paper year must be between 1000 and 9999.")
    doi_key = normalize_doi(doi)
    arxiv_key = normalize_arxiv_id(arxiv_id)

    identifier_matches: list[Paper] = []
    if doi_key or arxiv_key:
        conditions = []
        if doi_key:
            conditions.append(Paper.doi == doi_key)
        if arxiv_key:
            conditions.append(Paper.arxiv_id == arxiv_key)
        identifier_matches = list(session.scalars(select(Paper).where(or_(*conditions))))
    if len(identifier_matches) > 1:
        raise PaperIdentityConflict("DOI and arXiv ID identify different papers.")

    paper = identifier_matches[0] if identifier_matches else None
    if paper is None and year is not None:
        candidates = list(
            session.scalars(
                select(Paper).where(Paper.normalized_title == title_key, Paper.year == year)
            )
        )
        compatible = [
            candidate
            for candidate in candidates
            if (not doi_key or not candidate.doi or candidate.doi == doi_key)
            and (not arxiv_key or not candidate.arxiv_id or candidate.arxiv_id == arxiv_key)
        ]
        if len(compatible) == 1:
            paper = compatible[0]
        elif len(compatible) > 1:
            raise PaperIdentityConflict("Title and year match multiple papers; add an identifier.")

    clean_authors = [author.strip() for author in authors or [] if author.strip()]
    if paper is None:
        paper = Paper(
            title=clean_title,
            normalized_title=title_key,
            doi=doi_key,
            arxiv_id=arxiv_key,
            year=year,
            venue=venue.strip() if venue else None,
            authors=clean_authors,
            abstract=abstract.strip() if abstract else None,
            language=language.strip().casefold() if language else None,
        )
        session.add(paper)
    else:
        if (doi_key and paper.doi and paper.doi != doi_key) or (
            arxiv_key and paper.arxiv_id and paper.arxiv_id != arxiv_key
        ):
            raise PaperIdentityConflict("Incoming identifiers conflict with the stored paper.")
        paper.doi = paper.doi or doi_key
        paper.arxiv_id = paper.arxiv_id or arxiv_key
        paper.year = paper.year or year
        paper.venue = paper.venue or (venue.strip() if venue else None)
        paper.authors = paper.authors or clean_authors
        paper.abstract = paper.abstract or (abstract.strip() if abstract else None)
        paper.language = paper.language or (language.strip().casefold() if language else None)
    session.flush()
    return paper


def add_search_result(
    session: Session,
    *,
    search_query: SearchQuery,
    paper: Paper,
    source_record_id: str | None = None,
    source_url: str | None = None,
) -> SearchResult:
    """Preserve a source hit; multiple source records may share one paper."""
    session.flush()
    if source_record_id:
        result = session.scalar(
            select(SearchResult).where(
                SearchResult.search_query_id == search_query.id,
                SearchResult.source_record_id == source_record_id,
            )
        )
        if result is not None and result.paper_id != paper.id:
            raise PaperIdentityConflict("Source record ID is linked to a different paper.")
    else:
        result = session.scalar(
            select(SearchResult).where(
                SearchResult.search_query_id == search_query.id,
                SearchResult.paper_id == paper.id,
            )
        )
    if result is None:
        result = SearchResult(
            search_query_id=search_query.id,
            paper_id=paper.id,
            source_record_id=source_record_id,
            source_url=source_url,
        )
        session.add(result)
        session.flush()
    return result


def list_run_papers(session: Session, run_id: uuid.UUID) -> list[Paper]:
    """Return each canonical paper discovered in a run exactly once."""
    statement = (
        select(Paper)
        .join(SearchResult, SearchResult.paper_id == Paper.id)
        .join(SearchQuery, SearchQuery.id == SearchResult.search_query_id)
        .where(SearchQuery.run_id == run_id)
        .distinct()
        .order_by(Paper.title)
    )
    return list(session.scalars(statement))
