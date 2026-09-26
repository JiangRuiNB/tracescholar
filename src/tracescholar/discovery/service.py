"""Run a paper-source search and persist its normalized results."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import distinct, func, select
from sqlalchemy.orm import Session, sessionmaker

from tracescholar.database import get_session_factory, session_scope
from tracescholar.models import PlannedQuery, ResearchRun, SearchResult
from tracescholar.repositories import (
    add_search_result,
    create_search_query,
    list_run_papers,
    upsert_paper,
)
from tracescholar.sources import CrossrefSource, OpenAlexSource, PaperSource, SearchScope, SourceError


@dataclass(frozen=True, slots=True)
class SearchSummary:
    """Counts and identifiers for one persisted source search."""

    run_id: uuid.UUID
    search_query_id: uuid.UUID
    query: str
    source: str
    results_returned: int
    results_skipped: int
    results_persisted: int
    papers_persisted: int
    papers_in_run: int
    scope_filtered_count: int = 0


@dataclass(frozen=True, slots=True)
class SourceFailure:
    """A provider that failed without preventing other providers from saving."""

    source: str
    reason: str


@dataclass(frozen=True, slots=True)
class DiscoverySummary:
    """Combined outcome for one query sent to multiple paper sources."""

    run_id: uuid.UUID
    query: str
    searches: tuple[SearchSummary, ...]
    failures: tuple[SourceFailure, ...]
    total_raw_hits: int
    total_results_persisted: int
    unique_papers: int
    duplicate_papers_merged: int
    papers_in_run: int
    scope_filtered_count: int = 0


def search_and_persist(
    run_id: uuid.UUID,
    query: str,
    *,
    source: PaperSource,
    limit: int = 20,
    scope: SearchScope | None = None,
    planned_query_id: uuid.UUID | None = None,
    session_factory: sessionmaker[Session] | None = None,
) -> SearchSummary:
    """Search a source, then atomically store the query and canonical papers."""
    active_factory = session_factory or get_session_factory()
    with active_factory() as session:
        if session.get(ResearchRun, run_id) is None:
            raise LookupError(f"ResearchRun {run_id} does not exist.")

        if planned_query_id is not None:
            planned = session.get(PlannedQuery, planned_query_id)
            if planned is None or planned.run_id != run_id or planned.query != query.strip():
                raise ValueError("PlannedQuery does not match this ResearchRun and query.")

    batch = (
        source.search(query, limit=limit, scope=scope)
        if scope is not None
        else source.search(query, limit=limit)
    )
    if batch.query.strip() != query.strip() or batch.source.casefold() != source.name.casefold():
        raise ValueError("Paper source returned a mismatched query or source name.")
    accepted_results = tuple(
        hit for hit in batch.results if scope is None or scope.matches(hit.paper)
    )
    extra_scope_filtered = len(batch.results) - len(accepted_results)
    with session_scope(active_factory) as session:
        saved_query = create_search_query(
            session,
            run_id=run_id,
            query=batch.query,
            source=batch.source,
            filters={**batch.filters, **({"scope": scope.to_dict()} if scope else {})},
            executed_at=batch.executed_at,
            planned_query_id=planned_query_id,
            returned_count=batch.returned_count,
            skipped_count=batch.skipped_count,
            scope_filtered_count=batch.scope_filtered_count + extra_scope_filtered,
        )
        paper_ids: set[uuid.UUID] = set()
        result_ids: set[uuid.UUID] = set()
        for hit in accepted_results:
            metadata = hit.paper
            paper = upsert_paper(
                session,
                title=metadata.title,
                doi=metadata.doi,
                arxiv_id=metadata.arxiv_id,
                year=metadata.year,
                venue=metadata.venue,
                authors=list(metadata.authors),
                abstract=metadata.abstract,
                language=metadata.language,
            )
            result = add_search_result(
                session,
                search_query=saved_query,
                paper=paper,
                source_record_id=hit.source_record_id,
                source_url=hit.source_url,
            )
            paper_ids.add(paper.id)
            result_ids.add(result.id)
        papers_in_run = len(list_run_papers(session, run_id))
        summary = SearchSummary(
            run_id=run_id,
            search_query_id=saved_query.id,
            query=batch.query,
            source=batch.source,
            results_returned=batch.returned_count,
            results_skipped=batch.skipped_count,
            results_persisted=len(result_ids),
            papers_persisted=len(paper_ids),
            papers_in_run=papers_in_run,
            scope_filtered_count=batch.scope_filtered_count + extra_scope_filtered,
        )
    return summary


def discover_papers(
    run_id: uuid.UUID,
    query: str,
    *,
    sources: Sequence[PaperSource] | None = None,
    limit_per_source: int = 20,
    scope: SearchScope | None = None,
    planned_query_id: uuid.UUID | None = None,
    session_factory: sessionmaker[Session] | None = None,
) -> DiscoverySummary:
    """Search all configured sources, preserving successful results on partial failure.

    Each source uses its own transaction. A failed source therefore leaves no
    misleading SearchQuery, while completed sources remain durable.
    """
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("Search query must not be blank.")
    if not 1 <= limit_per_source <= 100:
        raise ValueError("Limit per source must be between 1 and 100.")
    active_sources = tuple(sources) if sources is not None else (OpenAlexSource(), CrossrefSource())
    if not active_sources:
        raise ValueError("At least one paper source is required.")
    names = [source.name.strip().casefold() for source in active_sources]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Paper sources must have unique, non-blank names.")

    active_factory = session_factory or get_session_factory()
    with active_factory() as session:
        if session.get(ResearchRun, run_id) is None:
            raise LookupError(f"ResearchRun {run_id} does not exist.")
        if planned_query_id is not None:
            planned = session.get(PlannedQuery, planned_query_id)
            if planned is None or planned.run_id != run_id or planned.query != clean_query:
                raise ValueError("PlannedQuery does not match this ResearchRun and query.")

    searches: list[SearchSummary] = []
    failures: list[SourceFailure] = []
    for source in active_sources:
        try:
            searches.append(
                search_and_persist(
                    run_id,
                    clean_query,
                    source=source,
                    limit=limit_per_source,
                    scope=scope,
                    planned_query_id=planned_query_id,
                    session_factory=active_factory,
                )
            )
        except (SourceError, ValueError) as error:
            failures.append(SourceFailure(source=source.name, reason=str(error)))

    query_ids = [search.search_query_id for search in searches]
    with active_factory() as session:
        unique_papers = (
            session.scalar(
                select(func.count(distinct(SearchResult.paper_id))).where(
                    SearchResult.search_query_id.in_(query_ids)
                )
            )
            if query_ids
            else 0
        )
        papers_in_run = len(list_run_papers(session, run_id))
    persisted = sum(search.results_persisted for search in searches)
    return DiscoverySummary(
        run_id=run_id,
        query=clean_query,
        searches=tuple(searches),
        failures=tuple(failures),
        total_raw_hits=sum(search.results_returned for search in searches),
        total_results_persisted=persisted,
        unique_papers=unique_papers or 0,
        duplicate_papers_merged=persisted - (unique_papers or 0),
        papers_in_run=papers_in_run,
        scope_filtered_count=sum(search.scope_filtered_count for search in searches),
    )
