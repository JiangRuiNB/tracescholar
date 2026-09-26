"""Bounded query generation and resumable execution of a frozen research plan."""

from __future__ import annotations

import re
import unicodedata
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from sqlalchemy import distinct, func, select
from sqlalchemy.orm import Session, sessionmaker

from tracescholar.database import get_session_factory, session_scope
from tracescholar.discovery.service import SourceFailure, discover_papers
from tracescholar.models import PlannedQuery, ResearchRun, SearchQuery, SearchResult
from tracescholar.planning.schemas import ResearchPlan, SearchTrack
from tracescholar.repositories import list_run_papers, load_research_plan
from tracescholar.sources import CrossrefSource, OpenAlexSource, PaperSource, SearchScope


GENERATION_VERSION = "deterministic-tracks-v1"
MAX_TRACKS = 6
MAX_QUERIES_PER_TRACK = 2
MAX_TOTAL_QUERIES = MAX_TRACKS * MAX_QUERIES_PER_TRACK
_NEGATIVE_CUES = ("failure", "negative", "limitation", "no improvement", "null result", "ineffective")


class QueryGenerationError(ValueError):
    """A frozen plan cannot produce a safe, bounded query set."""


@dataclass(frozen=True, slots=True)
class QueryCandidate:
    query: str
    query_key: str
    purpose: str
    variant: str
    origins: tuple[dict[str, str | int], ...]


@dataclass(frozen=True, slots=True)
class PlannedDiscoverySummary:
    run_id: uuid.UUID
    research_tracks: int
    generated_queries: int
    newly_executed: int
    skipped_existing: int
    failures: tuple[SourceFailure, ...]
    source_hits: dict[str, int]
    total_raw_hits: int
    total_results_persisted: int
    unique_papers: int
    duplicate_papers_merged: int
    papers_in_run: int
    scope_filtered_count: int


def _query_key(query: str) -> str:
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", query).casefold()))


def _plain_query(raw: str, *, expansion: bool) -> str:
    """Turn a planner's Boolean hint into plain terms accepted by both sources."""
    groups = re.findall(r"\(([^()]*)\)", raw)
    if groups:
        parts = []
        for group in groups:
            alternatives = re.split(r"\s+OR\s+", group, flags=re.IGNORECASE)
            parts.append(alternatives[min(1 if expansion else 0, len(alternatives) - 1)])
        remainder = re.sub(r"\([^()]*\)", " ", raw)
        parts.append(remainder)
        raw = " ".join(parts)
    else:
        alternatives = re.split(r"\s+OR\s+", raw, flags=re.IGNORECASE)
        raw = alternatives[min(1 if expansion else 0, len(alternatives) - 1)]
    raw = re.sub(r"\b(?:AND|OR|NOT)\b", " ", raw, flags=re.IGNORECASE)
    raw = re.sub(r"[()\"'{}\[\]]", " ", raw)
    return " ".join(raw.split()[:18])[:200].strip()


def _selected_tracks(plan: ResearchPlan) -> list[tuple[int, SearchTrack]]:
    indexed = list(enumerate(plan.search_tracks))
    if len(indexed) <= MAX_TRACKS:
        return indexed
    first = indexed[:MAX_TRACKS]
    if not any(track.intent == "counter_evidence" for _, track in first):
        counter = next(item for item in indexed[MAX_TRACKS:] if item[1].intent == "counter_evidence")
        first[-1] = counter
    return first


def _subquestion_index(plan: ResearchPlan, query: str, track_index: int) -> int:
    terms = set(_query_key(query).split())
    scores = [len(terms & set(_query_key(question).split())) for question in plan.sub_questions]
    return scores.index(max(scores)) if max(scores) else track_index % len(plan.sub_questions)


def generate_search_queries(plan: ResearchPlan) -> tuple[QueryCandidate, ...]:
    """Generate at most two traceable queries per selected track, without an LLM call."""
    if not plan.search_tracks or not plan.sub_questions or not plan.concepts:
        raise QueryGenerationError("ResearchPlan lacks tracks, sub-questions, or concepts.")
    candidates: dict[str, QueryCandidate] = {}
    for track_index, track in _selected_tracks(plan):
        for variant in ("precise", "expanded"):
            query = _plain_query(track.query, expansion=variant == "expanded")
            if variant == "expanded" and _query_key(query) == _query_key(_plain_query(track.query, expansion=False)):
                synonym = next(
                    (term for concept in plan.concepts
                     for term in (*concept.synonyms, *concept.abbreviations)
                     if _query_key(term) not in _query_key(query)),
                    None,
                )
                if synonym:
                    query = f"{query} {synonym}"[:200]
            if track.intent == "counter_evidence" and not any(
                cue in query.casefold() for cue in _NEGATIVE_CUES
            ):
                query = f"{query} failure no improvement"[:200]
            key = _query_key(query)
            if not key or len(key.split()) < 2:
                raise QueryGenerationError(f"Track {track_index} produced an unusable query.")
            origin: dict[str, str | int] = {
                "track_index": track_index,
                "track_label": track.label,
                "track_intent": track.intent,
                "sub_question_index": _subquestion_index(plan, query, track_index),
                "variant": variant,
            }
            if key in candidates:
                previous = candidates[key]
                candidates[key] = QueryCandidate(
                    previous.query, key, previous.purpose, previous.variant,
                    (*previous.origins, origin),
                )
            else:
                candidates[key] = QueryCandidate(query, key, track.intent, variant, (origin,))
    if not any(candidate.purpose == "counter_evidence" for candidate in candidates.values()):
        raise QueryGenerationError("No distinct counter-evidence query was generated.")
    if len(candidates) > MAX_TOTAL_QUERIES:
        raise QueryGenerationError("Generated query limit exceeded.")
    return tuple(candidates.values())


def _load_or_create_queries(
    run_id: uuid.UUID,
    factory: sessionmaker[Session],
    generator: Callable[[ResearchPlan], tuple[QueryCandidate, ...]],
) -> tuple[ResearchPlan, SearchScope, tuple[PlannedQuery, ...]]:
    with session_scope(factory) as session:
        run = session.scalar(select(ResearchRun).where(ResearchRun.id == run_id).with_for_update())
        if run is None:
            raise LookupError(f"ResearchRun {run_id} does not exist.")
        plan = load_research_plan(session, run_id)
        if plan is None:
            raise ValueError("ResearchRun needs a frozen ResearchPlan before discovery.")
        if plan.scope_snapshot != run.scope:
            raise ValueError("Frozen ResearchPlan scope differs from ResearchRun scope.")
        scope = SearchScope.from_snapshot(plan.scope_snapshot)
        existing = tuple(session.scalars(
            select(PlannedQuery).where(PlannedQuery.run_id == run_id).order_by(PlannedQuery.created_at, PlannedQuery.id)
        ))
        if existing:
            if any(item.plan_id != run.research_plan.id or item.generation_version != GENERATION_VERSION
                   for item in existing):
                raise QueryGenerationError("Stored query batch does not match the frozen plan or generator version.")
            session.expunge_all()
            return plan, scope, existing
        candidates = generator(plan)
        if not candidates or len(candidates) > MAX_TOTAL_QUERIES:
            raise QueryGenerationError("Generator returned an empty or excessive query batch.")
        keys = [candidate.query_key for candidate in candidates]
        if len(keys) != len(set(keys)) or not any(item.purpose == "counter_evidence" for item in candidates):
            raise QueryGenerationError("Generator returned duplicate queries or no counter-evidence query.")
        created = []
        for candidate in candidates:
            if not candidate.query.strip() or candidate.query_key != _query_key(candidate.query):
                raise QueryGenerationError("Generator returned an invalid query key.")
            item = PlannedQuery(
                run_id=run_id, plan_id=run.research_plan.id, query=candidate.query,
                query_key=candidate.query_key, purpose=candidate.purpose,
                variant=candidate.variant, origins=list(candidate.origins),
                generation_version=GENERATION_VERSION,
            )
            session.add(item)
            created.append(item)
        session.flush()
        session.expunge_all()
        return plan, scope, tuple(created)


def run_planned_discovery(
    run_id: uuid.UUID,
    *,
    sources: Sequence[PaperSource] | None = None,
    limit_per_source: int = 10,
    session_factory: sessionmaker[Session] | None = None,
    generator: Callable[[ResearchPlan], tuple[QueryCandidate, ...]] = generate_search_queries,
) -> PlannedDiscoverySummary:
    """Resume only missing query/source pairs, reusing the normal Discovery service."""
    if not 1 <= limit_per_source <= 100:
        raise ValueError("Limit per source must be between 1 and 100.")
    active_sources = tuple(sources) if sources is not None else (OpenAlexSource(), CrossrefSource())
    names = [source.name.strip().casefold() for source in active_sources]
    if not names or any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Paper sources must have unique, non-blank names.")
    factory = session_factory or get_session_factory()
    plan, scope, planned_queries = _load_or_create_queries(run_id, factory, generator)
    with factory() as session:
        completed = {
            (row.planned_query_id, row.source)
            for row in session.scalars(select(SearchQuery).where(SearchQuery.planned_query_id.in_(
                [item.id for item in planned_queries]
            )))
        }
    newly_executed = skipped_existing = 0
    failures: list[SourceFailure] = []
    for planned in planned_queries:
        for source in active_sources:
            key = (planned.id, source.name.casefold())
            if key in completed:
                skipped_existing += 1
                continue
            summary = discover_papers(
                run_id, planned.query, sources=(source,), limit_per_source=limit_per_source,
                scope=scope, planned_query_id=planned.id, session_factory=factory,
            )
            newly_executed += len(summary.searches)
            failures.extend(summary.failures)
            if summary.searches:
                completed.add(key)
    with factory() as session:
        query_ids = [item.id for item in session.scalars(
            select(SearchQuery).where(SearchQuery.planned_query_id.in_(
                [planned.id for planned in planned_queries]
            ))
        )]
        rows = list(session.scalars(select(SearchQuery).where(SearchQuery.id.in_(query_ids))))
        source_hits = {name: sum(row.returned_count for row in rows if row.source == name)
                       for name in names}
        persisted = session.scalar(select(func.count()).select_from(SearchResult).where(
            SearchResult.search_query_id.in_(query_ids)
        )) or 0
        unique = session.scalar(select(func.count(distinct(SearchResult.paper_id))).where(
            SearchResult.search_query_id.in_(query_ids)
        )) or 0
        papers_in_run = len(list_run_papers(session, run_id))
        filtered = sum(row.scope_filtered_count for row in rows)
    return PlannedDiscoverySummary(
        run_id=run_id, research_tracks=len(plan.search_tracks),
        generated_queries=len(planned_queries), newly_executed=newly_executed,
        skipped_existing=skipped_existing, failures=tuple(failures), source_hits=source_hits,
        total_raw_hits=sum(source_hits.values()), total_results_persisted=persisted,
        unique_papers=unique, duplicate_papers_merged=persisted - unique,
        papers_in_run=papers_in_run, scope_filtered_count=filtered,
    )
