"""Discovery fusion and partial-failure tests with in-memory persistence."""

from __future__ import annotations

import unittest
import uuid

import httpx
from sqlalchemy import create_engine, func, select
from sqlalchemy.pool import StaticPool

from tracescholar.config import Settings
from tracescholar.database import Base, create_session_factory
from tracescholar.discovery import discover_papers
from tracescholar.models import Paper, SearchQuery, SearchResult
from tracescholar.repositories import create_research_run, list_run_papers
from tracescholar.sources import CrossrefSource, OpenAlexSource


def _openalex_work(*, doi: str | None = "https://doi.org/10.5555/shared") -> dict:
    return {
        "id": "https://openalex.org/W123",
        "title": "Shared RAG Paper",
        "doi": doi,
        "publication_year": 2025,
    }


def _crossref_item() -> dict:
    return {
        "DOI": "10.5555/SHARED",
        "title": ["Shared RAG Paper"],
        "published": {"date-parts": [[2025]]},
    }


class MultiSourceDiscoveryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.session_factory = create_session_factory(self.engine)
        self.run = create_research_run("How reliable is RAG?", session_factory=self.session_factory)
        self.settings = Settings(_env_file=None)

    def tearDown(self) -> None:
        Base.metadata.drop_all(self.engine)
        self.engine.dispose()

    def _sources(self, *, openalex_doi: str | None = "https://doi.org/10.5555/shared"):
        openalex_client = httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={
                "meta": {"count": 1}, "results": [_openalex_work(doi=openalex_doi)]
            })
        ))
        crossref_client = httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={
                "message": {"total-results": 1, "items": [_crossref_item()]}
            })
        ))
        return (
            OpenAlexSource(settings=self.settings, client=openalex_client),
            CrossrefSource(settings=self.settings, client=crossref_client),
        )

    def test_two_sources_merge_same_doi_but_preserve_both_hits(self) -> None:
        sources = self._sources()
        try:
            summary = discover_papers(
                self.run.id, "retrieval augmented generation", sources=sources,
                limit_per_source=20, session_factory=self.session_factory,
            )
        finally:
            for source in sources:
                source._client.close()

        self.assertEqual(len(summary.searches), 2)
        self.assertEqual(summary.failures, ())
        self.assertEqual(summary.total_raw_hits, 2)
        self.assertEqual(summary.total_results_persisted, 2)
        self.assertEqual(summary.unique_papers, 1)
        self.assertEqual(summary.duplicate_papers_merged, 1)
        with self.session_factory() as session:
            queries = list(session.scalars(select(SearchQuery).where(SearchQuery.run_id == self.run.id)))
            papers = list_run_papers(session, self.run.id)
            hits = list(session.scalars(select(SearchResult)))
            self.assertEqual({query.source for query in queries}, {"openalex", "crossref"})
            self.assertEqual({query.query for query in queries}, {"retrieval augmented generation"})
            self.assertEqual(len(papers), 1)
            self.assertEqual(papers[0].doi, "10.5555/shared")
            self.assertEqual({hit.paper_id for hit in hits}, {papers[0].id})
            self.assertEqual({hit.source_record_id for hit in hits}, {"W123", "10.5555/shared"})

    def test_title_year_fallback_merges_when_first_source_has_no_doi(self) -> None:
        sources = self._sources(openalex_doi=None)
        try:
            summary = discover_papers(
                self.run.id, "RAG", sources=sources, session_factory=self.session_factory
            )
        finally:
            for source in sources:
                source._client.close()
        self.assertEqual(summary.unique_papers, 1)
        with self.session_factory() as session:
            self.assertEqual(session.scalar(select(func.count()).select_from(Paper)), 1)
            self.assertEqual(session.scalar(select(Paper.doi)), "10.5555/shared")

    def test_failed_source_does_not_erase_successful_source(self) -> None:
        openalex, crossref = self._sources()
        crossref._client.close()
        bad_client = httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(503)
        ))
        crossref = CrossrefSource(settings=self.settings, client=bad_client, max_retries=0)
        try:
            summary = discover_papers(
                self.run.id, "RAG", sources=(openalex, crossref),
                session_factory=self.session_factory,
            )
        finally:
            openalex._client.close()
            bad_client.close()
        self.assertEqual(len(summary.searches), 1)
        self.assertEqual(summary.searches[0].source, "openalex")
        self.assertEqual(len(summary.failures), 1)
        self.assertEqual(summary.failures[0].source, "crossref")
        self.assertEqual(summary.unique_papers, 1)
        with self.session_factory() as session:
            self.assertEqual(session.scalar(select(func.count()).select_from(SearchQuery)), 1)
            self.assertEqual(session.scalar(select(func.count()).select_from(SearchResult)), 1)

    def test_all_failed_sources_leave_no_queries(self) -> None:
        bad_client = httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(503)
        ))
        try:
            summary = discover_papers(
                self.run.id, "RAG",
                sources=(CrossrefSource(settings=self.settings, client=bad_client, max_retries=0),),
                session_factory=self.session_factory,
            )
        finally:
            bad_client.close()
        self.assertEqual(summary.searches, ())
        self.assertEqual(len(summary.failures), 1)
        self.assertEqual(summary.total_raw_hits, 0)
        self.assertEqual(summary.unique_papers, 0)
        with self.session_factory() as session:
            self.assertEqual(session.scalar(select(func.count()).select_from(SearchQuery)), 0)

    def test_second_source_succeeds_after_first_source_fails(self) -> None:
        openalex_client = httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(503)
        ))
        unused_openalex, crossref = self._sources()
        try:
            summary = discover_papers(
                self.run.id, "RAG",
                sources=(
                    OpenAlexSource(settings=self.settings, client=openalex_client, max_retries=0),
                    crossref,
                ),
                session_factory=self.session_factory,
            )
        finally:
            openalex_client.close()
            unused_openalex._client.close()
            crossref._client.close()
        self.assertEqual([search.source for search in summary.searches], ["crossref"])
        self.assertEqual([failure.source for failure in summary.failures], ["openalex"])
        self.assertEqual(summary.unique_papers, 1)

    def test_empty_provider_response_still_records_its_search_query(self) -> None:
        client = httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"message": {"items": []}})
        ))
        try:
            summary = discover_papers(
                self.run.id, "no matches",
                sources=(CrossrefSource(settings=self.settings, client=client),),
                session_factory=self.session_factory,
            )
        finally:
            client.close()
        self.assertEqual(len(summary.searches), 1)
        self.assertEqual(summary.total_raw_hits, 0)
        with self.session_factory() as session:
            self.assertEqual(session.scalar(select(func.count()).select_from(SearchQuery)), 1)

    def test_unknown_run_is_rejected_before_network_requests(self) -> None:
        called = False

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal called
            called = True
            return httpx.Response(200, json={"message": {"items": []}})

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaises(LookupError):
                discover_papers(
                    uuid.uuid4(), "RAG",
                    sources=(CrossrefSource(settings=self.settings, client=client),),
                    session_factory=self.session_factory,
                )
        self.assertFalse(called)

    def test_duplicate_source_names_are_rejected(self) -> None:
        sources = self._sources()
        try:
            with self.assertRaisesRegex(ValueError, "unique"):
                discover_papers(
                    self.run.id, "RAG", sources=(sources[0], sources[0]),
                    session_factory=self.session_factory,
                )
        finally:
            for source in sources:
                source._client.close()
