"""OpenAlex adapter and end-to-end discovery tests without network access."""

from __future__ import annotations

import unittest

import httpx
from sqlalchemy import create_engine, func, select
from sqlalchemy.pool import StaticPool

from tracescholar.config import Settings
from tracescholar.database import Base, create_session_factory
from tracescholar.discovery import search_and_persist
from tracescholar.models import Paper, ResearchRun, SearchQuery, SearchResult
from tracescholar.repositories import create_research_run, list_run_papers
from tracescholar.sources import OpenAlexSource, SourceError


def _work(work_id: str, *, title: str = "A Study of RAG") -> dict:
    return {
        "id": f"https://openalex.org/{work_id}",
        "title": title,
        "doi": "https://doi.org/10.5555/RAG.2025",
        "publication_year": 2025,
        "authorships": [
            {"author": {"display_name": "Ada Lovelace"}},
            {"author": {"display_name": "Alan Turing"}},
        ],
        "primary_location": {"source": {"display_name": "Example Journal"}},
        "abstract_inverted_index": {"retrieval": [2], "Augmented": [1], "Study": [0]},
        "locations": [{"landing_page_url": "https://arxiv.org/abs/2501.01234v2"}],
    }


class OpenAlexSourceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings(
            _env_file=None,
            openalex_base_url="https://api.openalex.org",
            openalex_api_key="example-secret",
            openalex_timeout_seconds=5,
        )

    def test_search_maps_openalex_fields_to_shared_paper_metadata(self) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200,
                json={"meta": {"count": 42}, "results": [_work("W123")]},
            )

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            batch = OpenAlexSource(settings=self.settings, client=client).search(
                " retrieval augmented generation ", limit=20
            )

        self.assertEqual(batch.query, "retrieval augmented generation")
        self.assertEqual(batch.source, "openalex")
        self.assertEqual(batch.returned_count, 1)
        self.assertEqual(batch.skipped_count, 0)
        self.assertEqual(batch.total_count, 42)
        self.assertEqual(batch.filters, {"page": 1, "per_page": 20})
        self.assertEqual(requests[0].url.params["search"], batch.query)
        self.assertEqual(requests[0].url.params["per_page"], "20")
        self.assertEqual(requests[0].headers["Authorization"], "Bearer example-secret")

        hit = batch.results[0]
        self.assertEqual(hit.source_record_id, "W123")
        self.assertEqual(hit.source_url, "https://openalex.org/W123")
        self.assertEqual(hit.paper.title, "A Study of RAG")
        self.assertEqual(hit.paper.doi, "https://doi.org/10.5555/RAG.2025")
        self.assertEqual(hit.paper.arxiv_id, "https://arxiv.org/abs/2501.01234v2")
        self.assertEqual(hit.paper.year, 2025)
        self.assertEqual(hit.paper.venue, "Example Journal")
        self.assertEqual(hit.paper.authors, ("Ada Lovelace", "Alan Turing"))
        self.assertEqual(hit.paper.abstract, "Study Augmented retrieval")

    def test_missing_optional_fields_are_empty_and_missing_required_fields_are_skipped(self) -> None:
        minimal = {"id": "https://openalex.org/W456", "title": "A minimal paper"}
        invalid = {"id": "https://openalex.org/W789", "title": None}
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json={"results": [minimal, invalid]})
        )

        with httpx.Client(transport=transport) as client:
            batch = OpenAlexSource(settings=self.settings, client=client).search("minimal")

        self.assertEqual(batch.returned_count, 2)
        self.assertEqual(batch.skipped_count, 1)
        self.assertEqual(len(batch.results), 1)
        self.assertIsNone(batch.results[0].paper.doi)
        self.assertIsNone(batch.results[0].paper.abstract)
        self.assertEqual(batch.results[0].paper.authors, ())

    def test_empty_results_are_valid(self) -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json={"meta": {"count": 0}, "results": []})
        )
        with httpx.Client(transport=transport) as client:
            batch = OpenAlexSource(settings=self.settings, client=client).search("no matches")

        self.assertEqual(batch.results, ())
        self.assertEqual(batch.returned_count, 0)
        self.assertEqual(batch.total_count, 0)

    def test_timeout_http_error_and_invalid_response_are_reported(self) -> None:
        def timed_out(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out", request=request)

        cases = (
            (timed_out, "timed out"),
            (lambda request: httpx.Response(429), "HTTP 429"),
            (lambda request: httpx.Response(200, text="not JSON"), "invalid JSON"),
            (lambda request: httpx.Response(200, json={"meta": {}}), "results envelope"),
        )
        for handler, expected in cases:
            with self.subTest(expected=expected):
                with httpx.Client(transport=httpx.MockTransport(handler)) as client:
                    with self.assertRaisesRegex(SourceError, expected):
                        OpenAlexSource(settings=self.settings, client=client, max_retries=0).search("query")

    def test_rate_limit_retries_after_header(self) -> None:
        calls = 0
        delays: list[float] = []

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(429, headers={"Retry-After": "2"})
            return httpx.Response(200, json={"meta": {"count": 0}, "results": []})

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            batch = OpenAlexSource(
                settings=self.settings,
                client=client,
                sleep=delays.append,
            ).search("retrieval")

        self.assertEqual(calls, 2)
        self.assertEqual(delays, [2.0])
        self.assertEqual(batch.returned_count, 0)


class OpenAlexPersistenceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.session_factory = create_session_factory(self.engine)
        self.run = create_research_run(
            "What is the evidence for RAG?", session_factory=self.session_factory
        )

    def tearDown(self) -> None:
        Base.metadata.drop_all(self.engine)
        self.engine.dispose()

    def test_normalized_results_are_persisted_with_every_source_record_id(self) -> None:
        results = [_work("W111"), _work("W222"), {"id": "https://openalex.org/W333"}]
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json={"meta": {"count": 3}, "results": results})
        )
        with httpx.Client(transport=transport) as client:
            source = OpenAlexSource(settings=Settings(_env_file=None), client=client)
            summary = search_and_persist(
                self.run.id,
                "retrieval augmented generation",
                source=source,
                limit=3,
                session_factory=self.session_factory,
            )

        self.assertEqual(summary.results_returned, 3)
        self.assertEqual(summary.results_skipped, 1)
        self.assertEqual(summary.results_persisted, 2)
        self.assertEqual(summary.papers_persisted, 1)
        self.assertEqual(summary.papers_in_run, 1)

        with self.session_factory() as session:
            restored_run = session.get(ResearchRun, self.run.id)
            restored_query = session.get(SearchQuery, summary.search_query_id)
            papers = list_run_papers(session, self.run.id)
            hits = list(session.scalars(select(SearchResult)))
            self.assertEqual(len(restored_run.search_queries), 1)
            self.assertEqual(restored_query.query, "retrieval augmented generation")
            self.assertEqual(restored_query.source, "openalex")
            self.assertEqual(restored_query.filters, {"page": 1, "per_page": 3})
            self.assertEqual(len(papers), 1)
            self.assertEqual(papers[0].doi, "10.5555/rag.2025")
            self.assertEqual(papers[0].arxiv_id, "2501.01234")
            self.assertEqual({hit.source_record_id for hit in hits}, {"W111", "W222"})

    def test_empty_search_is_recorded_without_papers(self) -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json={"meta": {"count": 0}, "results": []})
        )
        with httpx.Client(transport=transport) as client:
            summary = search_and_persist(
                self.run.id,
                "no matches",
                source=OpenAlexSource(settings=Settings(_env_file=None), client=client),
                session_factory=self.session_factory,
            )

        with self.session_factory() as session:
            self.assertIsNotNone(session.get(SearchQuery, summary.search_query_id))
            self.assertEqual(session.scalar(select(func.count()).select_from(Paper)), 0)

    def test_failed_request_does_not_create_search_query(self) -> None:
        transport = httpx.MockTransport(lambda request: httpx.Response(503))
        with httpx.Client(transport=transport) as client:
            with self.assertRaises(SourceError):
                search_and_persist(
                    self.run.id,
                    "temporarily unavailable",
                    source=OpenAlexSource(
                        settings=Settings(_env_file=None), client=client, max_retries=0
                    ),
                    session_factory=self.session_factory,
                )

        with self.session_factory() as session:
            self.assertEqual(session.scalar(select(func.count()).select_from(SearchQuery)), 0)
