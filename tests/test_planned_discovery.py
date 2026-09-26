"""Frozen-plan query generation, provenance, scope, and resumable execution."""

from __future__ import annotations

import unittest
import uuid
from datetime import datetime, timezone

import httpx
from sqlalchemy import create_engine, func, select
from sqlalchemy.pool import StaticPool

from tracescholar.database import Base, create_session_factory, session_scope
from tracescholar.config import Settings
from tracescholar.discovery import QueryGenerationError, generate_search_queries, run_planned_discovery
from tracescholar.models import Paper, PlannedQuery, ResearchRun, SearchQuery, SearchResult
from tracescholar.planning import ResearchPlan
from tracescholar.repositories import create_research_run, save_research_plan
from tracescholar.sources import (
    CrossrefSource, OpenAlexSource, PaperHit, PaperMetadata, SearchBatch, SearchScope,
    SourceError,
)


def _plan(scope: dict, *, duplicate: bool = False) -> ResearchPlan:
    core_query = "retrieval augmented generation query rewriting multi hop question answering"
    return ResearchPlan.model_validate({
        "normalized_question": "RAG query rewriting in multi-hop QA",
        "sub_questions": ["Does rewriting help multi hop QA?", "When does rewriting fail?"],
        "concepts": [{"term": "query rewriting", "synonyms": ["query reformulation"],
                      "abbreviations": ["QR"]}],
        "exclusion_terms": [],
        "inclusion_criteria": ["Multi-hop QA studies"],
        "exclusion_criteria": ["Single-hop-only studies"],
        "constraints": [],
        "ambiguity_items": [],
        "search_tracks": [
            {"label": "core", "intent": "core", "query": core_query, "rationale": "Main task"},
            {"label": "method", "intent": "method",
             "query": core_query if duplicate else "query reformulation retrieval multi hop reasoning",
             "rationale": "Method variants"},
            {"label": "negative", "intent": "counter_evidence",
             "query": "query rewriting multi hop no improvement failure",
             "rationale": "Counter-evidence"},
        ],
        "stop_conditions": ["Coverage and saturation"],
        "scope_snapshot": scope,
    })


class FakeSource:
    def __init__(self, name: str, *, fail_on: str | None = None,
                 paper_year: int = 2025, paper_language: str = "en",
                 enforce_scope: bool = True) -> None:
        self.name = name
        self.fail_on = fail_on
        self.paper_year = paper_year
        self.paper_language = paper_language
        self.enforce_scope = enforce_scope
        self.calls: list[tuple[str, SearchScope]] = []

    def search(self, query: str, *, limit: int = 20, scope: SearchScope | None = None) -> SearchBatch:
        assert scope is not None
        self.calls.append((query, scope))
        if self.fail_on and self.fail_on in query:
            raise SourceError(f"{self.name} temporary failure")
        paper = PaperMetadata(
            title="Shared evidence paper", doi="10.5555/shared", year=self.paper_year,
            language=self.paper_language,
        )
        hits = (PaperHit(paper, f"{self.name}-record", "https://example.org/paper"),)
        kept = tuple(hit for hit in hits if not self.enforce_scope or scope.matches(hit.paper))
        return SearchBatch(
            query=query, source=self.name, results=kept, returned_count=1,
            skipped_count=0, executed_at=datetime.now(timezone.utc),
            filters={"scope": scope.to_dict()}, scope_filtered_count=1 - len(kept),
        )


class PlannedDiscoveryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite+pysqlite:///:memory:",
                                    connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(self.engine)
        self.factory = create_session_factory(self.engine)
        self.scope = {"year_from": 2023, "languages": ["en"]}
        self.run = create_research_run("RAG query rewriting?", scope=self.scope,
                                       session_factory=self.factory)
        with session_scope(self.factory) as session:
            save_research_plan(session, run=session.get(ResearchRun, self.run.id),
                               plan=_plan(self.scope), model_name="test", prompt_version="test")

    def tearDown(self) -> None:
        Base.metadata.drop_all(self.engine)
        self.engine.dispose()

    def test_generation_is_bounded_traceable_and_deduplicated(self) -> None:
        generated = generate_search_queries(_plan(self.scope, duplicate=True))
        self.assertLessEqual(len(generated), 6)
        self.assertEqual(len({query.query_key for query in generated}), len(generated))
        self.assertTrue(any(query.purpose == "counter_evidence" for query in generated))
        self.assertTrue(any(len(query.origins) > 1 for query in generated))
        for query in generated:
            self.assertTrue(query.origins)
            self.assertIn("sub_question_index", query.origins[0])
            self.assertNotIn(" AND ", query.query)

    def test_executes_both_sources_and_second_run_makes_no_api_calls(self) -> None:
        sources = (FakeSource("openalex"), FakeSource("crossref"))
        first = run_planned_discovery(self.run.id, sources=sources, session_factory=self.factory)
        self.assertEqual(first.research_tracks, 3)
        self.assertGreaterEqual(first.generated_queries, 3)
        self.assertEqual(first.newly_executed, first.generated_queries * 2)
        self.assertEqual(first.unique_papers, 1)
        self.assertEqual(first.total_raw_hits, first.generated_queries * 2)
        self.assertEqual(first.duplicate_papers_merged, first.total_results_persisted - 1)
        self.assertEqual({name for name in first.source_hits}, {"openalex", "crossref"})
        with self.factory() as session:
            planned = list(session.scalars(select(PlannedQuery).where(PlannedQuery.run_id == self.run.id)))
            executions = list(session.scalars(select(SearchQuery).where(SearchQuery.run_id == self.run.id)))
            self.assertEqual(len(planned), first.generated_queries)
            self.assertEqual(len(executions), first.generated_queries * 2)
            self.assertEqual({query.planned_query_id for query in executions}, {item.id for item in planned})
            self.assertEqual(session.scalar(select(func.count()).select_from(Paper)), 1)
            self.assertEqual(session.scalar(select(func.count()).select_from(SearchResult)), len(executions))
            self.assertTrue(all(query.filters["scope"]["year_from"] == 2023 for query in executions))
            self.assertTrue(all(query.origins and query.generation_version for query in planned))
        second = run_planned_discovery(self.run.id, sources=sources, session_factory=self.factory,
                                       generator=lambda _: (_ for _ in ()).throw(AssertionError("regenerated")))
        self.assertEqual(second.newly_executed, 0)
        self.assertEqual(second.skipped_existing, first.generated_queries * 2)
        self.assertEqual(len(sources[0].calls), first.generated_queries)
        self.assertEqual(len(sources[1].calls), first.generated_queries)

    def test_one_query_failure_and_one_source_failure_can_be_retried(self) -> None:
        openalex = FakeSource("openalex", fail_on="reformulation")
        crossref = FakeSource("crossref", fail_on="query")
        first = run_planned_discovery(self.run.id, sources=(openalex, crossref),
                                      session_factory=self.factory)
        self.assertGreater(len(first.failures), 0)
        self.assertGreater(first.newly_executed, 0)
        openalex.fail_on = crossref.fail_on = None
        second = run_planned_discovery(self.run.id, sources=(openalex, crossref),
                                       session_factory=self.factory)
        self.assertEqual(second.failures, ())
        self.assertEqual(second.newly_executed, len(first.failures))
        self.assertEqual(second.total_raw_hits, second.generated_queries * 2)

    def test_generation_failure_rolls_back_before_network(self) -> None:
        source = FakeSource("openalex")
        with self.assertRaises(QueryGenerationError):
            run_planned_discovery(
                self.run.id, sources=(source,), session_factory=self.factory,
                generator=lambda _: (),
            )
        self.assertEqual(source.calls, [])
        with self.factory() as session:
            self.assertEqual(session.scalar(select(func.count()).select_from(PlannedQuery)), 0)

    def test_service_rechecks_frozen_scope_even_if_adapter_does_not(self) -> None:
        source = FakeSource("rogue", paper_year=2020, paper_language="fr", enforce_scope=False)
        summary = run_planned_discovery(self.run.id, sources=(source,), session_factory=self.factory)
        self.assertEqual(summary.total_raw_hits, summary.generated_queries)
        self.assertEqual(summary.scope_filtered_count, summary.generated_queries)
        self.assertEqual(summary.unique_papers, 0)
        with self.factory() as session:
            self.assertEqual(session.scalar(select(func.count()).select_from(Paper)), 0)

    def test_scope_is_frozen_and_unsupported_constraints_fail_closed(self) -> None:
        source = FakeSource("openalex")
        with session_scope(self.factory) as session:
            session.get(ResearchRun, self.run.id).scope = {"year_from": 2024}
        with self.assertRaises(ValueError):
            run_planned_discovery(self.run.id, sources=(source,), session_factory=self.factory)
        self.assertEqual(source.calls, [])

        other = create_research_run("Scope issue", scope={"venue": "ACL"},
                                    session_factory=self.factory)
        with session_scope(self.factory) as session:
            save_research_plan(session, run=session.get(ResearchRun, other.id),
                               plan=_plan({"venue": "ACL"}), model_name="test", prompt_version="test")
        with self.assertRaisesRegex(ValueError, "Unsupported frozen scope"):
            run_planned_discovery(other.id, sources=(source,), session_factory=self.factory)
        self.assertEqual(source.calls, [])

    def test_missing_plan_and_unknown_run_do_not_call_source(self) -> None:
        source = FakeSource("openalex")
        unplanned = create_research_run("Unplanned", session_factory=self.factory)
        with self.assertRaisesRegex(ValueError, "frozen ResearchPlan"):
            run_planned_discovery(unplanned.id, sources=(source,), session_factory=self.factory)
        with self.assertRaises(LookupError):
            run_planned_discovery(uuid.uuid4(), sources=(source,), session_factory=self.factory)
        self.assertEqual(source.calls, [])

    def test_real_adapters_send_scope_filters_and_reject_missing_language(self) -> None:
        requests: list[httpx.Request] = []

        def openalex_response(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={"results": [
                {"id": "https://openalex.org/W123", "title": "Shared evidence paper",
                 "doi": "https://doi.org/10.5555/shared", "publication_year": 2025,
                 "language": "en"},
                {"id": "https://openalex.org/W124", "title": "Old paper",
                 "publication_year": 2020, "language": "en"},
            ], "meta": {"count": 2}})

        def crossref_response(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={"message": {"items": [
                {"DOI": "10.5555/shared", "title": ["Shared evidence paper"],
                 "published": {"date-parts": [[2025]]}, "language": "en"},
                {"DOI": "10.5555/unknown", "title": ["Unknown language paper"],
                 "published": {"date-parts": [[2025]]}},
            ], "total-results": 2}})

        settings = Settings(_env_file=None)
        with httpx.Client(transport=httpx.MockTransport(openalex_response)) as openalex_client, \
                httpx.Client(transport=httpx.MockTransport(crossref_response)) as crossref_client:
            sources = (
                OpenAlexSource(settings=settings, client=openalex_client),
                CrossrefSource(settings=settings, client=crossref_client),
            )
            summary = run_planned_discovery(self.run.id, sources=sources, limit_per_source=2,
                                            session_factory=self.factory)
        self.assertEqual(summary.unique_papers, 1)
        self.assertEqual(summary.total_raw_hits, summary.generated_queries * 4)
        self.assertEqual(summary.scope_filtered_count, summary.generated_queries * 2)
        openalex_request = next(request for request in requests if "openalex" in str(request.url))
        crossref_request = next(request for request in requests if "crossref" in str(request.url))
        self.assertIn("from_publication_date:2023-01-01", openalex_request.url.params["filter"])
        self.assertIn("language:en", openalex_request.url.params["filter"])
        self.assertIn("from-pub-date:2023-01-01", crossref_request.url.params["filter"])
        self.assertNotIn("select", crossref_request.url.params)
        with self.factory() as session:
            paper = session.scalar(select(Paper))
            self.assertEqual(paper.language, "en")
            self.assertEqual(paper.year, 2025)


if __name__ == "__main__":
    unittest.main()
