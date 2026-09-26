"""Discovery persistence tests with hand-built search results."""

from __future__ import annotations

import unittest

from sqlalchemy import create_engine, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import selectinload
from sqlalchemy.pool import StaticPool

from tracescholar.database import Base, create_session_factory, session_scope
from tracescholar.models import Paper, ResearchRun, SearchQuery, SearchResult
from tracescholar.repositories import (
    PaperIdentityConflict,
    add_search_result,
    create_research_run,
    create_search_query,
    list_run_papers,
    upsert_paper,
)


class DiscoveryPersistenceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.session_factory = create_session_factory(self.engine)

    def tearDown(self) -> None:
        Base.metadata.drop_all(self.engine)
        self.engine.dispose()

    def test_run_queries_papers_and_provenance_survive_reload(self) -> None:
        run = create_research_run(
            "How does query rewriting affect multi-hop RAG?",
            session_factory=self.session_factory,
        )

        with session_scope(self.session_factory) as session:
            openalex_query = create_search_query(
                session,
                run_id=run.id,
                query="query rewriting multi-hop RAG",
                source="OpenAlex",
                filters={"from_year": 2023},
            )
            first_paper = upsert_paper(
                session,
                title="Query Rewriting in Multi-Hop RAG",
                doi="https://doi.org/10.1234/RAG.2026",
                arxiv_id="arXiv:2601.01234v2",
                year=2026,
                venue="Example Conference",
                authors=["Ada Lovelace", "Alan Turing"],
                abstract="A controlled study of query rewriting.",
            )
            add_search_result(
                session,
                search_query=openalex_query,
                paper=first_paper,
                source_record_id="W123",
            )
            add_search_result(session, search_query=openalex_query, paper=first_paper)

            second_paper = upsert_paper(
                session,
                title="Decomposition for Multi-Hop Retrieval",
                year=2025,
                authors=["Grace Hopper"],
            )
            add_search_result(session, search_query=openalex_query, paper=second_paper)

            crossref_query = create_search_query(
                session,
                run_id=run.id,
                query="query rewriting retrieval",
                source="Crossref",
                filters={"type": "journal-article"},
            )
            same_paper = upsert_paper(
                session,
                title="Query Rewriting in Multi-Hop RAG",
                doi="DOI:10.1234/rag.2026",
                year=2026,
            )
            self.assertEqual(first_paper.id, same_paper.id)
            add_search_result(
                session,
                search_query=crossref_query,
                paper=same_paper,
                source_record_id="10.1234/rag.2026",
            )

        with self.session_factory() as session:
            reloaded_run = session.scalars(
                select(ResearchRun)
                .where(ResearchRun.id == run.id)
                .options(
                    selectinload(ResearchRun.search_queries)
                    .selectinload(SearchQuery.results)
                    .selectinload(SearchResult.paper),
                    selectinload(ResearchRun.papers),
                )
            ).one()
            stored_papers = list_run_papers(session, run.id)
            paper_count = session.scalar(select(func.count()).select_from(Paper))
            result_count = session.scalar(select(func.count()).select_from(SearchResult))

            self.assertEqual(len(reloaded_run.search_queries), 2)
            self.assertEqual({query.source for query in reloaded_run.search_queries}, {"openalex", "crossref"})
            self.assertEqual(
                {query.source: query.filters for query in reloaded_run.search_queries},
                {"openalex": {"from_year": 2023}, "crossref": {"type": "journal-article"}},
            )
            self.assertTrue(all(query.executed_at for query in reloaded_run.search_queries))
            self.assertEqual(len(reloaded_run.papers), 2)
            self.assertEqual(len(stored_papers), 2)
            self.assertEqual(paper_count, 2)
            self.assertEqual(result_count, 3)

            canonical = next(paper for paper in stored_papers if paper.doi)
            self.assertEqual(canonical.doi, "10.1234/rag.2026")
            self.assertEqual(canonical.arxiv_id, "2601.01234")
            self.assertEqual(canonical.venue, "Example Conference")
            self.assertEqual(canonical.authors, ["Ada Lovelace", "Alan Turing"])
            self.assertEqual(canonical.abstract, "A controlled study of query rewriting.")
            self.assertEqual([linked_run.id for linked_run in canonical.research_runs], [run.id])
            provenance = {
                result.search_query.source: result.source_record_id
                for result in canonical.search_results
            }
            self.assertEqual(provenance, {"openalex": "W123", "crossref": "10.1234/rag.2026"})

    def test_same_paper_can_belong_to_multiple_runs(self) -> None:
        first_run = create_research_run("First question", session_factory=self.session_factory)
        second_run = create_research_run("Second question", session_factory=self.session_factory)

        with session_scope(self.session_factory) as session:
            paper = upsert_paper(session, title="Shared study", doi="10.5555/shared", year=2024)
            for run in (first_run, second_run):
                query = create_search_query(
                    session,
                    run_id=run.id,
                    query="shared study",
                    source="openalex",
                )
                add_search_result(session, search_query=query, paper=paper)

        with self.session_factory() as session:
            self.assertEqual(len(list_run_papers(session, first_run.id)), 1)
            self.assertEqual(len(list_run_papers(session, second_run.id)), 1)
            stored_paper = session.scalars(select(Paper)).one()
            self.assertEqual(
                {run.id for run in stored_paper.research_runs},
                {first_run.id, second_run.id},
            )

    def test_normalized_title_and_year_are_fallback_dedup_keys(self) -> None:
        with session_scope(self.session_factory) as session:
            original = upsert_paper(
                session,
                title="A Study of Query-Rewriting",
                year=2024,
            )
            duplicate = upsert_paper(
                session,
                title="A study of query rewriting!",
                year=2024,
                doi="10.5555/rewriting",
            )
            other_year = upsert_paper(
                session,
                title="A Study of Query-Rewriting",
                year=2025,
            )
            self.assertEqual(original.id, duplicate.id)
            self.assertNotEqual(original.id, other_year.id)

    def test_conflicting_identifiers_are_not_merged(self) -> None:
        with session_scope(self.session_factory) as session:
            upsert_paper(session, title="Paper A", doi="10.5555/a", year=2024)
            upsert_paper(session, title="Paper B", arxiv_id="2401.00001", year=2024)
            with self.assertRaises(PaperIdentityConflict):
                upsert_paper(
                    session,
                    title="Paper A",
                    doi="10.5555/a",
                    arxiv_id="2401.00001",
                    year=2024,
                )

    def test_database_enforces_identifier_uniqueness(self) -> None:
        with self.assertRaises(IntegrityError):
            with session_scope(self.session_factory) as session:
                session.add_all(
                    [
                        Paper(title="First", normalized_title="first", doi="10.5555/same"),
                        Paper(title="Second", normalized_title="second", doi="10.5555/same"),
                    ]
                )

        with self.assertRaises(IntegrityError):
            with session_scope(self.session_factory) as session:
                session.add_all(
                    [
                        Paper(title="Third", normalized_title="third", arxiv_id="2401.00001"),
                        Paper(title="Fourth", normalized_title="fourth", arxiv_id="2401.00001"),
                    ]
                )
