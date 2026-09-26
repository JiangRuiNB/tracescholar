"""Manifest snapshots are validated, database-backed, and content-addressed."""

from __future__ import annotations

import unittest
import uuid

from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from tracescholar.database import Base, create_session_factory, session_scope
from tracescholar.manifests import (
    create_run_manifest,
    get_latest_run_manifest,
    get_run_manifest,
)
from tracescholar.models import (
    CanonicalStudy,
    FullTextAcquisition,
    FullTextScreeningResult,
    Paper,
    PaperVersion,
    PlannedQuery,
    ResearchPlanRecord,
    ResearchRun,
    SearchQuery,
    SearchResult,
    StudyPaper,
    StudyRunSelection,
)


class RunManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.factory = create_session_factory(self.engine)
        self.run_id = uuid.uuid4()
        with session_scope(self.factory) as session:
            run = ResearchRun(
                id=self.run_id,
                question="Does query rewriting improve multi-hop QA?",
                scope={"year_from": 2023, "languages": ["en"]},
                config_snapshot={"llm_api_key": "must-not-leak"},
            )
            session.add(run)
            session.flush()
            plan = ResearchPlanRecord(
                run_id=run.id,
                plan_json={"normalized_question": run.question,
                           "sub_questions": ["How does rewriting affect QA?"]},
                input_question=run.question,
                input_scope=run.scope,
                schema_version=1,
                prompt_version="planner-v1",
                llm_model="test-planner-model",
            )
            session.add(plan)
            session.flush()
            paper = Paper(
                title="Query rewriting for multi-hop question answering",
                normalized_title="query rewriting for multi hop question answering",
                doi="10.1000/example", year=2024, venue="Test Venue",
                authors=["A. Researcher"], abstract="An example abstract.",
            )
            session.add(paper)
            session.flush()
            study = CanonicalStudy(canonical_paper_id=paper.id, canonical_reason="test fixture")
            session.add(study)
            session.flush()
            session.add(StudyPaper(
                study_id=study.id, paper_id=paper.id, publication_role="conference",
                relationship_reason="canonical record",
            ))
            version = PaperVersion(
                paper_id=paper.id, content_hash="a" * 64, storage_path="paper.pdf",
                content_bytes=1234,
                source_url="https://papers.example/paper.pdf?token=private-value",
                source_name="Open Access Repository", license="CC-BY-4.0",
                version_label="published",
            )
            session.add(version)
            session.flush()
            planned = PlannedQuery(
                run_id=run.id, plan_id=plan.id, query="query rewriting multi-hop QA",
                query_key="query-rewriting-multihop", purpose="core", variant="exact",
                origins=[{"track": "core", "sub_question_index": 0}],
                generation_version="query-generator-v1",
            )
            session.add(planned)
            session.flush()
            search = SearchQuery(
                run_id=run.id, planned_query_id=planned.id,
                query=planned.query, source="openalex", filters={"year_from": 2023},
                returned_count=1, skipped_count=0, scope_filtered_count=0,
            )
            session.add(search)
            session.flush()
            session.add(SearchResult(
                search_query_id=search.id, paper_id=paper.id,
                source_record_id="W123456789", source_url="https://openalex.org/W123456789",
            ))
            session.add(FullTextAcquisition(
                run_id=run.id, paper_id=paper.id, paper_version_id=version.id,
                status="downloaded", source_url=version.source_url,
                source_name="Open Access Repository", license="CC-BY-4.0",
            ))
            session.add(FullTextScreeningResult(
                run_id=run.id, paper_id=paper.id, paper_version_id=version.id,
                plan_id=plan.id, status="success", label="include",
                rationale="Directly evaluates the target task.",
                matched_inclusion_criteria=["multi-hop QA"],
                matched_exclusion_criteria=[], supported_sub_question_indices=[0],
                evidence_role="primary_empirical", quality_warnings=[], input_hash="b" * 64,
                input_snapshot={"plan_id": str(plan.id)}, prompt_version="fulltext-screen-v1",
                llm_model="test-screener-model", retrieval_model_revision="retrieval-v1",
            ))
            session.add(StudyRunSelection(
                run_id=run.id, study_id=study.id,
                preferred_paper_version_id=version.id, selection_reason="included study",
                input_hash="c" * 64, policy_version="study-policy-v1",
            ))

    def tearDown(self) -> None:
        self.engine.dispose()

    def test_manifest_is_persisted_readable_and_stable_until_facts_change(self) -> None:
        first = create_run_manifest(self.run_id, session_factory=self.factory)
        self.assertTrue(first.created)
        loaded = get_run_manifest(first.manifest_id, session_factory=self.factory)
        self.assertEqual(first.manifest, loaded.manifest)
        self.assertEqual(first.content_hash, loaded.content_hash)
        self.assertEqual(first.manifest.research_run.question,
                         "Does query rewriting improve multi-hop QA?")
        self.assertEqual(len(first.manifest.planned_queries), 1)
        self.assertEqual(len(first.manifest.search_queries), 1)
        self.assertEqual(first.manifest.data_sources, ["openalex"])
        self.assertEqual(len(first.manifest.studies), 1)
        self.assertEqual(first.manifest.studies[0].final_status, "include")
        self.assertEqual(len(first.manifest.paper_versions), 1)
        self.assertNotIn("private-value", first.manifest.paper_versions[0].source_url)
        self.assertNotIn("must-not-leak", str(first.manifest.model_dump(mode="json")))

        repeat = create_run_manifest(self.run_id, session_factory=self.factory)
        self.assertFalse(repeat.created)
        self.assertEqual(first.manifest_id, repeat.manifest_id)
        self.assertEqual(first.content_hash, repeat.content_hash)

        with session_scope(self.factory) as session:
            run = session.get(ResearchRun, self.run_id)
            assert run is not None
            run.scope = {"year_from": 2024, "languages": ["en"]}
        changed = create_run_manifest(self.run_id, session_factory=self.factory)
        self.assertTrue(changed.created)
        self.assertNotEqual(first.content_hash, changed.content_hash)
        self.assertNotEqual(first.manifest_id, changed.manifest_id)
        latest = get_latest_run_manifest(self.run_id, session_factory=self.factory)
        self.assertEqual(latest.manifest_id, changed.manifest_id)

    def test_missing_run_and_manifest_are_explicit(self) -> None:
        with self.assertRaises(LookupError):
            create_run_manifest(uuid.uuid4(), session_factory=self.factory)
        with self.assertRaises(LookupError):
            get_run_manifest(uuid.uuid4(), session_factory=self.factory)
        with self.assertRaises(LookupError):
            get_latest_run_manifest(uuid.uuid4(), session_factory=self.factory)


if __name__ == "__main__":
    unittest.main()
