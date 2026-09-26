"""Version linkage, uncertainty, result changes and run-scoped idempotency."""

from __future__ import annotations

import unittest
import uuid

from sqlalchemy import create_engine, func, select
from sqlalchemy.pool import StaticPool

from tracescholar.database import Base, create_session_factory, session_scope
from tracescholar.models import (
    CanonicalStudy, Chunk, Claim, ClaimGeneration, EvidenceExtraction,
    FullTextScreeningResult, Paper, PaperVersion, PdfParseRecord,
    ResearchPlanRecord, ResearchRun, StudyLinkCandidate, StudyPaper,
    StudyRunSelection, StudyVersionComparison,
)
from tracescholar.studies import (
    decide_study_link, get_study_details, normalize_studies, set_version_result_relation,
)


class StudyNormalizationTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite+pysqlite:///:memory:",
                                    connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(self.engine)
        self.factory = create_session_factory(self.engine)
        self.run_id = uuid.uuid4()
        self.other_run_id = uuid.uuid4()
        with session_scope(self.factory) as session:
            session.add_all([ResearchRun(id=self.run_id, question="RAG question"),
                             ResearchRun(id=self.other_run_id, question="Other RAG question")])
            session.flush()
            for run_id in (self.run_id, self.other_run_id):
                session.add(ResearchPlanRecord(
                    run_id=run_id, plan_json={}, input_question="RAG question", input_scope={},
                    schema_version=1, prompt_version="test", llm_model="test",
                ))

    def tearDown(self):
        Base.metadata.drop_all(self.engine)
        self.engine.dispose()

    def _paper(self, title: str, *, doi: str | None = None, arxiv_id: str | None = None,
               venue: str | None = None, authors: list[str] | None = None,
               abstract: str = "A long abstract about query rewriting in multi-hop question answering.",
               content_hash: str | None = None, run_id: uuid.UUID | None = None,
               parsed: bool = True) -> tuple[uuid.UUID, uuid.UUID]:
        run_id = run_id or self.run_id
        with session_scope(self.factory) as session:
            plan = session.scalar(select(ResearchPlanRecord).where(ResearchPlanRecord.run_id == run_id))
            paper = Paper(title=title, normalized_title=title.lower(), doi=doi,
                          arxiv_id=arxiv_id, venue=venue, year=2023,
                          authors=authors or ["A. Scholar", "B. Researcher"], abstract=abstract)
            session.add(paper)
            session.flush()
            version = PaperVersion(paper_id=paper.id, content_hash=content_hash or uuid.uuid4().hex * 2,
                                   storage_path=f"{paper.id}.pdf", content_bytes=100,
                                   source_url=f"https://example.org/{paper.id}.pdf", source_name="test")
            session.add(version)
            session.flush()
            if parsed:
                session.add(PdfParseRecord(
                    paper_version_id=version.id, status="success", parser_version="test",
                    input_hash=version.content_hash, page_count=2, text_page_count=2,
                    chunk_count=1, total_char_count=1000, quality_flags=[],
                ))
            session.add(FullTextScreeningResult(
                run_id=run_id, paper_id=paper.id, paper_version_id=version.id,
                plan_id=plan.id, status="success", label="include", rationale="Relevant",
                matched_inclusion_criteria=[], matched_exclusion_criteria=[],
                supported_sub_question_indices=[0], evidence_role="primary_empirical_evidence",
                quality_warnings=[], input_hash="a" * 64, input_snapshot={},
                prompt_version="test", llm_model="test", retrieval_model_revision="b" * 64,
                attempt_count=1,
            ))
            return paper.id, version.id

    def _version_pair(self):
        preprint = self._paper(
            "Query Rewriting for Retrieval-Augmented Large Language Models",
            doi="10.48550/arxiv.2305.14283", arxiv_id="2305.14283",
        )
        published = self._paper(
            "Query Rewriting in Retrieval-Augmented Large Language Models",
            doi="10.18653/v1/2023.emnlp-main.322", venue="EMNLP 2023",
        )
        return preprint, published

    def test_preprint_and_published_are_one_study_with_two_preserved_pdfs(self):
        (preprint_id, preprint_pdf), (published_id, published_pdf) = self._version_pair()
        first = normalize_studies(self.run_id, session_factory=self.factory)
        self.assertEqual((first.paper_records, first.paper_versions, first.potential_version_groups,
                          first.canonical_studies, first.linked_records, first.collapsed_surplus,
                          first.unresolved, first.newly_linked), (2, 2, 1, 1, 2, 1, 0, 2))
        with session_scope(self.factory) as session:
            study = session.scalar(select(CanonicalStudy))
            self.assertEqual(study.canonical_paper_id, published_id)
            self.assertEqual(session.scalar(select(func.count()).select_from(Paper)), 2)
            self.assertEqual(session.scalar(select(func.count()).select_from(PaperVersion)), 2)
            selection = session.scalar(select(StudyRunSelection))
            self.assertEqual(selection.preferred_paper_version_id, published_pdf)
            comparison = session.scalar(select(StudyVersionComparison))
            self.assertEqual(comparison.result_relation, "not_assessed")
            study_id = study.id
        details = get_study_details(self.run_id, study_id, session_factory=self.factory)
        self.assertEqual(details["independent_study_count"], 1)
        self.assertEqual({p["publication_role"] for p in details["papers"]},
                         {"preprint", "conference"})
        self.assertEqual(details["extraction_version_ids"], [str(published_pdf)])
        self.assertEqual({v["version_id"] for p in details["papers"] for v in p["versions"]},
                         {str(preprint_pdf), str(published_pdf)})
        second = normalize_studies(self.run_id, session_factory=self.factory)
        self.assertEqual(second.newly_linked, 0)
        self.assertEqual(second.skipped_unchanged, 1)
        with session_scope(self.factory) as session:
            self.assertEqual(session.scalar(select(func.count()).select_from(StudyPaper)), 2)
            self.assertEqual(session.scalar(select(func.count()).select_from(StudyVersionComparison)), 1)

    def test_ambiguous_pair_remains_two_independent_studies(self):
        self._paper("Graph query rewriting for retrieval augmented generation", authors=["A. Scholar"],
                    abstract="First set of results")
        self._paper("Graph query rewriting in retrieval augmented generation", authors=["A. Scholar"],
                    abstract="Different study, unknown relationship")
        summary = normalize_studies(self.run_id, session_factory=self.factory)
        self.assertEqual((summary.canonical_studies, summary.unresolved, summary.linked_records),
                         (2, 1, 0))
        with session_scope(self.factory) as session:
            candidate = session.scalar(select(StudyLinkCandidate))
            self.assertEqual(candidate.status, "unresolved")

    def test_human_resolution_merges_existing_singleton_studies(self):
        first_id, first_version_id = self._paper(
            "Graph query rewriting for retrieval augmented generation",
            authors=["A. Scholar"], abstract="First set of results")
        second_id, second_version_id = self._paper(
            "Graph query rewriting in retrieval augmented generation",
            authors=["A. Scholar"], abstract="Different abstract")
        initial = normalize_studies(self.run_id, session_factory=self.factory)
        self.assertEqual((initial.canonical_studies, initial.unresolved), (2, 1))
        with session_scope(self.factory) as session:
            source_study_id = session.scalar(select(StudyPaper.study_id).where(
                StudyPaper.paper_id == second_id))
            plan = session.scalar(select(ResearchPlanRecord).where(
                ResearchPlanRecord.run_id == self.run_id))
            chunk = Chunk(paper_version_id=second_version_id, ordinal=0,
                          text="The evaluated result differs.", page_start=1, page_end=1,
                          section="Results", document_char_start=0, document_char_end=29,
                          locator={"page": 1, "char_start": 0, "char_end": 29},
                          char_count=29, token_count=4, parser_version="test")
            session.add(chunk)
            session.flush()
            generation = ClaimGeneration(run_id=self.run_id, plan_id=plan.id,
                                         input_hash="e" * 64, input_snapshot={},
                                         prompt_version="test", llm_model="test",
                                         status="success", attempt_count=1)
            session.add(generation)
            session.flush()
            claim = Claim(generation_id=generation.id, sub_question_index=0,
                          statement="The result differs.", scope_kind="study_specific",
                          basis_study_id=source_study_id, basis_chunk_id=chunk.id,
                          basis_quote="The evaluated", basis_chunk_char_start=0,
                          basis_chunk_char_end=13)
            session.add(claim)
            session.flush()
            session.add(EvidenceExtraction(
                claim_id=claim.id, study_id=source_study_id,
                paper_version_id=second_version_id, input_hash="f" * 64,
                input_snapshot={}, prompt_version="test", llm_model="test",
                retrieval_model_revision="a" * 64, status="success",
                disposition="no_evidence", no_evidence_reason="Not enough text",
                attempt_count=1,
            ))
        decide_study_link(self.run_id, first_id, second_id, "confirmed",
                          "Manually checked both publication records", session_factory=self.factory)
        merged = normalize_studies(self.run_id, session_factory=self.factory)
        self.assertEqual((merged.canonical_studies, merged.unresolved, merged.linked_records),
                         (1, 0, 2))
        with session_scope(self.factory) as session:
            self.assertEqual(session.scalar(select(func.count()).select_from(Paper)), 2)
            self.assertEqual(session.scalar(select(func.count()).select_from(PaperVersion)), 2)
            self.assertEqual(session.scalar(select(func.count()).select_from(CanonicalStudy)), 1)
            self.assertEqual(session.scalar(select(func.count()).select_from(StudyRunSelection)), 1)
            merged_id = session.scalar(select(CanonicalStudy.id))
            self.assertEqual(session.scalar(select(Claim.basis_study_id)), merged_id)
            self.assertEqual(session.scalar(select(EvidenceExtraction.study_id)), merged_id)
        with self.assertRaisesRegex(ValueError, "explicit split"):
            decide_study_link(self.run_id, first_id, second_id, "rejected",
                              "Changed my mind", session_factory=self.factory)

    def test_human_result_change_preserves_both_extraction_versions(self):
        (preprint_id, preprint_pdf), (_, published_pdf) = self._version_pair()
        normalize_studies(self.run_id, session_factory=self.factory)
        with session_scope(self.factory) as session:
            study_id = session.scalar(select(StudyPaper.study_id).where(StudyPaper.paper_id == preprint_id))
        set_version_result_relation(self.run_id, study_id, preprint_pdf, published_pdf,
                                    "changed", "Table 3 reports a changed F1 result",
                                    session_factory=self.factory)
        normalize_studies(self.run_id, session_factory=self.factory)
        details = get_study_details(self.run_id, study_id, session_factory=self.factory)
        self.assertEqual(set(details["extraction_version_ids"]),
                         {str(preprint_pdf), str(published_pdf)})
        self.assertEqual(details["result_comparisons"][0]["result_relation"], "changed")
        self.assertEqual(details["independent_study_count"], 1)

    def test_other_run_and_no_pdf_fallback_are_isolated(self):
        self._paper("Same title", run_id=self.run_id)
        self._paper("Completely different title", run_id=self.other_run_id)
        first = normalize_studies(self.run_id, session_factory=self.factory)
        self.assertEqual((first.paper_records, first.canonical_studies), (1, 1))
        with session_scope(self.factory) as session:
            self.assertEqual(session.scalar(select(func.count()).select_from(StudyPaper)), 1)
        second = normalize_studies(self.other_run_id, session_factory=self.factory)
        self.assertEqual((second.paper_records, second.canonical_studies), (1, 1))
        with self.assertRaises(LookupError):
            get_study_details(self.run_id, uuid.uuid4(), session_factory=self.factory)

    def test_preprint_only_second_run_does_not_demote_global_canonical_record(self):
        (preprint_id, preprint_pdf), (published_id, _) = self._version_pair()
        normalize_studies(self.run_id, session_factory=self.factory)
        with session_scope(self.factory) as session:
            plan = session.scalar(select(ResearchPlanRecord).where(
                ResearchPlanRecord.run_id == self.other_run_id))
            session.add(FullTextScreeningResult(
                run_id=self.other_run_id, paper_id=preprint_id,
                paper_version_id=preprint_pdf, plan_id=plan.id, status="success",
                label="uncertain", rationale="Relevant", matched_inclusion_criteria=[],
                matched_exclusion_criteria=[], supported_sub_question_indices=[],
                evidence_role="methods", quality_warnings=[], input_hash="a" * 64,
                input_snapshot={}, prompt_version="test", llm_model="test",
                retrieval_model_revision="b" * 64, attempt_count=1,
            ))
        summary = normalize_studies(self.other_run_id, session_factory=self.factory)
        self.assertEqual((summary.paper_records, summary.canonical_studies, summary.newly_linked),
                         (1, 1, 0))
        with session_scope(self.factory) as session:
            study = session.scalar(select(CanonicalStudy))
            self.assertEqual(study.canonical_paper_id, published_id)
            selection = session.scalar(select(StudyRunSelection).where(
                StudyRunSelection.run_id == self.other_run_id))
            self.assertEqual(selection.preferred_paper_version_id, preprint_pdf)


if __name__ == "__main__":
    unittest.main()
