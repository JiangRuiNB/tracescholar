"""Exact quote grounding, study-level counting, retries and idempotence."""

from __future__ import annotations

import json
import unittest
import uuid
from unittest.mock import patch

from pydantic import ValidationError
from sqlalchemy import create_engine, func, select
from sqlalchemy.pool import StaticPool

from tracescholar.database import Base, create_session_factory, session_scope
from tracescholar.evidence import (
    ClaimDraftBatch, EvidenceDecision, extract_evidence, generate_claims, get_evidence_ledger,
    get_evidence_span,
)
from tracescholar.evidence.schemas import EvidenceValidationError
from tracescholar.evidence.service import _Task, _conservative_stance_filter
from tracescholar.models import (
    CanonicalStudy, Chunk, ChunkEmbedding, ClaimGeneration, EvidenceExtraction, EvidenceSpan,
    FullTextAcquisition, FullTextScreeningEvidence, FullTextScreeningResult,
    Paper, PaperVersion, ParsedPage, PdfParseRecord, ResearchPlanRecord,
    ResearchRun, StudyPaper, StudyRunSelection, StudyVersionComparison,
)
from tracescholar.planning.schemas import ResearchPlan


def _plan() -> ResearchPlan:
    return ResearchPlan.model_validate({
        "normalized_question": "Does query rewriting help multi-hop RAG?",
        "sub_questions": ["Does rewriting improve multi-hop QA F1?"],
        "concepts": [{"term": "query rewriting", "synonyms": [], "abbreviations": []}],
        "exclusion_terms": [], "inclusion_criteria": ["Multi-hop RAG evaluation"],
        "exclusion_criteria": ["Single-hop only"], "constraints": [],
        "ambiguity_items": [],
        "search_tracks": [{"label": "negative", "intent": "counter_evidence",
                           "query": "rewriting fails", "rationale": "Check negative outcomes"}],
        "stop_conditions": ["Saturation"], "scope_snapshot": {},
    })


class FakeEncoder:
    provider = "test"
    model_name = "tiny"
    model_revision = "a" * 64
    encoder_version = "1"
    dimensions = 3

    def __init__(self):
        self.calls = 0

    def embed_query(self, query):
        self.calls += 1
        return [1.0, 0.0, 0.0]


class FakeLLM:
    model_name = "evidence-test"

    def __init__(self):
        self.claim_calls = 0
        self.evidence_calls = 0
        self.bad_quote = False
        self.no_evidence = False
        self.bad_basis = False
        self.duplicate_span = False
        self.claim_scope_kind = "cross_study"
        self.false_contradiction = False

    def generate(self, schema, *, system_prompt, user_prompt):
        data = json.loads(user_prompt)
        if schema.__name__ == "ClaimDraftBatch":
            self.claim_calls += 1
            source = data["candidate_sources"][0]
            return {"claims": [{"sub_question_index": 0,
                                "statement": "Query rewriting improves multi-hop QA F1.",
                                "scope_kind": self.claim_scope_kind,
                                "basis_chunk_id": source["chunk_id"],
                                "basis_quote": "Invented text is not in any PDF." if self.bad_basis
                                else source["text"][:56]}],
                    "no_claim_reason": ""}
        self.evidence_calls += 1
        if self.no_evidence:
            return {"no_evidence": True, "no_evidence_reason": "Passages do not compare F1.",
                    "spans": []}
        passage = data["candidate_passages"][0]
        quote = "This result is not in the PDF." if self.bad_quote else passage["text"][:56]
        span = {
            "chunk_id": passage["chunk_id"], "quote": quote,
            "stance": "supports" if self.bad_quote else "contradicts" if self.false_contradiction or
            data["paper_title"] == "Independent study" else "supports",
            "confidence": 0.9, "rationale": "Measured F1 result in this paper.",
            "study_context": "Multi-hop QA benchmark", "limitations": "Single dataset",
        }
        return {"no_evidence": False, "no_evidence_reason": "",
                "spans": [span, span.copy()] if self.duplicate_span else [span]}


class EvidenceLedgerTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite+pysqlite:///:memory:",
                                    connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(self.engine)
        self.factory = create_session_factory(self.engine)
        self.run_id = uuid.uuid4()
        self.llm = FakeLLM()
        self.encoder = FakeEncoder()
        self.versions = {}
        with session_scope(self.factory) as session:
            run = ResearchRun(id=self.run_id, question="Does rewriting help multi-hop RAG?")
            session.add(run)
            session.flush()
            plan = ResearchPlanRecord(run_id=run.id, plan_json=_plan().model_dump(mode="json"),
                                      input_question=run.question, input_scope={}, schema_version=1,
                                      prompt_version="test", llm_model="test")
            session.add(plan)
            session.flush()
            first_study = None
            for index, (title, role, sentence) in enumerate((
                ("Query Rewriting preprint", "preprint",
                 "Query rewriting improved multi-hop QA F1 by 2 points. Experiment A."),
                ("Query Rewriting published", "conference",
                 "Query rewriting improved multi-hop QA F1 by 2 points. Experiment B."),
                ("Independent study", "conference",
                 "Query rewriting reduced multi-hop QA F1 by 1 point. Experiment C."),
            )):
                paper = Paper(title=title, normalized_title=title.lower(), year=2024,
                              doi=f"10.1000/{index}", authors=["Test Author"],
                              abstract="Multi-hop evaluation")
                session.add(paper)
                session.flush()
                version = PaperVersion(paper_id=paper.id, content_hash=f"{index + 1:064x}",
                                       storage_path=f"test-{index}.pdf", content_bytes=100,
                                       source_url=f"https://example.org/{index}.pdf", source_name="test")
                session.add(version)
                session.flush()
                self.versions[title] = version.id
                session.add(FullTextAcquisition(run_id=run.id, paper_id=paper.id,
                                                paper_version_id=version.id,
                                                status="downloaded", attempt_count=1))
                session.add(PdfParseRecord(paper_version_id=version.id, status="success",
                                           parser_version="test", input_hash=version.content_hash,
                                           page_count=1, text_page_count=1, chunk_count=1,
                                           total_char_count=len(sentence), quality_flags=[]))
                session.add(ParsedPage(paper_version_id=version.id, page_number=7,
                                       text=sentence, char_count=len(sentence), width=100,
                                       height=100, column_count=1, quality_flags=[]))
                chunk = Chunk(paper_version_id=version.id, ordinal=0, text=sentence,
                              page_start=7, page_end=7, section="Results",
                              document_char_start=0, document_char_end=len(sentence),
                              locator={"page": 7, "char_start": 0,
                                       "char_end": len(sentence)},
                              char_count=len(sentence), token_count=10, parser_version="test")
                session.add(chunk)
                session.flush()
                session.add(ChunkEmbedding(chunk_id=chunk.id, provider="test",
                                           model_name="tiny", model_revision="a" * 64,
                                           source_revision="1", endpoint_url="https://example.org",
                                           encoder_version="1", dimensions=3, input_hash="b" * 64,
                                           status="success", vector=[1.0, 0.0, 0.0],
                                           attempt_count=1))
                screening = FullTextScreeningResult(
                    run_id=run.id, paper_id=paper.id, paper_version_id=version.id,
                    plan_id=plan.id, status="success", label="include", rationale="Relevant",
                    matched_inclusion_criteria=[], matched_exclusion_criteria=[],
                    supported_sub_question_indices=[0],
                    evidence_role="primary_empirical_evidence", quality_warnings=[],
                    input_hash="c" * 64, input_snapshot={}, prompt_version="test",
                    llm_model="test", retrieval_model_revision="a" * 64,
                    attempt_count=1)
                session.add(screening)
                session.flush()
                session.add(FullTextScreeningEvidence(screening_result_id=screening.id,
                                                      chunk_id=chunk.id, rank=1,
                                                      similarity=0.9,
                                                      retrieval_sub_question_indices=[0]))
                if index == 0:
                    first_study = CanonicalStudy(canonical_paper_id=paper.id,
                                                 canonical_reason="test")
                    session.add(first_study)
                    session.flush()
                if index == 2:
                    study = CanonicalStudy(canonical_paper_id=paper.id,
                                           canonical_reason="test")
                    session.add(study)
                    session.flush()
                else:
                    study = first_study
                    if index == 1:
                        study.canonical_paper_id = paper.id
                session.add(StudyPaper(study_id=study.id, paper_id=paper.id,
                                       publication_role=role,
                                       relationship_reason="test"))
                if index in (1, 2):
                    session.add(StudyRunSelection(run_id=run.id, study_id=study.id,
                                                  preferred_paper_version_id=version.id,
                                                  selection_reason="test", input_hash="d" * 64,
                                                  policy_version="test"))
            session.add(StudyVersionComparison(
                study_id=first_study.id,
                version_a_id=self.versions["Query Rewriting preprint"],
                version_b_id=self.versions["Query Rewriting published"],
                result_relation="not_assessed", assessment_source="system",
                assessment_note="test"))

    def tearDown(self):
        Base.metadata.drop_all(self.engine)
        self.engine.dispose()

    def _extract(self, limit=None):
        return extract_evidence(self.run_id, llm=self.llm, encoder=self.encoder,
                                session_factory=self.factory, limit=limit)

    def test_grounded_ledger_and_idempotency(self):
        first = self._extract()
        self.assertEqual((first.claims, first.included_studies, first.extraction_tasks,
                          first.newly_extracted, first.spans, first.failed), (1, 2, 2, 2, 2, 0))
        ledger = get_evidence_ledger(self.run_id, session_factory=self.factory)
        item = ledger["claims"][0]
        self.assertEqual((item["evidence_spans"], item["independent_evidence_studies"]), (2, 2))
        self.assertEqual(item["independent_primary_studies"], 2)
        self.assertEqual((item["stance_spans"]["supports"],
                          item["stance_spans"]["contradicts"]), (1, 1))
        span = get_evidence_span(self.run_id, uuid.UUID(item["spans"][0]["evidence_span_id"]),
                                 session_factory=self.factory)
        self.assertTrue(span["verified_against_parsed_page"])
        self.assertEqual(span["page"], 7)
        self.assertEqual(span["chunk_text"][span["chunk_char_start"]:span["chunk_char_end"]],
                         span["quote"])
        with self.assertRaises(LookupError):
            get_evidence_span(uuid.uuid4(), uuid.UUID(item["spans"][0]["evidence_span_id"]),
                              session_factory=self.factory)
        second = self._extract()
        self.assertEqual((second.newly_extracted, second.skipped_unchanged), (0, 2))
        self.assertEqual((self.llm.claim_calls, self.llm.evidence_calls), (1, 2))

    def test_invalid_quote_is_failure_then_retry_succeeds(self):
        self.llm.bad_quote = True
        first = self._extract(limit=1)
        self.assertEqual((first.failed, first.spans), (1, 0))
        with session_scope(self.factory) as session:
            row = session.scalar(select(EvidenceExtraction))
            self.assertEqual((row.status, row.attempt_count), ("failed", 1))
        self.llm.bad_quote = False
        second = self._extract(limit=1)
        self.assertEqual((second.failed, second.newly_extracted), (0, 1))
        with session_scope(self.factory) as session:
            row = session.scalar(select(EvidenceExtraction))
            self.assertEqual((row.status, row.attempt_count), ("success", 2))

    def test_chunk_quote_must_also_match_physical_page_offsets(self):
        generate_claims(self.run_id, llm=self.llm, session_factory=self.factory)
        with session_scope(self.factory) as session:
            for page in session.scalars(select(ParsedPage)):
                page.text = "Tampered page text does not contain the chunk quote."
        summary = self._extract(limit=1)
        self.assertEqual((summary.failed, summary.spans), (1, 0))
        self.assertIn("parsed PDF page offsets", summary.failures[0].detail)

    def test_reloaded_span_rejects_tampered_locator(self):
        self._extract(limit=1)
        with session_scope(self.factory) as session:
            span = session.scalar(select(EvidenceSpan))
            span_id = span.id
            span.page_char_start += 1
        with self.assertRaises(EvidenceValidationError):
            get_evidence_span(self.run_id, span_id, session_factory=self.factory)

    def test_duplicate_quote_in_one_extraction_is_rejected(self):
        self.llm.duplicate_span = True
        summary = self._extract(limit=1)
        self.assertEqual((summary.failed, summary.spans), (1, 0))
        self.assertIn("Duplicate evidence", summary.failures[0].detail)

    def test_explicit_no_evidence_is_success_without_span(self):
        self.llm.no_evidence = True
        summary = self._extract()
        self.assertEqual((summary.no_evidence, summary.spans, summary.failed), (2, 0, 0))
        with session_scope(self.factory) as session:
            self.assertEqual(session.scalar(select(func.count()).select_from(EvidenceSpan)), 0)
        with patch("tracescholar.evidence.service.OpenAICompatibleLLM",
                   side_effect=AssertionError("aggregation must not call an LLM")):
            item = get_evidence_ledger(self.run_id, session_factory=self.factory)["claims"][0]
        self.assertEqual(item["outcomes"]["no_evidence"]["study_count"], 2)
        self.assertEqual(item["outcomes"]["no_evidence"]["task_count"], 2)
        self.assertEqual(item["outcomes"]["supports"]["study_count"], 0)
        self.assertEqual({row["status"] for row in item["study_results"]}, {"no_evidence"})
        self.assertTrue(all(row["no_evidence_reasons"] for row in item["study_results"]))

    def test_changed_results_versions_are_separate_spans_one_study(self):
        with session_scope(self.factory) as session:
            comparison = session.scalar(select(StudyVersionComparison))
            comparison.result_relation = "changed"
            comparison.assessment_source = "human"
        summary = self._extract()
        self.assertEqual((summary.extraction_tasks, summary.spans), (3, 3))
        ledger = get_evidence_ledger(self.run_id, session_factory=self.factory)
        item = ledger["claims"][0]
        self.assertEqual((item["evidence_spans"], item["independent_evidence_studies"]), (3, 2))
        self.assertEqual(item["stance_studies"]["supports"], 1)
        self.assertEqual(item["outcomes"]["supports"]["study_count"], 1)
        self.assertEqual(item["outcomes"]["supports"]["span_count"], 2)

    def test_one_version_without_evidence_does_not_erase_other_version_evidence(self):
        with session_scope(self.factory) as session:
            comparison = session.scalar(select(StudyVersionComparison))
            comparison.result_relation = "changed"
            comparison.assessment_source = "human"
        self._extract()
        with session_scope(self.factory) as session:
            preprint = session.scalar(select(EvidenceExtraction).where(
                EvidenceExtraction.paper_version_id == self.versions["Query Rewriting preprint"]))
            preprint.spans.clear()
            preprint.disposition = "no_evidence"
            preprint.no_evidence_reason = "This PDF version has no direct result."
        item = get_evidence_ledger(self.run_id, session_factory=self.factory)["claims"][0]
        self.assertEqual(item["outcomes"]["no_evidence"]["task_count"], 1)
        self.assertEqual(item["outcomes"]["no_evidence"]["study_count"], 0)
        self.assertEqual(item["independent_evidence_studies"], 2)
        self.assertEqual(item["evaluated_studies"], 2)
        merged = next(row for row in item["study_results"] if len(row["paper_version_ids"]) == 2)
        self.assertEqual(merged["status"], "evidence")
        self.assertEqual(len(merged["no_evidence_reasons"]), 1)

    def test_newly_eligible_version_keeps_study_pending_until_extracted(self):
        self._extract()
        with session_scope(self.factory) as session:
            comparison = session.scalar(select(StudyVersionComparison))
            comparison.result_relation = "changed"
            comparison.assessment_source = "human"
        item = get_evidence_ledger(self.run_id, session_factory=self.factory)["claims"][0]
        self.assertEqual((item["evaluated_studies"], item["pending_studies"]), (1, 1))
        pending = next(row for row in item["study_results"] if row["status"] == "pending")
        self.assertEqual(pending["missing_versions"],
                         [str(self.versions["Query Rewriting preprint"])])

    def test_schema_rejects_invalid_or_fabricated_shape(self):
        with self.assertRaises(ValidationError):
            ClaimDraftBatch.model_validate({"claims": [{
                "sub_question_index": 0, "statement": "A valid looking claim statement",
                "basis_chunk_id": str(uuid.uuid4()), "basis_quote": "A quoted fragment",
            }], "no_claim_reason": ""})
        with self.assertRaises(ValidationError):
            EvidenceDecision.model_validate({"no_evidence": True, "no_evidence_reason": "",
                                             "spans": []})
        with self.assertRaises(ValidationError):
            EvidenceDecision.model_validate({"no_evidence": False, "no_evidence_reason": "",
                                             "spans": [{"chunk_id": "x", "quote": "short",
                                                        "stance": "maybe", "confidence": 2,
                                                        "rationale": "", "study_context": "",
                                                       "limitations": ""}]})

    def test_claim_generation_failure_is_saved_and_retryable(self):
        self.llm.bad_basis = True
        with self.assertRaisesRegex(ValueError, "exact retrieved substring"):
            generate_claims(self.run_id, llm=self.llm, session_factory=self.factory)
        with session_scope(self.factory) as session:
            generation = session.scalar(select(ClaimGeneration))
            self.assertEqual((generation.status, generation.attempt_count), ("failed", 1))
        self.llm.bad_basis = False
        result = generate_claims(self.run_id, llm=self.llm, session_factory=self.factory)
        self.assertEqual(result.claims, 1)
        with session_scope(self.factory) as session:
            generation = session.scalar(select(ClaimGeneration))
            self.assertEqual((generation.status, generation.attempt_count), ("success", 2))

    def test_model_revision_change_reextracts_without_overwriting_audit_history(self):
        first = self._extract()
        self.assertEqual(first.newly_extracted, 2)
        self.encoder.model_revision = "z" * 64
        with session_scope(self.factory) as session:
            for row in session.scalars(select(ChunkEmbedding)):
                row.model_revision = "z" * 64
        second = self._extract()
        self.assertEqual((second.newly_extracted, second.spans), (2, 2))
        with session_scope(self.factory) as session:
            self.assertEqual(session.scalar(select(func.count()).select_from(EvidenceExtraction)), 4)

    def test_default_ledger_ignores_outdated_extraction_prompt(self):
        self._extract()
        with session_scope(self.factory) as session:
            current = session.scalar(select(EvidenceExtraction))
            session.add(EvidenceExtraction(
                claim_id=current.claim_id, study_id=current.study_id,
                paper_version_id=current.paper_version_id,
                input_hash="f" * 64, input_snapshot={}, prompt_version="older-prompt",
                llm_model="old-model", retrieval_model_revision="a" * 64,
                status="success", disposition="no_evidence",
                no_evidence_reason="Old result", attempt_count=1,
            ))
        ledger = get_evidence_ledger(self.run_id, session_factory=self.factory)
        self.assertEqual(ledger["claims"][0]["evidence_spans"], 2)

    def test_study_specific_claim_is_not_evaluated_as_global_fact(self):
        self.llm.claim_scope_kind = "study_specific"
        summary = self._extract()
        self.assertEqual((summary.claims, summary.extraction_tasks,
                          summary.newly_extracted), (1, 1, 1))
        ledger = get_evidence_ledger(self.run_id, session_factory=self.factory)
        self.assertEqual(ledger["claims"][0]["scope_kind"], "study_specific")
        self.assertEqual(ledger["claims"][0]["independent_evidence_studies"], 1)

    def test_scope_difference_is_not_mislabeled_as_contradiction(self):
        self.llm.false_contradiction = True
        summary = self._extract()
        self.assertEqual((summary.newly_extracted, summary.failed), (2, 0))
        with session_scope(self.factory) as session:
            row = session.scalar(select(EvidenceExtraction).where(
                EvidenceExtraction.paper_version_id == self.versions["Query Rewriting published"]))
            # A positive result sentence cannot contradict a positive claim.
            self.assertEqual(row.disposition, "no_evidence")

    def test_limited_some_can_claim_is_not_refuted_by_another_study(self):
        task = _Task(uuid.uuid4(), 0,
                     "Some rewriting methods can perform worse than original queries.",
                     uuid.uuid4(), uuid.uuid4(), uuid.uuid4(), "a" * 64,
                     {"claim_scope_kind": "cross_study",
                      "claim_basis_study_id": str(uuid.uuid4())})
        decision = EvidenceDecision.model_validate({
            "no_evidence": False, "no_evidence_reason": "", "spans": [{
                "chunk_id": str(uuid.uuid4()),
                "quote": "Our method improves F1 over original query retrieval.",
                "stance": "contradicts", "confidence": 0.8,
                "rationale": "Opposite trend", "study_context": "", "limitations": "",
            }],
        })
        filtered = _conservative_stance_filter(decision, task)
        self.assertTrue(filtered.no_evidence)
        self.assertEqual(filtered.spans, [])


if __name__ == "__main__":
    unittest.main()
