"""Focused Writer and renderer checks with no external model or network calls."""

from __future__ import annotations

import json
import copy
import unittest
import uuid
from unittest.mock import patch

from sqlalchemy import create_engine, select
from sqlalchemy.pool import StaticPool

from tracescholar.database import Base, create_session_factory, session_scope
from tracescholar.models import (
    CanonicalStudy, Chunk, Claim, ClaimGeneration, EvidenceExtraction, EvidenceSpan,
    Paper, PaperVersion, ParsedPage, ResearchPlanRecord, ResearchRun, StudyPaper, SynthesisDraft,
    CitationAudit, CitationSentenceAudit, CitationAuditSentenceLink,
    SemanticCitationAudit, OmissionAudit,
)
from tracescholar.synthesis import (
    audit_omitted_counterevidence, audit_synthesis_citations,
    audit_synthesis_semantics, render_synthesis, write_synthesis,
)
from tracescholar.synthesis.omission_schemas import OmissionJudgment
from tracescholar.synthesis.semantic_schemas import SemanticCitationJudgment
from tracescholar.synthesis.validation import SynthesisValidationError


QUESTION = "Does rewriting improve multi-hop QA?"


def _plan() -> dict:
    return {
        "normalized_question": QUESTION,
        "sub_questions": ["What changes on multi-hop QA?"],
        "concepts": [{"term": "query rewriting", "synonyms": [], "abbreviations": []}],
        "exclusion_terms": [], "inclusion_criteria": ["Multi-hop QA evaluation"],
        "exclusion_criteria": ["Unrelated tasks"], "constraints": [],
        "ambiguity_items": [],
        "search_tracks": [{"label": "core", "intent": "counter_evidence",
                           "query": "rewriting no benefit", "rationale": "Check limitations"}],
        "stop_conditions": ["Evidence saturation"], "scope_snapshot": {},
    }


class FakeWriterLLM:
    model_name = "writer-test-model"

    def __init__(self, claim_id: uuid.UUID, evidence_id: uuid.UUID) -> None:
        self.claim_id = claim_id
        self.evidence_id = evidence_id
        self.calls = 0

    def generate(self, schema, *, system_prompt: str, user_prompt: str) -> dict:
        self.calls += 1
        assert schema.__name__ == "SynthesisDocument"
        assert "evidence_ledger" in json.loads(user_prompt)
        return {
            "title": "Query rewriting evidence",
            "research_question": QUESTION,
            "sections": [{"heading": "Findings", "paragraphs": [{"sentences": [
                {"text": "One measured result improved [F1] by two points.",
                 "claim_ids": [str(self.claim_id)],
                 "evidence_ids": [str(self.evidence_id)]},
                {"text": "The result is limited to one benchmark.",
                 "claim_ids": [str(self.claim_id)],
                 "evidence_ids": [str(self.evidence_id)]},
            ]}]}],
        }


class FakeSemanticLLM:
    model_name = "semantic-test-model"

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.fail_once_at: int | None = None
        self.invalid_once_at: int | None = None
        self.reject_at: int | None = None

    def generate(self, schema, *, system_prompt: str, user_prompt: str) -> dict:
        assert schema is SemanticCitationJudgment
        assert "Do not search for omitted counterevidence" in system_prompt
        payload = json.loads(user_prompt)
        self.calls.append(payload)
        index = payload["position"][2]
        if self.fail_once_at == index:
            self.fail_once_at = None
            raise RuntimeError("temporary model error")
        if self.invalid_once_at == index:
            self.invalid_once_at = None
            return {"verdict": "pass", "entailment": "unsupported",
                    "scope": "aligned", "strength": "calibrated",
                    "rationale": "inconsistent output", "minimal_revision": None}
        if self.reject_at == index:
            return {"verdict": "reject", "entailment": "unsupported",
                    "scope": "mismatched", "strength": "overstated",
                    "rationale": "The quote concerns a different task.",
                    "minimal_revision": None}
        if index == 0:
            return {"verdict": "pass", "entailment": "entailed",
                    "scope": "aligned", "strength": "calibrated",
                    "rationale": "The quoted result states the measured improvement.",
                    "minimal_revision": None}
        return {"verdict": "revise", "entailment": "partial",
                "scope": "too_broad", "strength": "overstated",
                "rationale": "Only this benchmark is shown.",
                "minimal_revision": "This result is limited to the reported benchmark."}


class FakeOmissionLLM:
    model_name = "omission-test-model"

    def __init__(self, verdict: str = "revise") -> None:
        self.verdict = verdict
        self.calls: list[dict] = []
        self.fail_once = False
        self.hallucinate_once = False

    def generate(self, schema, *, system_prompt: str, user_prompt: str) -> dict:
        assert schema is OmissionJudgment
        assert "Do NOT look beyond" in system_prompt
        payload = json.loads(user_prompt)
        self.calls.append(payload)
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("temporary model error")
        if self.hallucinate_once:
            self.hallucinate_once = False
            return {"verdict": "revise", "impacts": [{
                "evidence_id": str(uuid.uuid4()), "impact_type": "weakens",
            }],
                    "rationale": "invalid ID"}
        return {"verdict": self.verdict,
                "impacts": [] if self.verdict == "pass" else [{
                    "evidence_id": item["evidence_id"], "impact_type": "weakens",
                } for item in payload["candidates"]],
                "rationale": "The omitted condition materially narrows the sentence."}


class SynthesisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite+pysqlite:///:memory:",
                                    connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(self.engine)
        self.factory = create_session_factory(self.engine)
        self.run_id = uuid.uuid4()
        with session_scope(self.factory) as session:
            run = ResearchRun(id=self.run_id, question=QUESTION)
            session.add(run)
            session.flush()
            plan = ResearchPlanRecord(
                run_id=run.id, plan_json=_plan(), input_question=QUESTION,
                input_scope={}, schema_version=1, prompt_version="test", llm_model="test")
            session.add(plan)
            session.flush()
            paper = Paper(title="Measured [Rewrite] Outcome", normalized_title="measured rewrite outcome",
                          doi="10.1000/rewrite", year=2024, venue="Test Conference",
                          authors=["A. Author"], abstract="Measured result")
            session.add(paper)
            session.flush()
            version = PaperVersion(paper_id=paper.id, content_hash="a" * 64,
                                   storage_path="test.pdf", content_bytes=1000,
                                   source_url="https://example.org/test.pdf", source_name="test")
            study = CanonicalStudy(canonical_paper_id=paper.id, canonical_reason="test")
            session.add_all((version, study))
            session.flush()
            session.add(StudyPaper(study_id=study.id, paper_id=paper.id,
                                   publication_role="conference", relationship_reason="test"))
            chunk = Chunk(paper_version_id=version.id, ordinal=0,
                          text="Rewrite improved F1 by two points.", page_start=7,
                          page_end=7, section="Results", document_char_start=0,
                          document_char_end=34, locator={"page": 7, "char_start": 0,
                                                         "char_end": 34}, char_count=34,
                          token_count=6, parser_version="test")
            session.add(chunk)
            session.add(ParsedPage(paper_version_id=version.id, page_number=7,
                                   text=chunk.text, char_count=len(chunk.text),
                                   width=600, height=800, column_count=1,
                                   quality_flags=[]))
            session.flush()
            generation = ClaimGeneration(run_id=run.id, plan_id=plan.id,
                                         input_hash="b" * 64, input_snapshot={},
                                         prompt_version="test", llm_model="test",
                                         status="success", attempt_count=1)
            session.add(generation)
            session.flush()
            claim = Claim(generation_id=generation.id, sub_question_index=0,
                          statement="Rewriting can improve F1 on multi-hop QA.",
                          scope_kind="cross_study", basis_study_id=study.id,
                          basis_chunk_id=chunk.id, basis_quote=chunk.text,
                          basis_chunk_char_start=0, basis_chunk_char_end=len(chunk.text))
            session.add(claim)
            session.flush()
            extraction = EvidenceExtraction(
                claim_id=claim.id, study_id=study.id, paper_version_id=version.id,
                input_hash="c" * 64, input_snapshot={}, prompt_version="test",
                llm_model="test", retrieval_model_revision="d" * 64,
                status="success", disposition="evidence", attempt_count=1)
            session.add(extraction)
            session.flush()
            span = EvidenceSpan(extraction_id=extraction.id, study_id=study.id,
                                paper_version_id=version.id, chunk_id=chunk.id,
                                quote=chunk.text, stance="supports", confidence=0.9,
                                rationale="Measured outcome", study_context="Multi-hop QA",
                                limitations="One benchmark", page_number=7,
                                section="Results", chunk_char_start=0,
                                chunk_char_end=len(chunk.text), page_char_start=0,
                                page_char_end=len(chunk.text), locator={"quote_char_start": 0,
                                                                       "quote_char_end": len(chunk.text)})
            session.add(span)
            session.flush()
            self.generation_id, self.claim_id, self.evidence_id = generation.id, claim.id, span.id
        self.ledger = {
            "run_id": str(self.run_id), "generation_id": str(self.generation_id),
            "claim_count": 1, "included_studies": 1,
            "claims": [{"claim_id": str(self.claim_id), "statement": "Rewriting can improve F1",
                        "pending_studies": 0, "failed_tasks": 0,
                        "spans": [{"evidence_span_id": str(self.evidence_id)}]}],
        }

    def tearDown(self) -> None:
        Base.metadata.drop_all(self.engine)
        self.engine.dispose()

    def test_writer_persists_schema_and_skips_unchanged_input(self) -> None:
        llm = FakeWriterLLM(self.claim_id, self.evidence_id)
        with patch("tracescholar.synthesis.writer.get_evidence_ledger", return_value=self.ledger):
            first = write_synthesis(self.run_id, llm=llm, session_factory=self.factory)
            second = write_synthesis(self.run_id, llm=llm, session_factory=self.factory)
        self.assertTrue(first.generated)
        self.assertFalse(second.generated)
        self.assertEqual(first.draft_id, second.draft_id)
        self.assertEqual(llm.calls, 1)
        self.assertEqual(second.document.sections[0].paragraphs[0].sentences[0].claim_ids,
                         [self.claim_id])
        with session_scope(self.factory) as session:
            row = session.scalar(select(SynthesisDraft))
            self.assertEqual((row.status, row.attempt_count), ("success", 1))

    def test_invalid_evidence_reference_is_saved_and_retryable(self) -> None:
        llm = FakeWriterLLM(self.claim_id, uuid.uuid4())
        with patch("tracescholar.synthesis.writer.get_evidence_ledger", return_value=self.ledger):
            with self.assertRaises(SynthesisValidationError):
                write_synthesis(self.run_id, llm=llm, session_factory=self.factory)
            with session_scope(self.factory) as session:
                row = session.scalar(select(SynthesisDraft))
                self.assertEqual((row.status, row.attempt_count), ("failed", 1))
            llm.evidence_id = self.evidence_id
            result = write_synthesis(self.run_id, llm=llm, session_factory=self.factory)
        self.assertTrue(result.generated)
        self.assertEqual(result.attempt_count, 2)
        self.assertEqual(llm.calls, 2)

    def test_schema_rejects_a_sentence_without_evidence_citation(self) -> None:
        class NoCitationLLM(FakeWriterLLM):
            def generate(self, schema, *, system_prompt: str, user_prompt: str) -> dict:
                output = super().generate(schema, system_prompt=system_prompt,
                                          user_prompt=user_prompt)
                for sentence in output["sections"][0]["paragraphs"][0]["sentences"]:
                    sentence["evidence_ids"] = []
                return output

        llm = NoCitationLLM(self.claim_id, self.evidence_id)
        with patch("tracescholar.synthesis.writer.get_evidence_ledger", return_value=self.ledger):
            with self.assertRaisesRegex(ValueError, "at least 1 item"):
                write_synthesis(self.run_id, llm=llm, session_factory=self.factory)

    def test_renderer_uses_database_metadata_and_reuses_footnote(self) -> None:
        llm = FakeWriterLLM(self.claim_id, self.evidence_id)
        with patch("tracescholar.synthesis.writer.get_evidence_ledger", return_value=self.ledger):
            result = write_synthesis(self.run_id, llm=llm, session_factory=self.factory)
        first = render_synthesis(self.run_id, draft_id=result.draft_id,
                                 session_factory=self.factory)
        second = render_synthesis(self.run_id, draft_id=result.draft_id,
                                  session_factory=self.factory)
        self.assertEqual(first, second)
        self.assertIn("Measured \\[Rewrite\\] Outcome", first)
        self.assertIn("improved \\[F1\\]", first)
        self.assertEqual(first.count("[^e1]:"), 1)
        self.assertEqual(first.count("[^e1]"), 3)
        self.assertIn("PDF p. 7, section Results", first)
        self.assertIn("10.1000/rewrite", first)
        self.assertIn(f"Claim `{self.claim_id}`", first)
        self.assertIn("PaperVersion `", first)
        self.assertIn(str(self.evidence_id), first)
        with session_scope(self.factory) as session:
            paper = session.scalar(select(Paper))
            paper.title = "Updated database title"
        updated = render_synthesis(self.run_id, draft_id=result.draft_id,
                                   session_factory=self.factory)
        self.assertIn("Updated database title", updated)
        self.assertNotIn("Measured \\[Rewrite\\] Outcome", updated)

    def _write_draft(self) -> uuid.UUID:
        llm = FakeWriterLLM(self.claim_id, self.evidence_id)
        with patch("tracescholar.synthesis.writer.get_evidence_ledger", return_value=self.ledger):
            return write_synthesis(self.run_id, llm=llm,
                                   session_factory=self.factory).draft_id

    def _add_counterevidence(self, stance: str = "qualifies") -> uuid.UUID:
        quote = "The benefit did not transfer to a second benchmark."
        with session_scope(self.factory) as session:
            page = session.scalar(select(ParsedPage))
            original = page.text
            start = len(original) + 1
            page.text = original + "\n" + quote
            page.char_count = len(page.text)
            extraction = session.scalar(select(EvidenceExtraction))
            version = session.scalar(select(PaperVersion))
            chunk = Chunk(paper_version_id=version.id, ordinal=1, text=quote,
                          page_start=7, page_end=7, section="Results",
                          document_char_start=start, document_char_end=start + len(quote),
                          locator={"page": 7, "char_start": start,
                                   "char_end": start + len(quote)},
                          char_count=len(quote), token_count=9, parser_version="test")
            session.add(chunk)
            session.flush()
            span = EvidenceSpan(
                extraction_id=extraction.id, study_id=extraction.study_id,
                paper_version_id=version.id, chunk_id=chunk.id,
                quote=quote, stance=stance, confidence=0.8,
                rationale="Boundary condition", study_context="Second benchmark",
                limitations="One benchmark did not transfer", page_number=7,
                section="Results", chunk_char_start=0, chunk_char_end=len(quote),
                page_char_start=start, page_char_end=start + len(quote),
                locator={"page": 7, "quote_char_start": start,
                         "quote_char_end": start + len(quote)},
            )
            session.add(span)
            session.flush()
            return span.id

    def _omission_ledger(self, candidate_id: uuid.UUID | None = None,
                         stance: str = "qualifies") -> dict:
        ledger = copy.deepcopy(self.ledger)
        with session_scope(self.factory) as session:
            cited = session.get(EvidenceSpan, self.evidence_id)
            spans = [{"evidence_span_id": str(cited.id),
                      "study_id": str(cited.study_id),
                      "paper_version_id": str(cited.paper_version_id),
                      "quote": cited.quote, "stance": cited.stance}]
            if candidate_id is not None:
                candidate = session.get(EvidenceSpan, candidate_id)
                spans.append({"evidence_span_id": str(candidate.id),
                              "study_id": str(candidate.study_id),
                              "paper_version_id": str(candidate.paper_version_id),
                              "quote": candidate.quote, "stance": stance})
        ledger["claims"][0]["scope_kind"] = "cross_study"
        ledger["claims"][0]["spans"] = spans
        return ledger

    def test_auditor_persists_valid_chain_without_changing_draft(self) -> None:
        draft_id = self._write_draft()
        with session_scope(self.factory) as session:
            draft = session.get(SynthesisDraft, draft_id)
            original = (draft.status, draft.document_json.copy(), draft.updated_at)
        first = audit_synthesis_citations(self.run_id, draft_id=draft_id,
                                          session_factory=self.factory)
        second = audit_synthesis_citations(self.run_id, draft_id=draft_id,
                                           session_factory=self.factory)
        self.assertEqual(first.status, "passed")
        self.assertTrue(first.created)
        self.assertFalse(second.created)
        self.assertEqual(first.audit_id, second.audit_id)
        self.assertEqual((first.sentence_count, first.citation_count,
                          first.unique_evidence_count), (2, 2, 1))
        self.assertEqual(len(first.sentence_results), 2)
        self.assertTrue(all(item.created for item in first.sentence_results))
        self.assertTrue(all(not item.created for item in second.sentence_results))
        with session_scope(self.factory) as session:
            draft = session.get(SynthesisDraft, draft_id)
            self.assertEqual((draft.status, draft.document_json, draft.updated_at), original)
            self.assertEqual(len(session.scalars(select(CitationAudit)).all()), 1)
            sentence_rows = session.scalars(select(CitationSentenceAudit)).all()
            self.assertEqual(len(sentence_rows), 2)
            self.assertEqual(len(session.scalars(select(CitationAuditSentenceLink)).all()), 2)
            self.assertTrue(all("lineage" in row.input_snapshot for row in sentence_rows))

    def test_auditor_reuses_unchanged_sentence_and_rechecks_only_changed_sentence(self) -> None:
        draft_id = self._write_draft()
        first = audit_synthesis_citations(self.run_id, draft_id=draft_id,
                                          session_factory=self.factory)
        with session_scope(self.factory) as session:
            draft = session.get(SynthesisDraft, draft_id)
            draft.document_json["sections"][0]["paragraphs"][0]["sentences"][1]["text"] = (
                "The result is limited to this one benchmark.")
            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(draft, "document_json")
        changed = audit_synthesis_citations(self.run_id, draft_id=draft_id,
                                            session_factory=self.factory)
        self.assertTrue(changed.created)
        self.assertEqual(len(changed.sentence_results), 2)
        self.assertFalse(changed.sentence_results[0].created)
        self.assertTrue(changed.sentence_results[1].created)
        self.assertEqual(first.sentence_results[0].sentence_audit_id,
                         changed.sentence_results[0].sentence_audit_id)
        self.assertNotEqual(first.sentence_results[1].sentence_audit_id,
                            changed.sentence_results[1].sentence_audit_id)
        with session_scope(self.factory) as session:
            self.assertEqual(len(session.scalars(select(CitationSentenceAudit)).all()), 3)

    def test_auditor_detects_broken_page_quote_and_rechecks_changed_input(self) -> None:
        draft_id = self._write_draft()
        first = audit_synthesis_citations(self.run_id, draft_id=draft_id,
                                          session_factory=self.factory)
        with session_scope(self.factory) as session:
            page = session.scalar(select(ParsedPage))
            page.text = "A different passage with the same length"
        failed = audit_synthesis_citations(self.run_id, draft_id=draft_id,
                                           session_factory=self.factory)
        self.assertEqual(failed.status, "failed")
        self.assertNotEqual(first.audit_id, failed.audit_id)
        self.assertIn("quote_page_offset_mismatch", {issue["code"] for issue in failed.issues})
        with session_scope(self.factory) as session:
            self.assertEqual(len(session.scalars(select(CitationAudit)).all()), 2)

    def test_auditor_detects_wrong_study_and_uncited_claim(self) -> None:
        draft_id = self._write_draft()
        with session_scope(self.factory) as session:
            span = session.get(EvidenceSpan, self.evidence_id)
            other_paper = Paper(title="Other", normalized_title="other", year=2023,
                                authors=[], abstract="")
            session.add(other_paper)
            session.flush()
            other_study = CanonicalStudy(canonical_paper_id=other_paper.id,
                                         canonical_reason="other")
            session.add(other_study)
            session.flush()
            session.add(StudyPaper(study_id=other_study.id, paper_id=other_paper.id,
                                   publication_role="other", relationship_reason="other"))
            span.study_id = other_study.id
            draft = session.get(SynthesisDraft, draft_id)
            draft.document_json["sections"][0]["paragraphs"][0]["sentences"][0]["claim_ids"] = [str(uuid.uuid4())]
            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(draft, "document_json")
        result = audit_synthesis_citations(self.run_id, draft_id=draft_id,
                                           session_factory=self.factory)
        codes = {issue["code"] for issue in result.issues}
        self.assertEqual(result.status, "failed")
        self.assertIn("paper_version_outside_study", codes)
        self.assertIn("span_extraction_mismatch", codes)
        self.assertIn("evidence_claim_not_cited_in_sentence", codes)
        self.assertIn("missing_claim", codes)

    def test_semantic_audit_persists_per_sentence_and_preserves_draft(self) -> None:
        draft_id = self._write_draft()
        llm = FakeSemanticLLM()
        with session_scope(self.factory) as session:
            original = json.dumps(session.get(SynthesisDraft, draft_id).document_json,
                                  sort_keys=True)
        first = audit_synthesis_semantics(self.run_id, draft_id=draft_id,
                                          llm=llm, session_factory=self.factory)
        second = audit_synthesis_semantics(self.run_id, draft_id=draft_id,
                                           llm=llm, session_factory=self.factory)
        self.assertEqual(first.counts, {"pass": 1, "revise": 1,
                                        "reject": 0, "failed": 0})
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual([row.audit_id for row in first.sentences],
                         [row.audit_id for row in second.sentences])
        self.assertTrue(all(not row.generated for row in second.sentences))
        self.assertEqual(first.sentences[1].minimal_revision,
                         "This result is limited to the reported benchmark.")
        self.assertEqual(llm.calls[0]["evidence"][0]["quote"],
                         "Rewrite improved F1 by two points.")
        with session_scope(self.factory) as session:
            self.assertEqual(json.dumps(session.get(SynthesisDraft, draft_id).document_json,
                                        sort_keys=True), original)
            self.assertEqual(len(session.scalars(select(SemanticCitationAudit)).all()), 2)

    def test_semantic_audit_retries_only_failed_sentence(self) -> None:
        draft_id = self._write_draft()
        llm = FakeSemanticLLM()
        llm.fail_once_at = 1
        first = audit_synthesis_semantics(self.run_id, draft_id=draft_id,
                                          llm=llm, session_factory=self.factory)
        self.assertEqual(first.counts["failed"], 1)
        self.assertEqual(first.sentences[1].failure_code, "RuntimeError")
        second = audit_synthesis_semantics(self.run_id, draft_id=draft_id,
                                           llm=llm, session_factory=self.factory)
        self.assertEqual(second.counts["failed"], 0)
        self.assertEqual(len(llm.calls), 3)
        self.assertFalse(second.sentences[0].generated)
        self.assertTrue(second.sentences[1].generated)
        self.assertEqual(second.sentences[1].attempt_count, 2)

    def test_semantic_audit_rejects_invalid_schema_and_retries(self) -> None:
        draft_id = self._write_draft()
        llm = FakeSemanticLLM()
        llm.invalid_once_at = 0
        first = audit_synthesis_semantics(self.run_id, draft_id=draft_id,
                                          llm=llm, session_factory=self.factory)
        self.assertEqual(first.sentences[0].status, "failed")
        second = audit_synthesis_semantics(self.run_id, draft_id=draft_id,
                                           llm=llm, session_factory=self.factory)
        self.assertEqual(second.sentences[0].verdict, "pass")
        self.assertEqual(len(llm.calls), 3)

    def test_semantic_audit_requires_valid_structural_chain(self) -> None:
        draft_id = self._write_draft()
        with session_scope(self.factory) as session:
            session.scalar(select(ParsedPage)).text = "broken page"
        llm = FakeSemanticLLM()
        with self.assertRaisesRegex(ValueError, "citation-chain audit must pass"):
            audit_synthesis_semantics(self.run_id, draft_id=draft_id,
                                      llm=llm, session_factory=self.factory)
        self.assertEqual(llm.calls, [])
        with session_scope(self.factory) as session:
            self.assertEqual(session.scalars(select(SemanticCitationAudit)).all(), [])

    def test_semantic_audit_rejects_and_reaudits_changed_claim_input(self) -> None:
        draft_id = self._write_draft()
        llm = FakeSemanticLLM()
        llm.reject_at = 1
        first = audit_synthesis_semantics(self.run_id, draft_id=draft_id,
                                          llm=llm, session_factory=self.factory)
        self.assertEqual(first.counts["reject"], 1)
        self.assertIsNone(first.sentences[1].minimal_revision)
        with session_scope(self.factory) as session:
            session.get(Claim, self.claim_id).statement = "Narrower claim after correction."
        second = audit_synthesis_semantics(self.run_id, draft_id=draft_id,
                                           llm=llm, session_factory=self.factory)
        self.assertEqual(len(llm.calls), 4)
        self.assertNotEqual(first.sentences[0].audit_id, second.sentences[0].audit_id)
        self.assertTrue(all(row.generated for row in second.sentences))

    def test_omission_audit_passes_without_candidates_and_never_calls_llm(self) -> None:
        draft_id = self._write_draft()
        llm = FakeOmissionLLM()
        with patch("tracescholar.synthesis.omission_auditor.get_evidence_ledger",
                   return_value=self._omission_ledger()):
            first = audit_omitted_counterevidence(
                self.run_id, draft_id=draft_id, llm=llm, session_factory=self.factory)
            second = audit_omitted_counterevidence(
                self.run_id, draft_id=draft_id, llm=llm, session_factory=self.factory)
        self.assertEqual(first.counts, {"pass": 2, "revise": 0,
                                        "flag": 0, "failed": 0})
        self.assertEqual(llm.calls, [])
        self.assertTrue(all(not row.generated for row in second.sentences))

    def test_omission_audit_persists_material_ids_and_preserves_draft(self) -> None:
        draft_id = self._write_draft()
        counter_id = self._add_counterevidence("contradicts")
        ledger = self._omission_ledger(counter_id, "contradicts")
        llm = FakeOmissionLLM("revise")
        with session_scope(self.factory) as session:
            original = json.dumps(session.get(SynthesisDraft, draft_id).document_json,
                                  sort_keys=True)
        with patch("tracescholar.synthesis.omission_auditor.get_evidence_ledger",
                   return_value=ledger):
            first = audit_omitted_counterevidence(
                self.run_id, draft_id=draft_id, llm=llm, session_factory=self.factory)
            second = audit_omitted_counterevidence(
                self.run_id, draft_id=draft_id, llm=llm, session_factory=self.factory)
        self.assertEqual(first.counts["revise"], 2)
        self.assertEqual(len(llm.calls), 2)
        self.assertTrue(all(row.omitted_evidence_ids == (counter_id,)
                            for row in first.sentences))
        self.assertTrue(all(row.impacts[0].impact_type == "weakens"
                            for row in first.sentences))
        self.assertTrue(all(not row.generated for row in second.sentences))
        with session_scope(self.factory) as session:
            self.assertEqual(json.dumps(session.get(SynthesisDraft, draft_id).document_json,
                                        sort_keys=True), original)
            self.assertEqual(len(session.scalars(select(OmissionAudit)).all()), 2)

    def test_omission_candidate_cited_in_one_sentence_is_not_missing_there(self) -> None:
        draft_id = self._write_draft()
        counter_id = self._add_counterevidence()
        ledger = self._omission_ledger(counter_id)
        with session_scope(self.factory) as session:
            draft = session.get(SynthesisDraft, draft_id)
            draft.document_json["sections"][0]["paragraphs"][0]["sentences"][0]["evidence_ids"].append(str(counter_id))
            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(draft, "document_json")
        llm = FakeOmissionLLM()
        with patch("tracescholar.synthesis.omission_auditor.get_evidence_ledger",
                   return_value=ledger):
            result = audit_omitted_counterevidence(
                self.run_id, draft_id=draft_id, llm=llm, session_factory=self.factory)
        self.assertEqual([row.verdict for row in result.sentences], ["pass", "revise"])
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(llm.calls[0]["position"], [0, 0, 1])

    def test_omission_considers_uncited_supporting_span_by_content(self) -> None:
        draft_id = self._write_draft()
        other_id = self._add_counterevidence("supports")
        ledger = self._omission_ledger(other_id, "supports")
        llm = FakeOmissionLLM()
        with patch("tracescholar.synthesis.omission_auditor.get_evidence_ledger",
                   return_value=ledger):
            result = audit_omitted_counterevidence(
                self.run_id, draft_id=draft_id, llm=llm, session_factory=self.factory)
        self.assertEqual(result.counts["revise"], 2)
        self.assertEqual(len(llm.calls), 2)
        self.assertTrue(all(str(other_id) in {
            item["evidence_id"] for item in call["candidates"]
        } for call in llm.calls))

    def test_omission_reaudits_when_current_ledger_gains_a_candidate(self) -> None:
        draft_id = self._write_draft()
        llm = FakeOmissionLLM()
        with patch("tracescholar.synthesis.omission_auditor.get_evidence_ledger",
                   return_value=self._omission_ledger()):
            first = audit_omitted_counterevidence(
                self.run_id, draft_id=draft_id, llm=llm, session_factory=self.factory)
        counter_id = self._add_counterevidence()
        with patch("tracescholar.synthesis.omission_auditor.get_evidence_ledger",
                   return_value=self._omission_ledger(counter_id)):
            second = audit_omitted_counterevidence(
                self.run_id, draft_id=draft_id, llm=llm, session_factory=self.factory)
        self.assertEqual(first.counts["pass"], 2)
        self.assertEqual(second.counts["revise"], 2)
        self.assertEqual(len(llm.calls), 2)
        self.assertNotEqual(first.sentences[0].audit_id, second.sentences[0].audit_id)
        self.assertEqual(llm.calls[0]["candidates"][0]["study_id"],
                         llm.calls[0]["cited_evidence"][0]["study_id"])

    def test_omission_audit_nonmaterial_pass_and_uncertain_flag(self) -> None:
        draft_id = self._write_draft()
        counter_id = self._add_counterevidence()
        ledger = self._omission_ledger(counter_id)
        llm = FakeOmissionLLM("pass")
        with patch("tracescholar.synthesis.omission_auditor.get_evidence_ledger",
                   return_value=ledger):
            first = audit_omitted_counterevidence(
                self.run_id, draft_id=draft_id, llm=llm, session_factory=self.factory)
            self.assertEqual(first.counts["pass"], 2)
            self.assertTrue(all(not row.omitted_evidence_ids for row in first.sentences))
            llm.verdict = "flag"
            llm.model_name = "omission-test-model-v2"
            second = audit_omitted_counterevidence(
                self.run_id, draft_id=draft_id, llm=llm, session_factory=self.factory)
        self.assertEqual(second.counts["flag"], 2)
        self.assertTrue(all(row.omitted_evidence_ids == (counter_id,)
                            for row in second.sentences))

    def test_omission_audit_retries_failed_or_invalid_model_output(self) -> None:
        draft_id = self._write_draft()
        counter_id = self._add_counterevidence()
        ledger = self._omission_ledger(counter_id)
        llm = FakeOmissionLLM()
        llm.hallucinate_once = True
        with patch("tracescholar.synthesis.omission_auditor.get_evidence_ledger",
                   return_value=ledger):
            first = audit_omitted_counterevidence(
                self.run_id, draft_id=draft_id, llm=llm, session_factory=self.factory)
            self.assertEqual(first.counts["failed"], 1)
            second = audit_omitted_counterevidence(
                self.run_id, draft_id=draft_id, llm=llm, session_factory=self.factory)
        self.assertEqual(second.counts["failed"], 0)
        self.assertFalse(second.sentences[1].generated)
        self.assertEqual(second.sentences[0].attempt_count, 2)
        self.assertEqual(len(llm.calls), 3)

    def test_omission_audit_flags_unverifiable_candidate_without_llm(self) -> None:
        draft_id = self._write_draft()
        counter_id = self._add_counterevidence()
        ledger = self._omission_ledger(counter_id)
        with session_scope(self.factory) as session:
            session.get(EvidenceSpan, counter_id).quote = "Not actually on the PDF page"
        llm = FakeOmissionLLM()
        with patch("tracescholar.synthesis.omission_auditor.get_evidence_ledger",
                   return_value=ledger):
            result = audit_omitted_counterevidence(
                self.run_id, draft_id=draft_id, llm=llm, session_factory=self.factory)
        self.assertEqual(result.counts["flag"], 2)
        self.assertEqual(llm.calls, [])
        self.assertEqual(result.sentences[0].omitted_evidence_ids, (counter_id,))

    def test_omission_audit_requires_valid_cited_chain(self) -> None:
        draft_id = self._write_draft()
        with session_scope(self.factory) as session:
            session.scalar(select(ParsedPage)).text = "broken page"
        with self.assertRaisesRegex(ValueError, "citation-chain audit must pass"):
            audit_omitted_counterevidence(
                self.run_id, draft_id=draft_id, llm=FakeOmissionLLM(),
                session_factory=self.factory)
