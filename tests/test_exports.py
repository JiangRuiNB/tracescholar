"""Deterministic report export uses saved draft, evidence, audits, and manifest."""

from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
import uuid
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from tracescholar.database import Base, create_session_factory, session_scope
from tracescholar.exports import build_grounded_review_export, write_grounded_review_export
from tracescholar.manifests.schemas import (
    ManifestAuditSummary,
    ManifestDraft,
    ManifestDraftSentence,
    ManifestEvidenceSpan,
    ManifestPaperVersion,
    ManifestRun,
    ManifestSentenceAudit,
    ManifestStageAudit,
    RunManifest,
)
from tracescholar.models import (
    CanonicalStudy,
    CitationAudit,
    CitationAuditSentenceLink,
    CitationSentenceAudit,
    Claim,
    ClaimGeneration,
    Chunk,
    EvidenceExtraction,
    EvidenceSpan,
    OmissionAudit,
    Paper,
    PaperVersion,
    ParsedPage,
    ResearchPlanRecord,
    ResearchRun,
    RunManifestRecord,
    SemanticCitationAudit,
    StudyPaper,
    SynthesisDraft,
)
from tracescholar.synthesis.schemas import SYNTHESIS_SCHEMA_VERSION


class GroundedReviewExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.factory = create_session_factory(self.engine)
        self.run_id = uuid.uuid4()
        self.manifest_id = uuid.uuid4()
        self.draft_id = uuid.uuid4()
        self.audit_id = uuid.uuid4()
        self.claim_id = uuid.uuid4()
        self.evidence_ids = [uuid.uuid4(), uuid.uuid4()]
        question = "Does rewriting improve multi-hop QA?"
        texts = [
            "Query rewriting improved exact match.",
            "The effect was small.",
            "The effect was small.",
        ]
        with session_scope(self.factory) as session:
            run = ResearchRun(id=self.run_id, question=question, status="completed")
            session.add(run)
            session.flush()
            plan = ResearchPlanRecord(
                run_id=run.id,
                plan_json={"normalized_question": question, "sub_questions": ["Effect?"]},
                input_question=question,
                input_scope={},
                schema_version=1,
                prompt_version="planner-v1",
                llm_model="planner-model",
            )
            session.add(plan)
            session.flush()
            paper = Paper(
                title="A Study of Query Rewriting",
                normalized_title="a study of query rewriting",
                doi="10.5555/export-test",
                year=2025,
                venue="Test Conference",
                authors=["Example Author"],
                abstract="A test abstract.",
            )
            session.add(paper)
            session.flush()
            study = CanonicalStudy(canonical_paper_id=paper.id, canonical_reason="test fixture")
            session.add(study)
            session.flush()
            session.add(StudyPaper(
                study_id=study.id,
                paper_id=paper.id,
                publication_role="conference",
                relationship_reason="test canonical version",
            ))
            version = PaperVersion(
                paper_id=paper.id,
                content_hash="a" * 64,
                storage_path="fixture.pdf",
                content_bytes=4096,
                source_url="https://example.org/fixture.pdf?token=do-not-export",
                source_name="Example Repository",
                license="CC-BY-4.0",
                version_label="published",
            )
            session.add(version)
            session.flush()
            page_text = "Query rewriting improved exact match. The effect was small."
            chunk = Chunk(
                paper_version_id=version.id,
                ordinal=0,
                text=page_text,
                page_start=7,
                page_end=7,
                section="Results",
                document_char_start=0,
                document_char_end=len(page_text),
                locator={"page": 7, "char_start": 0, "char_end": len(page_text)},
                char_count=len(page_text),
                token_count=10,
                parser_version="parser-v1",
            )
            session.add(chunk)
            session.add(ParsedPage(
                paper_version_id=version.id,
                page_number=7,
                text=page_text,
                char_count=len(page_text),
                width=600,
                height=800,
                column_count=1,
                quality_flags=[],
            ))
            generation = ClaimGeneration(
                id=uuid.uuid4(),
                run_id=run.id,
                plan_id=plan.id,
                input_hash="b" * 64,
                input_snapshot={},
                prompt_version="claims-v1",
                llm_model="writer-model",
                status="success",
            )
            session.add(generation)
            session.flush()
            claim = Claim(
                id=self.claim_id,
                generation_id=generation.id,
                sub_question_index=0,
                statement="Query rewriting improves exact match, with a small effect.",
                scope_kind="study_specific",
                basis_study_id=study.id,
                basis_chunk_id=chunk.id,
                basis_quote=texts[0],
                basis_chunk_char_start=0,
                basis_chunk_char_end=len(texts[0]),
            )
            session.add(claim)
            session.flush()
            extraction = EvidenceExtraction(
                id=uuid.uuid4(),
                claim_id=claim.id,
                study_id=study.id,
                paper_version_id=version.id,
                input_hash="c" * 64,
                input_snapshot={},
                prompt_version="extract-v1",
                llm_model="writer-model",
                retrieval_model_revision="embed-v1",
                status="success",
                disposition="evidence",
            )
            session.add(extraction)
            session.flush()
            first_start = page_text.index(texts[0])
            second_start = page_text.index(texts[1])
            spans = [
                EvidenceSpan(
                    id=self.evidence_ids[0],
                    extraction_id=extraction.id,
                    study_id=study.id,
                    paper_version_id=version.id,
                    chunk_id=chunk.id,
                    quote=texts[0],
                    stance="supports",
                    confidence=0.9,
                    rationale="The result directly reports the metric.",
                    study_context="One benchmark.",
                    limitations="Single benchmark.",
                    page_number=7,
                    section="Results",
                    chunk_char_start=first_start,
                    chunk_char_end=first_start + len(texts[0]),
                    page_char_start=first_start,
                    page_char_end=first_start + len(texts[0]),
                    locator={"page": 7, "char_start": first_start,
                             "char_end": first_start + len(texts[0])},
                ),
                EvidenceSpan(
                    id=self.evidence_ids[1],
                    extraction_id=extraction.id,
                    study_id=study.id,
                    paper_version_id=version.id,
                    chunk_id=chunk.id,
                    quote=texts[1],
                    stance="qualifies",
                    confidence=0.8,
                    rationale="The measured effect is described as small.",
                    study_context="One benchmark.",
                    limitations="Single benchmark.",
                    page_number=7,
                    section="Results",
                    chunk_char_start=second_start,
                    chunk_char_end=second_start + len(texts[1]),
                    page_char_start=second_start,
                    page_char_end=second_start + len(texts[1]),
                    locator={"page": 7, "char_start": second_start,
                             "char_end": second_start + len(texts[1])},
                ),
            ]
            session.add_all(spans)
            session.flush()
            document = {
                "title": "Grounded Review Test",
                "research_question": question,
                "sections": [{"heading": "Findings", "paragraphs": [{"sentences": [
                    {"text": texts[0], "claim_ids": [str(claim.id)],
                     "evidence_ids": [str(self.evidence_ids[0])]},
                    {"text": texts[1], "claim_ids": [str(claim.id)],
                     "evidence_ids": [str(self.evidence_ids[1])]},
                    {"text": texts[2], "claim_ids": [str(claim.id)],
                     "evidence_ids": [str(self.evidence_ids[1])]},
                ]}]}],
            }
            ledger = {"claims": [{
                "claim_id": str(claim.id),
                "spans": [{"evidence_span_id": str(span.id)} for span in spans],
            }]}
            draft = SynthesisDraft(
                id=self.draft_id,
                run_id=run.id,
                claim_generation_id=generation.id,
                input_hash="d" * 64,
                input_snapshot={"ledger": ledger, "research_question": question},
                schema_version=SYNTHESIS_SCHEMA_VERSION,
                prompt_version="synthesis-v1",
                llm_model="writer-model",
                status="success",
                document_json=document,
            )
            session.add(draft)
            session.flush()
            citation_audit = CitationAudit(
                id=self.audit_id,
                run_id=run.id,
                draft_id=draft.id,
                auditor_version="citation-v3",
                input_hash="e" * 64,
                status="passed",
                sentence_count=3,
                citation_count=3,
                unique_evidence_count=2,
                issues_json=[],
            )
            session.add(citation_audit)
            session.flush()

            citation_sentences = []
            semantic_sentences = []
            omission_sentences = []
            semantic_verdicts = ["pass", "revise", "reject"]
            for index, sentence_text in enumerate(texts):
                input_hash = f"{index + 1:064x}"
                citation_sentence = CitationSentenceAudit(
                    id=uuid.uuid4(),
                    run_id=run.id,
                    draft_id=draft.id,
                    auditor_version="citation-v3",
                    section_index=0,
                    paragraph_index=0,
                    sentence_index=index,
                    input_hash=input_hash,
                    input_snapshot={"sentence": sentence_text},
                    status="passed",
                    issues_json=[],
                )
                session.add(citation_sentence)
                session.flush()
                session.add(CitationAuditSentenceLink(
                    citation_audit_id=citation_audit.id,
                    sentence_audit_id=citation_sentence.id,
                ))
                citation_sentences.append(citation_sentence)
                semantic = SemanticCitationAudit(
                    id=uuid.uuid4(),
                    run_id=run.id,
                    draft_id=draft.id,
                    citation_audit_id=citation_audit.id,
                    section_index=0,
                    paragraph_index=0,
                    sentence_index=index,
                    input_hash=input_hash,
                    input_snapshot={"sentence": sentence_text},
                    prompt_version="semantic-v1",
                    llm_model="semantic-model",
                    status="success",
                    verdict=semantic_verdicts[index],
                    entailment="entailed" if index == 0 else "partial",
                    scope="aligned",
                    strength="calibrated" if index == 0 else "overstated",
                    rationale=None if index == 0 else f"Review sentence {index + 1}.",
                    minimal_revision="A narrower wording is safer." if index == 1 else None,
                )
                omission = OmissionAudit(
                    id=uuid.uuid4(),
                    run_id=run.id,
                    draft_id=draft.id,
                    citation_audit_id=citation_audit.id,
                    section_index=0,
                    paragraph_index=0,
                    sentence_index=index,
                    input_hash=input_hash,
                    input_snapshot={"sentence": sentence_text},
                    prompt_version="omission-v1",
                    llm_model="omission-model",
                    status="success",
                    verdict="pass",
                    omitted_evidence_ids=[],
                    impact_types_json=[],
                    rationale="No material omitted evidence.",
                )
                session.add_all((semantic, omission))
                semantic_sentences.append(ManifestSentenceAudit(
                    section_index=0, paragraph_index=0, sentence_index=index,
                    status="success", verdict=semantic_verdicts[index],
                    input_hash=input_hash, model_name="semantic-model",
                    prompt_version="semantic-v1",
                    rationale=semantic.rationale,
                    minimal_revision=semantic.minimal_revision,
                ))
                omission_sentences.append(ManifestSentenceAudit(
                    section_index=0, paragraph_index=0, sentence_index=index,
                    status="success", verdict="pass", input_hash=input_hash,
                    model_name="omission-model", prompt_version="omission-v1",
                    rationale="No material omitted evidence.",
                ))
            session.flush()

            manifest = RunManifest(
                research_run=ManifestRun(
                    id=run.id, question=question, status="completed", scope={},
                    config_snapshot={}, created_at=run.created_at, updated_at=run.updated_at,
                ),
                evidence_spans=[ManifestEvidenceSpan(
                    id=span.id,
                    claim_id=claim.id,
                    study_id=span.study_id,
                    paper_version_id=span.paper_version_id,
                    chunk_id=span.chunk_id,
                    quote=span.quote,
                    stance=span.stance,
                    cited_by_current_draft=True,
                    page_number=span.page_number,
                    section=span.section,
                    page_char_start=span.page_char_start,
                    page_char_end=span.page_char_end,
                    locator=span.locator,
                ) for span in spans],
                paper_versions=[ManifestPaperVersion(
                    id=version.id,
                    study_id=study.id,
                    paper_id=paper.id,
                    paper_title=paper.title,
                    content_hash=version.content_hash,
                    source_name=version.source_name,
                    source_url="https://example.org/fixture.pdf",
                    license=version.license,
                    version_label=version.version_label,
                    retrieved_at=version.retrieved_at,
                    used_by=["cited_in_current_draft"],
                )],
                current_draft=ManifestDraft(
                    id=draft.id,
                    status="success",
                    input_hash=draft.input_hash,
                    claim_generation_id=generation.id,
                    schema_version=draft.schema_version,
                    prompt_version=draft.prompt_version,
                    model_name=draft.llm_model,
                    created_at=draft.created_at,
                    title=document["title"],
                    research_question=question,
                    sentences=[
                        ManifestDraftSentence(
                            section_index=0, paragraph_index=0, sentence_index=index,
                            text=texts[index], claim_ids=[claim.id],
                            evidence_ids=[self.evidence_ids[0] if index == 0 else self.evidence_ids[1]],
                        ) for index in range(3)
                    ],
                ),
                audit_summary=ManifestAuditSummary(
                    citation=ManifestStageAudit(
                        id=citation_audit.id, status="passed", input_hash=citation_audit.input_hash,
                        auditor_version=citation_audit.auditor_version,
                        sentence_count=3, issue_count=0, citation_count=3,
                        unique_evidence_count=2,
                        sentences=[ManifestSentenceAudit(
                            section_index=0, paragraph_index=0, sentence_index=index,
                            status="passed", verdict=None,
                            input_hash=citation_sentences[index].input_hash,
                            issues=[],
                        ) for index in range(3)],
                    ),
                    semantic=ManifestStageAudit(
                        id=citation_audit.id, status="complete", sentence_count=3,
                        issue_count=0, verdict_counts={"pass": 1, "revise": 1, "reject": 1},
                        prompt_versions=["semantic-v1"],
                        sentences=semantic_sentences,
                    ),
                    omission=ManifestStageAudit(
                        id=citation_audit.id, status="complete", sentence_count=3,
                        issue_count=0, verdict_counts={"pass": 3},
                        prompt_versions=["omission-v1"],
                        sentences=omission_sentences,
                    ),
                ),
            )
            manifest_json = manifest.model_dump(mode="json")
            manifest_hash = hashlib.sha256(json.dumps(
                manifest_json, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
            ).encode("utf-8")).hexdigest()
            session.add(RunManifestRecord(
                id=self.manifest_id,
                run_id=run.id,
                manifest_version=1,
                content_hash=manifest_hash,
                manifest_json=manifest_json,
            ))

    def tearDown(self) -> None:
        self.engine.dispose()

    def test_exports_keep_original_text_audit_states_and_citation_chain(self) -> None:
        first = build_grounded_review_export(self.run_id, session_factory=self.factory)
        again = build_grounded_review_export(self.run_id, session_factory=self.factory)
        self.assertEqual(first.markdown, again.markdown)
        self.assertEqual(first.json_text, again.json_text)
        self.assertIn(str(self.manifest_id), first.json_text)
        self.assertIn("PaperVersion", first.markdown)
        self.assertIn("EvidenceSpan", first.markdown)
        self.assertIn("Study", first.markdown)
        self.assertIn("PDF p. 7", first.markdown)
        self.assertIn("Semantic: revise", first.markdown)
        self.assertIn("Semantic: reject", first.markdown)
        self.assertIn("REJECTED: do not treat as a supported fact", first.markdown)
        self.assertIn("Suggested revision (not applied)", first.markdown)
        self.assertNotIn("do-not-export", first.markdown)

        payload = json.loads(first.json_text)
        self.assertEqual(payload["manifest_id"], str(self.manifest_id))
        sentences = payload["sections"][0]["paragraphs"][0]["sentences"]
        self.assertEqual([item["semantic_audit"]["verdict"] for item in sentences],
                         ["pass", "revise", "reject"])
        self.assertEqual(sentences[1]["original_text"], "The effect was small.")
        self.assertEqual(sentences[1]["semantic_audit"]["minimal_revision"],
                         "A narrower wording is safer.")
        self.assertIn("original wording is retained", sentences[1]["export_warning"])
        self.assertIn("do not treat it as a supported fact", sentences[2]["export_warning"])
        self.assertEqual(len(payload["evidence"]), 2)
        self.assertEqual(payload["evidence"][0]["page"], 7)
        self.assertTrue(payload["evidence"][0]["chunk_id"])

    def test_writer_outputs_same_report_bytes_on_repeated_exports(self) -> None:
        result = build_grounded_review_export(self.run_id, session_factory=self.factory)
        with tempfile.TemporaryDirectory() as directory:
            markdown_path, json_path = write_grounded_review_export(result, directory)
            first_markdown = markdown_path.read_bytes()
            first_json = json_path.read_bytes()
            write_grounded_review_export(
                build_grounded_review_export(self.run_id, session_factory=self.factory), directory
            )
            self.assertEqual(first_markdown, markdown_path.read_bytes())
            self.assertEqual(first_json, json_path.read_bytes())
            self.assertEqual(Path(directory, "report.md"), markdown_path)
            self.assertEqual(Path(directory, "report.json"), json_path)


if __name__ == "__main__":
    unittest.main()
