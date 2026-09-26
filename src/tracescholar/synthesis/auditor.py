"""Deterministic, per-sentence citation-chain audit; no LLM or prose edits."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from tracescholar.database import get_session_factory, session_scope
from tracescholar.models import (
    CanonicalStudy, CitationAudit, CitationAuditSentenceLink,
    CitationSentenceAudit, Chunk, Claim, ClaimGeneration, EvidenceExtraction,
    EvidenceSpan, Paper, PaperVersion, ParsedPage, StudyPaper, SynthesisDraft,
)
from tracescholar.synthesis.schemas import SYNTHESIS_SCHEMA_VERSION, SynthesisDocument


AUDITOR_VERSION = "citation-chain-v3-sentence-persisted"


@dataclass(frozen=True, slots=True)
class CitationSentenceResult:
    sentence_audit_id: uuid.UUID
    section_index: int
    paragraph_index: int
    sentence_index: int
    input_hash: str
    status: str
    issues: tuple[dict[str, Any], ...]
    created: bool


@dataclass(frozen=True, slots=True)
class CitationAuditResult:
    audit_id: uuid.UUID
    draft_id: uuid.UUID
    run_id: uuid.UUID
    status: str
    sentence_count: int
    citation_count: int
    unique_evidence_count: int
    issues: tuple[dict[str, Any], ...]
    created: bool
    sentence_results: tuple[CitationSentenceResult, ...]


def _fingerprint(snapshot: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        snapshot, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _claim_snapshot(session: Session, claim_id: uuid.UUID) -> dict[str, Any] | None:
    claim = session.get(Claim, claim_id)
    if claim is None:
        return None
    generation = session.get(ClaimGeneration, claim.generation_id)
    return {
        "id": str(claim.id),
        "generation_id": str(claim.generation_id),
        "generation_run_id": str(generation.run_id) if generation else None,
        "sub_question_index": claim.sub_question_index,
        "statement": claim.statement,
        "scope_kind": claim.scope_kind,
        "basis_study_id": str(claim.basis_study_id) if claim.basis_study_id else None,
        "basis_chunk_id": str(claim.basis_chunk_id),
        "basis_quote": claim.basis_quote,
        "basis_chunk_char_start": claim.basis_chunk_char_start,
        "basis_chunk_char_end": claim.basis_chunk_char_end,
    }


def _evidence_snapshot(session: Session, evidence_id: uuid.UUID) -> dict[str, Any] | None:
    span = session.get(EvidenceSpan, evidence_id)
    if span is None:
        return None
    extraction = session.get(EvidenceExtraction, span.extraction_id)
    version = session.get(PaperVersion, span.paper_version_id)
    paper = session.get(Paper, version.paper_id) if version else None
    chunk = session.get(Chunk, span.chunk_id)
    study = session.get(CanonicalStudy, span.study_id)
    page = session.scalar(select(ParsedPage).where(
        ParsedPage.paper_version_id == span.paper_version_id,
        ParsedPage.page_number == span.page_number))
    membership = session.scalar(select(StudyPaper).where(
        StudyPaper.study_id == span.study_id,
        StudyPaper.paper_id == version.paper_id)) if version else None
    return {
        "evidence_span": {
            "id": str(span.id), "extraction_id": str(span.extraction_id),
            "study_id": str(span.study_id),
            "paper_version_id": str(span.paper_version_id),
            "chunk_id": str(span.chunk_id), "quote": span.quote,
            "stance": span.stance, "confidence": span.confidence,
            "rationale": span.rationale, "study_context": span.study_context,
            "limitations": span.limitations, "page_number": span.page_number,
            "section": span.section, "chunk_char_start": span.chunk_char_start,
            "chunk_char_end": span.chunk_char_end,
            "page_char_start": span.page_char_start,
            "page_char_end": span.page_char_end, "locator": span.locator,
        },
        "extraction": ({
            "id": str(extraction.id), "claim_id": str(extraction.claim_id),
            "study_id": str(extraction.study_id),
            "paper_version_id": str(extraction.paper_version_id),
            "status": extraction.status, "disposition": extraction.disposition,
        } if extraction else None),
        "study": ({
            "id": str(study.id),
            "canonical_paper_id": str(study.canonical_paper_id),
            "canonical_reason": study.canonical_reason,
        } if study else None),
        "study_paper_membership": ({
            "id": str(membership.id), "study_id": str(membership.study_id),
            "paper_id": str(membership.paper_id),
            "publication_role": membership.publication_role,
            "relationship_reason": membership.relationship_reason,
        } if membership else None),
        "paper_version": ({
            "id": str(version.id), "paper_id": str(version.paper_id),
            "content_hash": version.content_hash, "source_url": version.source_url,
            "source_name": version.source_name, "license": version.license,
            "version_label": version.version_label,
            "retrieved_at": version.retrieved_at.isoformat(),
        } if version else None),
        "paper": ({
            "id": str(paper.id), "title": paper.title, "doi": paper.doi,
            "arxiv_id": paper.arxiv_id, "year": paper.year,
        } if paper else None),
        "chunk": ({
            "id": str(chunk.id), "paper_version_id": str(chunk.paper_version_id),
            "ordinal": chunk.ordinal, "page_start": chunk.page_start,
            "page_end": chunk.page_end, "section": chunk.section,
            "text": chunk.text, "document_char_start": chunk.document_char_start,
            "document_char_end": chunk.document_char_end, "locator": chunk.locator,
            "parser_version": chunk.parser_version,
        } if chunk else None),
        "parsed_page": ({
            "id": str(page.id), "paper_version_id": str(page.paper_version_id),
            "page_number": page.page_number, "text": page.text,
        } if page else None),
    }


def _chain_issues(lineage: dict[str, Any], cited_claims: set[str],
                  claims: dict[str, Any], draft: SynthesisDraft) -> list[str]:
    """Return structural failures, never judgments about sentence meaning."""
    problems: list[str] = []
    extraction = lineage["extraction"]
    span = lineage["evidence_span"]
    if extraction is None:
        return ["missing_extraction"]
    claim_id = extraction["claim_id"]
    if claim_id not in cited_claims:
        problems.append("evidence_claim_not_cited_in_sentence")
    linked_claim = claims.get(claim_id)
    if linked_claim is None:
        problems.append("missing_evidence_claim")
    elif linked_claim["generation_id"] != str(draft.claim_generation_id) or \
            linked_claim["generation_run_id"] != str(draft.run_id):
        problems.append("evidence_claim_outside_draft")
    if extraction["status"] != "success" or extraction["disposition"] != "evidence":
        problems.append("extraction_not_successful_evidence")
    if extraction["study_id"] != span["study_id"] or \
            extraction["paper_version_id"] != span["paper_version_id"]:
        problems.append("span_extraction_mismatch")
    if lineage["study"] is None:
        problems.append("missing_study")
    version = lineage["paper_version"]
    if version is None:
        problems.append("missing_paper_version")
    elif lineage["study_paper_membership"] is None:
        problems.append("paper_version_outside_study")
    chunk = lineage["chunk"]
    if chunk is None:
        problems.append("missing_chunk")
    else:
        if chunk["paper_version_id"] != span["paper_version_id"]:
            problems.append("chunk_version_mismatch")
        if chunk["page_start"] != span["page_number"] or \
                chunk["page_end"] != span["page_number"] or \
                chunk["section"] != span["section"]:
            problems.append("chunk_page_or_section_mismatch")
    page = lineage["parsed_page"]
    if page is None:
        problems.append("missing_parsed_page")
    if chunk is not None and page is not None:
        locator = chunk["locator"]
        start = locator.get("char_start") if isinstance(locator, dict) else None
        end = locator.get("char_end") if isinstance(locator, dict) else None
        if not isinstance(locator, dict) or locator.get("page") != span["page_number"]:
            problems.append("chunk_page_locator_mismatch")
        valid_chunk_range = isinstance(start, int) and isinstance(end, int) and \
            0 <= start < end <= len(page["text"])
        if not valid_chunk_range or page["text"][start:end] != chunk["text"]:
            problems.append("chunk_page_locator_mismatch")
        local_start, local_end = span["chunk_char_start"], span["chunk_char_end"]
        valid_quote_range = 0 <= local_start < local_end <= len(chunk["text"])
        if not valid_quote_range or chunk["text"][local_start:local_end] != span["quote"]:
            problems.append("quote_chunk_offset_mismatch")
        page_start, page_end = span["page_char_start"], span["page_char_end"]
        if not (0 <= page_start < page_end <= len(page["text"])) or \
                page["text"][page_start:page_end] != span["quote"] or \
                not valid_chunk_range or page_start != start + local_start or \
                page_end != start + local_end:
            problems.append("quote_page_offset_mismatch")
        span_locator = span["locator"]
        if not isinstance(span_locator, dict) or \
                span_locator.get("quote_char_start") != page_start or \
                span_locator.get("quote_char_end") != page_end or \
                ("page" in span_locator and span_locator["page"] != span["page_number"]):
            problems.append("span_locator_mismatch")
    return problems


def _sentence_issues(snapshot: dict[str, Any], draft: SynthesisDraft) -> list[dict[str, Any]]:
    position = snapshot["position"]
    location = {"section_index": position[0], "paragraph_index": position[1],
                "sentence_index": position[2]}
    claims = snapshot["claims"]
    issues: list[dict[str, Any]] = []
    for claim_id, claim in claims.items():
        if claim is None:
            issues.append({**location, "code": "missing_claim", "claim_id": claim_id})
        elif claim["generation_id"] != str(draft.claim_generation_id) or \
                claim["generation_run_id"] != str(draft.run_id):
            issues.append({**location, "code": "claim_outside_draft", "claim_id": claim_id})

    cited_claims = set(claims)
    evidenced_claims: set[str] = set()
    for evidence_id, lineage in snapshot["lineage"].items():
        evidence_location = {**location, "evidence_id": evidence_id}
        if lineage is None:
            issues.append({**evidence_location, "code": "missing_evidence_span"})
            continue
        extraction = lineage["extraction"]
        if extraction is not None:
            evidenced_claims.add(extraction["claim_id"])
        for code in _chain_issues(lineage, cited_claims, claims, draft):
            issues.append({**evidence_location, "code": code})

    for claim_id in claims:
        if claim_id not in evidenced_claims:
            issues.append({**location, "code": "claim_without_evidence", "claim_id": claim_id})
    return issues


def _sentence_result(row: CitationSentenceAudit, *, created: bool) -> CitationSentenceResult:
    return CitationSentenceResult(
        sentence_audit_id=row.id, section_index=row.section_index,
        paragraph_index=row.paragraph_index, sentence_index=row.sentence_index,
        input_hash=row.input_hash, status=row.status,
        issues=tuple(row.issues_json), created=created,
    )


def _result(row: CitationAudit, *, created: bool,
            sentence_results: tuple[CitationSentenceResult, ...]) -> CitationAuditResult:
    return CitationAuditResult(
        audit_id=row.id, draft_id=row.draft_id, run_id=row.run_id,
        status=row.status, sentence_count=row.sentence_count,
        citation_count=row.citation_count,
        unique_evidence_count=row.unique_evidence_count,
        issues=tuple(row.issues_json), created=created,
        sentence_results=sentence_results,
    )


def audit_synthesis_citations(
    run_id: uuid.UUID, *, draft_id: uuid.UUID | None = None,
    session_factory: sessionmaker[Session] | None = None,
) -> CitationAuditResult:
    """Audit and persist each sentence, then derive the draft-level summary."""
    factory = session_factory or get_session_factory()
    with session_scope(factory) as session:
        statement = select(SynthesisDraft).where(SynthesisDraft.run_id == run_id)
        if draft_id is not None:
            statement = statement.where(SynthesisDraft.id == draft_id)
        else:
            statement = statement.where(SynthesisDraft.status == "success")
        draft = session.scalar(statement.order_by(
            SynthesisDraft.created_at.desc(), SynthesisDraft.id.desc()))
        if draft is None:
            raise LookupError("No matching synthesis draft exists for this ResearchRun")

        global_issues: list[dict[str, Any]] = []
        document: SynthesisDocument | None = None
        if draft.schema_version != SYNTHESIS_SCHEMA_VERSION:
            global_issues.append({"code": "unsupported_schema_version"})
        elif draft.status != "success" or draft.document_json is None:
            global_issues.append({"code": "draft_not_successful"})
        else:
            try:
                document = SynthesisDocument.model_validate_json(json.dumps(draft.document_json))
            except ValidationError:
                global_issues.append({"code": "invalid_document_schema"})

        sentence_rows: list[tuple[CitationSentenceAudit, bool]] = []
        if document is not None:
            for section_index, section in enumerate(document.sections):
                for paragraph_index, paragraph in enumerate(section.paragraphs):
                    for sentence_index, sentence in enumerate(paragraph.sentences):
                        claim_ids = [str(value) for value in sentence.claim_ids]
                        cited_evidence_ids = [str(value) for value in sentence.evidence_ids]
                        claims = {
                            claim_id: _claim_snapshot(session, uuid.UUID(claim_id))
                            for claim_id in claim_ids
                        }
                        lineage = {
                            evidence_id: _evidence_snapshot(session, uuid.UUID(evidence_id))
                            for evidence_id in cited_evidence_ids
                        }
                        position = [section_index, paragraph_index, sentence_index]
                        snapshot = {
                            "auditor_version": AUDITOR_VERSION,
                            "run_id": str(draft.run_id), "draft_id": str(draft.id),
                            "claim_generation_id": str(draft.claim_generation_id),
                            "schema_version": draft.schema_version,
                            "position": position, "sentence": sentence.text,
                            "claim_ids": claim_ids, "claims": claims,
                            "evidence_ids": cited_evidence_ids, "lineage": lineage,
                        }
                        input_hash = _fingerprint(snapshot)
                        row = session.scalar(select(CitationSentenceAudit).where(
                            CitationSentenceAudit.draft_id == draft.id,
                            CitationSentenceAudit.auditor_version == AUDITOR_VERSION,
                            CitationSentenceAudit.section_index == section_index,
                            CitationSentenceAudit.paragraph_index == paragraph_index,
                            CitationSentenceAudit.sentence_index == sentence_index,
                            CitationSentenceAudit.input_hash == input_hash,
                        ))
                        created = row is None
                        if row is None:
                            sentence_issues = _sentence_issues(snapshot, draft)
                            row = CitationSentenceAudit(
                                run_id=draft.run_id, draft_id=draft.id,
                                auditor_version=AUDITOR_VERSION,
                                section_index=section_index,
                                paragraph_index=paragraph_index,
                                sentence_index=sentence_index,
                                input_hash=input_hash, input_snapshot=snapshot,
                                status="failed" if sentence_issues else "passed",
                                issues_json=sentence_issues,
                            )
                            session.add(row)
                        session.flush()
                        sentence_rows.append((row, created))

        sentence_results = tuple(
            _sentence_result(row, created=created) for row, created in sentence_rows)
        sentence_issues = [issue for row, _ in sentence_rows for issue in row.issues_json]
        issues = [*global_issues, *sentence_issues]
        sentence_count = len(sentence_rows)
        citation_count = sum(
            len(row.input_snapshot["evidence_ids"]) for row, _ in sentence_rows)
        evidence_ids = {
            evidence_id for row, _ in sentence_rows
            for evidence_id in row.input_snapshot["evidence_ids"]
        }
        document_metadata = None
        if document is not None:
            document_metadata = {
                "title": document.title,
                "research_question": document.research_question,
                "section_headings": [section.heading for section in document.sections],
            }
        aggregate_snapshot = {
            "auditor_version": AUDITOR_VERSION, "run_id": str(draft.run_id),
            "draft_id": str(draft.id), "schema_version": draft.schema_version,
            "draft_status": draft.status, "document_metadata": document_metadata,
            "draft_content_hash": _fingerprint({"document_json": draft.document_json}),
            "global_issues": global_issues,
            "sentence_audits": [{
                "id": str(row.id), "position": [row.section_index,
                                                     row.paragraph_index,
                                                     row.sentence_index],
                "input_hash": row.input_hash,
            } for row, _ in sentence_rows],
        }
        aggregate_hash = _fingerprint(aggregate_snapshot)
        existing = session.scalar(select(CitationAudit).where(
            CitationAudit.draft_id == draft.id,
            CitationAudit.auditor_version == AUDITOR_VERSION,
            CitationAudit.input_hash == aggregate_hash,
        ))
        if existing is not None:
            # A previous aggregate links precisely these already-reused sentence rows.
            return _result(existing, created=False, sentence_results=sentence_results)

        row = CitationAudit(
            run_id=draft.run_id, draft_id=draft.id,
            auditor_version=AUDITOR_VERSION, input_hash=aggregate_hash,
            status="failed" if issues else "passed",
            sentence_count=sentence_count, citation_count=citation_count,
            unique_evidence_count=len(evidence_ids), issues_json=issues,
        )
        session.add(row)
        session.flush()
        for sentence_row, _ in sentence_rows:
            session.add(CitationAuditSentenceLink(
                citation_audit_id=row.id, sentence_audit_id=sentence_row.id,
            ))
        session.flush()
        return _result(row, created=True, sentence_results=sentence_results)
