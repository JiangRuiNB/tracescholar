"""Read saved research artifacts and deterministically emit Markdown and JSON."""

from __future__ import annotations

import json
import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from tracescholar.database import get_session_factory
from tracescholar.exports.schemas import (
    ExportAuditStatus,
    ExportEvidence,
    ExportParagraph,
    ExportSection,
    ExportSentence,
    ExportStageAudit,
    ExportAuditSummary,
    GroundedReviewExport,
)
from tracescholar.manifests.schemas import ManifestSentenceAudit, RunManifest
from tracescholar.models import (
    CitationAudit,
    CitationAuditSentenceLink,
    CitationSentenceAudit,
    EvidenceSpan,
    OmissionAudit,
    PaperVersion,
    RunManifestRecord,
    SemanticCitationAudit,
    StudyPaper,
    SynthesisDraft,
)
from tracescholar.synthesis.renderer import CitationSource, render_markdown
from tracescholar.synthesis.renderer import _escape_markdown
from tracescholar.synthesis.schemas import (
    SynthesisDocument,
    SynthesisParagraph,
    SynthesisSection,
    SynthesisSentence,
    SYNTHESIS_SCHEMA_VERSION,
)
from tracescholar.synthesis.validation import (
    SynthesisValidationError,
    validate_synthesis_references,
)


@dataclass(frozen=True, slots=True)
class GroundedReviewExportResult:
    """Validated export content and its immutable run/draft identity."""

    run_id: uuid.UUID
    manifest_id: uuid.UUID
    draft_id: uuid.UUID
    content: GroundedReviewExport
    markdown: str
    json_text: str


def _manifest_audit_sentence_map(
    rows: list[ManifestSentenceAudit] | None,
) -> dict[tuple[int, int, int], ManifestSentenceAudit]:
    return {
        (row.section_index, row.paragraph_index, row.sentence_index): row
        for row in (rows or [])
    }


def _safe_source_url(value: str) -> str:
    """Do not copy credentials, signed query strings, or fragments into exports."""
    try:
        parts = urlsplit(value)
        host = parts.hostname
        port = parts.port
    except ValueError:
        return ""
    if parts.scheme not in {"http", "https"} or not host:
        return ""
    if port:
        host = f"{host}:{port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def _resolve_sentence_record(
    rows: list[Any],
    position: tuple[int, int, int],
    snapshot: ManifestSentenceAudit | None,
) -> Any | None:
    if snapshot is None:
        return None
    for row in rows:
        if (row.section_index, row.paragraph_index, row.sentence_index) == position \
                and row.input_hash == snapshot.input_hash:
            return row
    return None


def _audit_status(
    stage_snapshot: ManifestSentenceAudit | None,
    record: Any | None,
    *,
    default_status: str = "not_recorded",
) -> ExportAuditStatus:
    if stage_snapshot is None:
        return ExportAuditStatus(status=default_status)
    if record is None:
        raise SynthesisValidationError(
            "The selected RunManifest audit snapshot no longer matches its persisted sentence record"
        )
    if record.status != stage_snapshot.status:
        raise SynthesisValidationError("RunManifest sentence audit status does not match its saved record")
    if hasattr(record, "verdict") and record.verdict != stage_snapshot.verdict:
        raise SynthesisValidationError("RunManifest sentence audit verdict does not match its saved record")
    if hasattr(record, "prompt_version") and record.prompt_version != stage_snapshot.prompt_version:
        raise SynthesisValidationError("RunManifest sentence audit prompt does not match its saved record")
    if hasattr(record, "llm_model") and record.llm_model != stage_snapshot.model_name:
        raise SynthesisValidationError("RunManifest sentence audit model does not match its saved record")
    if hasattr(record, "rationale") and record.rationale != stage_snapshot.rationale:
        raise SynthesisValidationError("RunManifest sentence audit rationale does not match its saved record")
    if hasattr(record, "minimal_revision") and record.minimal_revision != stage_snapshot.minimal_revision:
        raise SynthesisValidationError("RunManifest sentence audit suggestion does not match its saved record")
    if hasattr(record, "issues_json") and record.issues_json != stage_snapshot.issues:
        raise SynthesisValidationError("RunManifest citation issues do not match their saved record")
    if hasattr(record, "omitted_evidence_ids") and [uuid.UUID(value) for value in record.omitted_evidence_ids] != stage_snapshot.omitted_evidence_ids:
        raise SynthesisValidationError("RunManifest omitted Evidence IDs do not match their saved record")
    if hasattr(record, "impact_types_json") and record.impact_types_json != stage_snapshot.impact_types:
        raise SynthesisValidationError("RunManifest omission impacts do not match their saved record")
    return ExportAuditStatus(
        status=stage_snapshot.status,
        verdict=stage_snapshot.verdict,
        record_id=record.id,
        input_hash=stage_snapshot.input_hash,
        entailment=getattr(record, "entailment", None),
        scope=getattr(record, "scope", None),
        strength=getattr(record, "strength", None),
        model_name=stage_snapshot.model_name,
        prompt_version=stage_snapshot.prompt_version,
        rationale=stage_snapshot.rationale,
        minimal_revision=stage_snapshot.minimal_revision,
        issues=stage_snapshot.issues,
        omitted_evidence_ids=stage_snapshot.omitted_evidence_ids,
        impact_types=stage_snapshot.impact_types,
    )


def _citation_source(session: Session, span: EvidenceSpan) -> tuple[CitationSource, ExportEvidence]:
    extraction = span.extraction
    if extraction is None or extraction.claim is None:
        raise SynthesisValidationError("EvidenceSpan is missing its Claim/Extraction relation")
    version = session.get(PaperVersion, span.paper_version_id)
    if version is None:
        raise SynthesisValidationError("EvidenceSpan PaperVersion does not exist")
    if extraction.study_id != span.study_id or extraction.paper_version_id != span.paper_version_id:
        raise SynthesisValidationError("EvidenceSpan and extraction provenance disagree")
    if not session.scalar(select(StudyPaper.id).where(
        StudyPaper.study_id == span.study_id,
        StudyPaper.paper_id == version.paper_id,
    )):
        raise SynthesisValidationError("EvidenceSpan PaperVersion is outside its Canonical Study")
    paper = version.paper
    if paper is None:
        raise SynthesisValidationError("EvidenceSpan PaperVersion has no Paper metadata")
    citation = CitationSource(
        evidence_id=span.id,
        claim_id=extraction.claim.id,
        claim_statement=extraction.claim.statement,
        study_id=span.study_id,
        paper_version_id=version.id,
        paper_title=paper.title,
        authors=tuple(paper.authors or []),
        year=paper.year,
        venue=paper.venue,
        doi=paper.doi,
        arxiv_id=paper.arxiv_id,
        version_label=version.version_label,
        source_url=_safe_source_url(version.source_url),
        page=span.page_number,
        section=span.section,
        quote=span.quote,
        stance=span.stance,
    )
    exported = ExportEvidence(
        evidence_id=span.id,
        claim_id=extraction.claim.id,
        claim_statement=extraction.claim.statement,
        study_id=span.study_id,
        paper_version_id=version.id,
        chunk_id=span.chunk_id,
        paper_title=paper.title,
        authors=list(paper.authors or []),
        year=paper.year,
        venue=paper.venue,
        doi=paper.doi,
        arxiv_id=paper.arxiv_id,
        content_hash=version.content_hash,
        version_label=version.version_label,
        source_url=_safe_source_url(version.source_url),
        page=span.page_number,
        section=span.section,
        quote=span.quote,
        stance=span.stance,
        locator=span.locator,
    )
    return citation, exported


def _stage_audit(manifest: RunManifest, name: str) -> ExportStageAudit:
    stage = getattr(manifest.audit_summary, name)
    if stage is None:
        return ExportStageAudit(status="not_recorded")
    return ExportStageAudit(
        audit_id=stage.id,
        status=stage.status,
        sentence_count=stage.sentence_count,
        issue_count=stage.issue_count,
        citation_count=stage.citation_count,
        unique_evidence_count=stage.unique_evidence_count,
        verdict_counts=stage.verdict_counts,
    )


def _annotated_document(
    document: SynthesisDocument,
    sentence_statuses: dict[tuple[int, int, int], tuple[str, str, str]],
) -> SynthesisDocument:
    sections: list[SynthesisSection] = []
    for section_index, section in enumerate(document.sections):
        paragraphs: list[SynthesisParagraph] = []
        for paragraph_index, paragraph in enumerate(section.paragraphs):
            sentences: list[SynthesisSentence] = []
            for sentence_index, sentence in enumerate(paragraph.sentences):
                citation, semantic, omission = sentence_statuses[
                    (section_index, paragraph_index, sentence_index)
                ]
                label = (f"〔Citation: {citation} | Semantic: {semantic} | "
                         f"Omission: {omission}〕")
                if semantic == "reject":
                    label += "〔REJECTED: do not treat as a supported fact〕"
                elif semantic == "revise":
                    label += "〔Human review required; original wording retained〕"
                sentences.append(sentence.model_copy(update={"text": f"{sentence.text} {label}"}))
            paragraphs.append(paragraph.model_copy(update={"sentences": sentences}))
        sections.append(section.model_copy(update={"paragraphs": paragraphs}))
    return document.model_copy(update={"sections": sections})


def build_grounded_review_export(
    run_id: uuid.UUID,
    *,
    manifest_id: uuid.UUID | None = None,
    session_factory: sessionmaker[Session] | None = None,
) -> GroundedReviewExportResult:
    """Build deterministic report content from a saved draft, audits, and manifest."""
    factory = session_factory or get_session_factory()
    with factory() as session:
        if manifest_id is None:
            manifest_row = session.scalar(select(RunManifestRecord).where(
                RunManifestRecord.run_id == run_id
            ).order_by(RunManifestRecord.created_at.desc(), RunManifestRecord.id.desc()).limit(1))
        else:
            manifest_row = session.get(RunManifestRecord, manifest_id)
        if manifest_row is None or manifest_row.run_id != run_id:
            raise LookupError("No matching RunManifest exists for this ResearchRun")
        manifest = RunManifest.model_validate_json(
            json.dumps(manifest_row.manifest_json, ensure_ascii=False)
        )
        canonical_manifest_json = json.dumps(
            manifest.model_dump(mode="json"),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if hashlib.sha256(canonical_manifest_json.encode("utf-8")).hexdigest() != manifest_row.content_hash:
            raise SynthesisValidationError("RunManifest content hash does not match its stored JSON")
        if manifest.manifest_version != manifest_row.manifest_version:
            raise SynthesisValidationError("RunManifest schema version does not match its database record")
        if manifest.current_draft is None:
            raise LookupError("The selected RunManifest has no successful synthesis draft")
        draft = session.get(SynthesisDraft, manifest.current_draft.id)
        if draft is None or draft.run_id != run_id or draft.status != "success" \
                or draft.document_json is None:
            raise SynthesisValidationError("The RunManifest draft is missing or no longer successful")
        if draft.schema_version != SYNTHESIS_SCHEMA_VERSION:
            raise ValueError("Unsupported SynthesisDocument schema version")
        if (draft.input_hash != manifest.current_draft.input_hash
                or draft.schema_version != manifest.current_draft.schema_version
                or draft.prompt_version != manifest.current_draft.prompt_version
                or draft.llm_model != manifest.current_draft.model_name):
            raise SynthesisValidationError("RunManifest draft metadata does not match the saved draft")
        document = SynthesisDocument.model_validate_json(json.dumps(draft.document_json))
        validate_synthesis_references(
            document, draft.input_snapshot["ledger"], draft.input_snapshot["research_question"]
        )

        if (document.title != manifest.current_draft.title
                or document.research_question != manifest.current_draft.research_question):
            raise SynthesisValidationError("RunManifest and stored draft content disagree")
        manifest_sentences = {
            (item.section_index, item.paragraph_index, item.sentence_index): item
            for item in manifest.current_draft.sentences
        }
        document_positions: set[tuple[int, int, int]] = set()
        cited_ids: set[uuid.UUID] = set()
        for section_index, section in enumerate(document.sections):
            for paragraph_index, paragraph in enumerate(section.paragraphs):
                for sentence_index, sentence in enumerate(paragraph.sentences):
                    position = (section_index, paragraph_index, sentence_index)
                    document_positions.add(position)
                    snapshot = manifest_sentences.get(position)
                    if snapshot is None or snapshot.text != sentence.text \
                            or snapshot.claim_ids != sentence.claim_ids \
                            or snapshot.evidence_ids != sentence.evidence_ids:
                        raise SynthesisValidationError("RunManifest sentence snapshot does not match the saved draft")
                    cited_ids.update(sentence.evidence_ids)
        if document_positions != set(manifest_sentences):
            raise SynthesisValidationError("RunManifest has a different sentence inventory than its draft")

        citation_audit = None
        citation_summary = manifest.audit_summary.citation
        semantic_summary = manifest.audit_summary.semantic
        omission_summary = manifest.audit_summary.omission
        if citation_summary is None or semantic_summary is None or omission_summary is None:
            raise LookupError("Export requires Citation, Semantic, and Omission audit snapshots in the selected RunManifest")
        citation_audit = session.get(CitationAudit, citation_summary.id)
        if citation_audit is None or citation_audit.run_id != run_id \
                or citation_audit.draft_id != draft.id:
            raise SynthesisValidationError("RunManifest Citation Audit does not match the selected draft")
        if (citation_audit.input_hash != citation_summary.input_hash
                or citation_audit.status != citation_summary.status
                or citation_audit.auditor_version != citation_summary.auditor_version):
            raise SynthesisValidationError("RunManifest Citation Audit snapshot is stale")
        if citation_audit is not None:
            for audit_stage in (manifest.audit_summary.semantic, manifest.audit_summary.omission):
                if audit_stage is not None and audit_stage.id != citation_audit.id:
                    raise SynthesisValidationError("RunManifest sentence audit is linked to another Citation Audit")

        citation_snapshot = _manifest_audit_sentence_map(citation_summary.sentences if citation_summary else None)
        semantic_snapshot = _manifest_audit_sentence_map(semantic_summary.sentences if semantic_summary else None)
        omission_snapshot = _manifest_audit_sentence_map(omission_summary.sentences if omission_summary else None)
        positions = sorted(document_positions)
        for stage_name, snapshot in (
            ("Citation", citation_snapshot),
            ("Semantic", semantic_snapshot),
            ("Omission", omission_snapshot),
        ):
            missing_positions = set(positions) - set(snapshot)
            if missing_positions:
                raise LookupError(f"{stage_name} audit is missing {len(missing_positions)} draft sentence(s)")

        citation_rows: list[CitationSentenceAudit] = []
        if citation_audit is not None:
            linked_ids = session.scalars(select(CitationAuditSentenceLink.sentence_audit_id).where(
                CitationAuditSentenceLink.citation_audit_id == citation_audit.id
            )).all()
            if linked_ids:
                citation_rows = session.scalars(select(CitationSentenceAudit).where(
                    CitationSentenceAudit.id.in_(linked_ids)
                )).all()
        semantic_rows: list[SemanticCitationAudit] = []
        omission_rows: list[OmissionAudit] = []
        if citation_audit is not None:
            semantic_rows = session.scalars(select(SemanticCitationAudit).where(
                SemanticCitationAudit.run_id == run_id,
                SemanticCitationAudit.draft_id == draft.id,
                SemanticCitationAudit.citation_audit_id == citation_audit.id,
            )).all()
            omission_rows = session.scalars(select(OmissionAudit).where(
                OmissionAudit.run_id == run_id,
                OmissionAudit.draft_id == draft.id,
                OmissionAudit.citation_audit_id == citation_audit.id,
            )).all()

        audit_data: dict[tuple[int, int, int], tuple[ExportAuditStatus, ExportAuditStatus, ExportAuditStatus]] = {}
        for position in positions:
            citation_record = _resolve_sentence_record(citation_rows, position, citation_snapshot.get(position))
            semantic_record = _resolve_sentence_record(semantic_rows, position, semantic_snapshot.get(position))
            omission_record = _resolve_sentence_record(omission_rows, position, omission_snapshot.get(position))
            audit_data[position] = (
                _audit_status(citation_snapshot.get(position), citation_record),
                _audit_status(semantic_snapshot.get(position), semantic_record),
                _audit_status(omission_snapshot.get(position), omission_record),
            )

        spans = session.scalars(select(EvidenceSpan).where(EvidenceSpan.id.in_(cited_ids))).all()
        if {span.id for span in spans} != cited_ids:
            raise SynthesisValidationError("A draft EvidenceSpan reference is missing")
        manifest_evidence = {item.id: item for item in manifest.evidence_spans}
        citation_sources: dict[uuid.UUID, CitationSource] = {}
        evidence_records: list[ExportEvidence] = []
        for span in sorted(spans, key=lambda row: str(row.id)):
            citation_source, exported_evidence = _citation_source(session, span)
            snapshot = manifest_evidence.get(span.id)
            if snapshot is None or (
                snapshot.claim_id != exported_evidence.claim_id
                or snapshot.study_id != span.study_id
                or snapshot.paper_version_id != span.paper_version_id
                or snapshot.chunk_id != span.chunk_id
                or snapshot.quote != span.quote
                or snapshot.stance != span.stance
                or snapshot.page_number != span.page_number
                or snapshot.section != span.section
                or snapshot.page_char_start != span.page_char_start
                or snapshot.page_char_end != span.page_char_end
                or snapshot.locator != span.locator
                or not snapshot.cited_by_current_draft
            ):
                raise SynthesisValidationError("RunManifest EvidenceSpan snapshot does not match the saved evidence")
            saved_version = session.get(PaperVersion, span.paper_version_id)
            manifest_version = next((item for item in manifest.paper_versions
                                     if item.id == span.paper_version_id), None)
            if (manifest_version is None or saved_version is None
                    or manifest_version.content_hash != saved_version.content_hash
                    or manifest_version.paper_id != saved_version.paper_id
                    or manifest_version.study_id != span.study_id):
                raise SynthesisValidationError("RunManifest PaperVersion snapshot does not match the saved version")
            citation_sources[span.id] = citation_source
            evidence_records.append(exported_evidence)

        output_sections: list[ExportSection] = []
        sentence_statuses: dict[tuple[int, int, int], tuple[str, str, str]] = {}
        for section_index, section in enumerate(document.sections):
            paragraphs: list[ExportParagraph] = []
            for paragraph_index, paragraph in enumerate(section.paragraphs):
                sentences: list[ExportSentence] = []
                for sentence_index, sentence in enumerate(paragraph.sentences):
                    position = (section_index, paragraph_index, sentence_index)
                    citation_audit_status, semantic_audit_status, omission_audit_status = audit_data[position]
                    warnings: list[str] = []
                    if citation_audit_status.status == "failed":
                        warnings.append("Citation-chain audit failed; citation provenance needs review.")
                    if semantic_audit_status.verdict == "reject":
                        warnings.append("Semantic audit rejected this sentence; do not treat it as a supported fact.")
                    elif semantic_audit_status.verdict == "revise":
                        warnings.append("Semantic audit requested revision; original wording is retained and suggestion is not applied.")
                    if omission_audit_status.verdict in {"revise", "flag"}:
                        warnings.append("Omission audit flagged potentially material uncited evidence.")
                    sentences.append(ExportSentence(
                        section_index=section_index,
                        paragraph_index=paragraph_index,
                        sentence_index=sentence_index,
                        original_text=sentence.text,
                        claim_ids=sentence.claim_ids,
                        evidence_ids=sentence.evidence_ids,
                        citation_audit=citation_audit_status,
                        semantic_audit=semantic_audit_status,
                        omission_audit=omission_audit_status,
                        export_warning=" ".join(warnings) if warnings else None,
                    ))
                    sentence_statuses[position] = (
                        citation_audit_status.status,
                        semantic_audit_status.verdict or semantic_audit_status.status,
                        omission_audit_status.verdict or omission_audit_status.status,
                    )
                paragraphs.append(ExportParagraph(sentences=sentences))
            output_sections.append(ExportSection(heading=section.heading, paragraphs=paragraphs))

        content = GroundedReviewExport(
            run_id=run_id,
            manifest_id=manifest_row.id,
            manifest_hash=manifest_row.content_hash,
            draft_id=draft.id,
            draft_input_hash=draft.input_hash,
            synthesis_schema_version=draft.schema_version,
            title=document.title,
            research_question=document.research_question,
            sections=output_sections,
            evidence=evidence_records,
            audit_summary=ExportAuditSummary(
                citation=_stage_audit(manifest, "citation"),
                semantic=_stage_audit(manifest, "semantic"),
                omission=_stage_audit(manifest, "omission"),
            ),
        )

        rejected_count = sum(
            sentence.semantic_audit.verdict == "reject"
            for section in content.sections
            for paragraph in section.paragraphs
            for sentence in paragraph.sentences
        )
        revised_count = sum(
            sentence.semantic_audit.verdict == "revise"
            for section in content.sections
            for paragraph in section.paragraphs
            for sentence in paragraph.sentences
        )
        annotated = _annotated_document(document, sentence_statuses)
        markdown = render_markdown(annotated, citation_sources)
        notices = [
            "> **Audit status:** Sentence wording below is the saved draft and has not been automatically revised.",
            f"> RunManifest `{manifest_row.id}` (content hash `{manifest_row.content_hash}`); "
            f"Draft `{draft.id}`; ResearchRun `{run_id}`.",
        ]
        if revised_count:
            notices.append(f"> **Human review required:** {revised_count} sentence(s) received semantic `revise`; suggested revisions, if any, are not applied.")
        if rejected_count:
            notices.append(f"> **Rejected claims present:** {rejected_count} sentence(s) received semantic `reject`; they remain visible only with explicit warnings and must not be treated as supported facts.")
        if not revised_count and not rejected_count:
            notices.append("> Semantic audit found no `revise` or `reject` sentences in this saved audit snapshot.")
        markdown = "\n".join(notices) + "\n\n" + markdown

        # Append clearly segregated review notes; never substitute suggested wording.
        review_notes: list[str] = []
        for section_index, section in enumerate(content.sections):
            for paragraph_index, paragraph in enumerate(section.paragraphs):
                for sentence in paragraph.sentences:
                    verdict = sentence.semantic_audit.verdict
                    if verdict not in {"revise", "reject"}:
                        continue
                    review_notes.append(
                        f"- `{_escape_markdown(section.heading)}` ({paragraph_index + 1}.{sentence.sentence_index + 1}) — "
                        f"semantic `{verdict}`; original sentence retained: “{_escape_markdown(sentence.original_text)}”"
                    )
                    if sentence.semantic_audit.rationale:
                        review_notes.append(f"  - Rationale: {_escape_markdown(sentence.semantic_audit.rationale)}")
                    if sentence.semantic_audit.minimal_revision:
                        review_notes.append(
                            f"  - Suggested revision (not applied): "
                            f"{_escape_markdown(sentence.semantic_audit.minimal_revision)}"
                        )
        if review_notes:
            markdown = markdown.rstrip() + "\n\n## Audit review notes\n\n" + "\n".join(review_notes) + "\n"

        json_text = json.dumps(content.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
        return GroundedReviewExportResult(
            run_id=run_id,
            manifest_id=manifest_row.id,
            draft_id=draft.id,
            content=content,
            markdown=markdown,
            json_text=json_text,
        )


def write_grounded_review_export(
    result: GroundedReviewExportResult,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Write the already-built deterministic Markdown and JSON payloads."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    markdown_path = destination / "report.md"
    json_path = destination / "report.json"
    markdown_path.write_text(result.markdown, encoding="utf-8", newline="\n")
    json_path.write_text(result.json_text, encoding="utf-8", newline="\n")
    return markdown_path, json_path
