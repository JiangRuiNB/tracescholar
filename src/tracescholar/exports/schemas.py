"""Validated JSON contract for deterministic Grounded Review exports."""

from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ExportRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ExportAuditStatus(ExportRecord):
    status: str
    verdict: str | None = None
    record_id: uuid.UUID | None = None
    input_hash: str | None = None
    entailment: str | None = None
    scope: str | None = None
    strength: str | None = None
    model_name: str | None = None
    prompt_version: str | None = None
    rationale: str | None = None
    minimal_revision: str | None = None
    issues: list[dict[str, Any]] = Field(default_factory=list)
    omitted_evidence_ids: list[uuid.UUID] = Field(default_factory=list)
    impact_types: list[dict[str, str]] = Field(default_factory=list)


class ExportSentence(ExportRecord):
    section_index: int
    paragraph_index: int
    sentence_index: int
    original_text: str
    claim_ids: list[uuid.UUID]
    evidence_ids: list[uuid.UUID]
    citation_audit: ExportAuditStatus
    semantic_audit: ExportAuditStatus
    omission_audit: ExportAuditStatus
    export_warning: str | None = None


class ExportParagraph(ExportRecord):
    sentences: list[ExportSentence]


class ExportSection(ExportRecord):
    heading: str
    paragraphs: list[ExportParagraph]


class ExportEvidence(ExportRecord):
    evidence_id: uuid.UUID
    claim_id: uuid.UUID
    claim_statement: str
    study_id: uuid.UUID
    paper_version_id: uuid.UUID
    chunk_id: uuid.UUID
    paper_title: str
    authors: list[str]
    year: int | None
    venue: str | None
    doi: str | None
    arxiv_id: str | None
    content_hash: str
    version_label: str | None
    source_url: str
    page: int
    section: str
    quote: str
    stance: str
    locator: dict[str, Any]


class ExportStageAudit(ExportRecord):
    audit_id: uuid.UUID | None = None
    status: str
    sentence_count: int = 0
    issue_count: int = 0
    citation_count: int = 0
    unique_evidence_count: int = 0
    verdict_counts: dict[str, int] = Field(default_factory=dict)


class ExportAuditSummary(ExportRecord):
    citation: ExportStageAudit
    semantic: ExportStageAudit
    omission: ExportStageAudit


class GroundedReviewExport(ExportRecord):
    """Portable report data tied to one immutable draft and RunManifest."""

    export_schema_version: Literal[1] = 1
    run_id: uuid.UUID
    manifest_id: uuid.UUID
    manifest_hash: str
    draft_id: uuid.UUID
    draft_input_hash: str
    synthesis_schema_version: int
    title: str
    research_question: str
    sections: list[ExportSection]
    evidence: list[ExportEvidence]
    audit_summary: ExportAuditSummary
