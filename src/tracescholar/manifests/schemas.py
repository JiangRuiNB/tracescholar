"""Fixed, validated JSON contract for a reproducible research-run manifest."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ManifestRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ManifestRun(ManifestRecord):
    id: uuid.UUID
    question: str
    status: str
    scope: dict[str, Any]
    config_snapshot: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class ManifestPlan(ManifestRecord):
    id: uuid.UUID
    input_question: str
    input_scope: dict[str, Any]
    plan: dict[str, Any]
    schema_version: int
    prompt_version: str
    model_name: str
    created_at: datetime


class ManifestPlannedQuery(ManifestRecord):
    id: uuid.UUID
    plan_id: uuid.UUID
    query: str
    query_key: str
    purpose: str
    variant: str
    origins: list[dict[str, Any]]
    generation_version: str
    executed_search_query_ids: list[uuid.UUID]
    created_at: datetime


class ManifestSearchResult(ManifestRecord):
    id: uuid.UUID
    paper_id: uuid.UUID
    source_record_id: str | None
    source_url: str | None
    discovered_at: datetime


class ManifestSearchQuery(ManifestRecord):
    id: uuid.UUID
    planned_query_id: uuid.UUID | None
    query: str
    source: str
    filters: dict[str, Any]
    returned_count: int
    skipped_count: int
    scope_filtered_count: int
    executed_at: datetime
    results: list[ManifestSearchResult]


class ManifestPaperDecision(ManifestRecord):
    paper_id: uuid.UUID
    title: str
    publication_role: str
    decision: Literal["include", "uncertain", "exclude", "not_screened"]
    status: str
    rationale: str | None


class ManifestStudy(ManifestRecord):
    id: uuid.UUID
    canonical_paper_id: uuid.UUID
    canonical_title: str
    final_status: Literal["include", "uncertain", "exclude", "selected_unresolved"]
    selection_reason: str | None
    selection_policy_version: str | None
    preferred_paper_version_id: uuid.UUID | None
    papers: list[ManifestPaperDecision]


class ManifestPaperVersion(ManifestRecord):
    id: uuid.UUID
    study_id: uuid.UUID | None
    paper_id: uuid.UUID
    paper_title: str
    content_hash: str
    source_name: str
    source_url: str
    license: str | None
    version_label: str | None
    retrieved_at: datetime
    used_by: list[str]


class ManifestEvidenceSpan(ManifestRecord):
    id: uuid.UUID
    claim_id: uuid.UUID
    study_id: uuid.UUID
    paper_version_id: uuid.UUID
    chunk_id: uuid.UUID
    quote: str
    stance: str
    cited_by_current_draft: bool
    page_number: int
    section: str
    page_char_start: int
    page_char_end: int
    locator: dict[str, Any]


class ManifestDraftSentence(ManifestRecord):
    section_index: int
    paragraph_index: int
    sentence_index: int
    text: str
    claim_ids: list[uuid.UUID]
    evidence_ids: list[uuid.UUID]


class ManifestDraft(ManifestRecord):
    id: uuid.UUID
    status: str
    input_hash: str
    claim_generation_id: uuid.UUID
    schema_version: int
    prompt_version: str
    model_name: str
    created_at: datetime
    title: str
    research_question: str
    sentences: list[ManifestDraftSentence]


class ManifestSentenceAudit(ManifestRecord):
    section_index: int
    paragraph_index: int
    sentence_index: int
    status: str
    verdict: str | None
    input_hash: str | None = None
    model_name: str | None = None
    prompt_version: str | None = None
    schema_version: int | str | None = None
    rationale: str | None = None
    minimal_revision: str | None = None
    issues: list[dict[str, Any]] = Field(default_factory=list)
    omitted_evidence_ids: list[uuid.UUID] = Field(default_factory=list)
    impact_types: list[dict[str, str]] = Field(default_factory=list)


class ManifestStageAudit(ManifestRecord):
    id: uuid.UUID
    status: str
    input_hash: str | None = None
    auditor_version: str | None = None
    prompt_versions: list[str] = Field(default_factory=list)
    schema_versions: list[int | str] = Field(default_factory=list)
    sentence_count: int
    issue_count: int
    citation_count: int = 0
    unique_evidence_count: int = 0
    verdict_counts: dict[str, int] = Field(default_factory=dict)
    sentences: list[ManifestSentenceAudit] = Field(default_factory=list)


class ManifestAuditSummary(ManifestRecord):
    citation: ManifestStageAudit | None = None
    semantic: ManifestStageAudit | None = None
    omission: ManifestStageAudit | None = None


class ManifestModelUse(ManifestRecord):
    stage: str
    task: str
    model_name: str | None
    provider: str | None = None
    endpoint_url: str | None = None
    model_revision: str | None = None
    encoder_version: str | None = None
    dimensions: int | None = None
    record_ids: list[uuid.UUID] = Field(default_factory=list)


class ManifestPromptSchemaVersion(ManifestRecord):
    stage: str
    task: str
    prompt_versions: list[str] = Field(default_factory=list)
    schema_versions: list[int | str] = Field(default_factory=list)
    schema_version_recorded: bool
    implementation_versions: list[str] = Field(default_factory=list)
    record_ids: list[uuid.UUID] = Field(default_factory=list)


class ManifestEmbeddingConfig(ManifestRecord):
    provider: str
    model_name: str
    model_revision: str
    source_revision: str
    encoder_version: str
    dimensions: int
    endpoint_url: str | None
    successful_records: int
    failed_records: int


class ManifestStageTiming(ManifestRecord):
    stage: str
    first_event_at: datetime
    last_event_at: datetime
    observed_span_seconds: float = Field(ge=0)
    event_count: int = Field(ge=1)
    basis: Literal["persisted_event_span"] = "persisted_event_span"


class RunManifest(ManifestRecord):
    """Versioned, stable snapshot of only facts already persisted for one run."""

    manifest_version: Literal[1] = 1
    research_run: ManifestRun
    research_plan: ManifestPlan | None = None
    planned_queries: list[ManifestPlannedQuery] = Field(default_factory=list)
    search_queries: list[ManifestSearchQuery] = Field(default_factory=list)
    data_sources: list[str] = Field(default_factory=list)
    studies: list[ManifestStudy] = Field(default_factory=list)
    final_included_study_ids: list[uuid.UUID] = Field(default_factory=list)
    paper_versions: list[ManifestPaperVersion] = Field(default_factory=list)
    evidence_spans: list[ManifestEvidenceSpan] = Field(default_factory=list)
    current_draft: ManifestDraft | None = None
    audit_summary: ManifestAuditSummary = Field(default_factory=ManifestAuditSummary)
    model_uses: list[ManifestModelUse] = Field(default_factory=list)
    prompt_schema_versions: list[ManifestPromptSchemaVersion] = Field(default_factory=list)
    embedding_configs: list[ManifestEmbeddingConfig] = Field(default_factory=list)
    stage_timings: list[ManifestStageTiming] = Field(default_factory=list)
    capture_gaps: list[str] = Field(default_factory=list)
