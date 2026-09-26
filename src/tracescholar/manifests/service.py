"""Create and retrieve stable manifests from persisted ResearchRun facts only."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from sqlalchemy import or_, select
from sqlalchemy.orm import Session, sessionmaker

from tracescholar.database import session_scope
from tracescholar.manifests.schemas import (
    ManifestAuditSummary,
    ManifestDraft,
    ManifestDraftSentence,
    ManifestEmbeddingConfig,
    ManifestEvidenceSpan,
    ManifestModelUse,
    ManifestPaperDecision,
    ManifestPaperVersion,
    ManifestPlan,
    ManifestPlannedQuery,
    ManifestPromptSchemaVersion,
    ManifestRun,
    ManifestSearchQuery,
    ManifestSearchResult,
    ManifestSentenceAudit,
    ManifestStageAudit,
    ManifestStageTiming,
    ManifestStudy,
    RunManifest,
)
from tracescholar.models import (
    CanonicalStudy,
    CitationAudit,
    CitationAuditSentenceLink,
    CitationSentenceAudit,
    ClaimGeneration,
    ChunkEmbedding,
    Chunk,
    Claim,
    EvidenceExtraction,
    EvidenceSpan,
    FullTextAcquisition,
    FullTextScreeningResult,
    OmissionAudit,
    Paper,
    PaperVersion,
    PdfParseRecord,
    PlannedQuery,
    ResearchPlanRecord,
    ResearchRun,
    ScreeningResult,
    SearchQuery,
    SearchResult,
    SemanticCitationAudit,
    StudyPaper,
    StudyRunSelection,
    SynthesisDraft,
)


@dataclass(frozen=True)
class ManifestResult:
    """Persisted manifest and whether this call created its immutable snapshot."""

    manifest_id: uuid.UUID
    run_id: uuid.UUID
    content_hash: str
    created: bool
    manifest: RunManifest


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _safe_url(value: str | None) -> str | None:
    """Remove credentials and query parameters from persisted external URLs."""
    if not value:
        return value
    parts = urlsplit(value)
    hostname = parts.hostname or ""
    if parts.port:
        hostname = f"{hostname}:{parts.port}"
    return urlunsplit((parts.scheme, hostname, parts.path, "", ""))


def _safe_value(value: Any, *, key: str = "") -> Any:
    """Recursively remove credentials if an old config snapshot contains them."""
    lowered = key.casefold()
    if any(token in lowered for token in ("key", "secret", "password", "token", "credential", "authorization")):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): _safe_value(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_value(item, key=key) for item in value]
    if isinstance(value, str) and value.startswith(("http://", "https://")):
        return _safe_url(value)
    return value


def _jsonable(model: RunManifest) -> dict[str, Any]:
    return model.model_dump(mode="json")


def _stage_audit(
    *, audit_id: uuid.UUID, status: str, input_hash: str | None = None,
    auditor_version: str | None = None, prompt_versions: set[str] | None = None,
    schema_versions: set[int | str] | None = None,
    sentence_rows: list[ManifestSentenceAudit] | None = None,
    sentence_count: int = 0, issue_count: int = 0,
    citation_count: int = 0, unique_evidence_count: int = 0,
) -> ManifestStageAudit:
    sentences = sentence_rows or []
    counts: dict[str, int] = defaultdict(int)
    for sentence in sentences:
        counts[sentence.verdict or sentence.status] += 1
    return ManifestStageAudit(
        id=audit_id,
        status=status,
        input_hash=input_hash,
        auditor_version=auditor_version,
        prompt_versions=sorted(prompt_versions or set()),
        schema_versions=sorted(schema_versions or set(), key=str),
        sentence_count=sentence_count or len(sentences),
        issue_count=issue_count,
        citation_count=citation_count,
        unique_evidence_count=unique_evidence_count,
        verdict_counts=dict(sorted(counts.items())),
        sentences=sentences,
    )


def _add_model_use(
    records: list[ManifestModelUse], *, stage: str, task: str, model_name: str | None,
    record_id: uuid.UUID, provider: str | None = None, endpoint_url: str | None = None,
    model_revision: str | None = None, encoder_version: str | None = None,
    dimensions: int | None = None,
) -> None:
    safe_endpoint = _safe_url(endpoint_url)
    for item in records:
        if (item.stage, item.task, item.model_name, item.provider, item.endpoint_url,
                item.model_revision, item.encoder_version, item.dimensions) == (
                stage, task, model_name, provider, safe_endpoint, model_revision,
                encoder_version, dimensions):
            if record_id not in item.record_ids:
                item.record_ids.append(record_id)
            return
    records.append(ManifestModelUse(
        stage=stage, task=task, model_name=model_name, provider=provider,
        endpoint_url=safe_endpoint, model_revision=model_revision,
        encoder_version=encoder_version, dimensions=dimensions, record_ids=[record_id],
    ))


def _add_versions(
    session: Session,
    *,
    study_by_paper: dict[uuid.UUID, uuid.UUID],
    selections: list[StudyRunSelection],
    acquisitions: list[FullTextAcquisition],
    fulltext_rows: list[FullTextScreeningResult],
    spans: list[EvidenceSpan],
    cited_evidence_ids: set[uuid.UUID],
) -> list[ManifestPaperVersion]:
    used: dict[uuid.UUID, set[str]] = defaultdict(set)
    for row in selections:
        if row.preferred_paper_version_id:
            used[row.preferred_paper_version_id].add("preferred_for_evidence_extraction")
    for row in acquisitions:
        if row.paper_version_id:
            used[row.paper_version_id].add("fulltext_acquisition")
    for row in fulltext_rows:
        if row.paper_version_id:
            used[row.paper_version_id].add("fulltext_screening")
    for span in spans:
        used[span.paper_version_id].add("evidence_ledger")
        if span.id in cited_evidence_ids:
            used[span.paper_version_id].add("cited_in_current_draft")

    if not used:
        return []
    versions = session.scalars(
        select(PaperVersion).where(PaperVersion.id.in_(used.keys()))
    ).all()
    papers = {
        paper.id: paper for paper in session.scalars(
            select(Paper).where(Paper.id.in_({version.paper_id for version in versions}))
        ).all()
    }
    result: list[ManifestPaperVersion] = []
    for version in sorted(versions, key=lambda item: str(item.id)):
        paper = papers.get(version.paper_id)
        if paper is None:
            continue
        result.append(ManifestPaperVersion(
            id=version.id,
            study_id=study_by_paper.get(version.paper_id),
            paper_id=version.paper_id,
            paper_title=paper.title,
            content_hash=version.content_hash,
            source_name=version.source_name,
            source_url=_safe_url(version.source_url) or "",
            license=version.license,
            version_label=version.version_label,
            retrieved_at=_utc(version.retrieved_at),
            used_by=sorted(used[version.id]),
        ))
    return result


def _build_manifest(session: Session, run_id: uuid.UUID) -> RunManifest:
    run = session.get(ResearchRun, run_id)
    if run is None:
        raise LookupError(f"ResearchRun {run_id} does not exist.")

    gaps: set[str] = set()
    run_data = ManifestRun(
        id=run.id, question=run.question, status=run.status.value,
        scope=_safe_value(run.scope or {}),
        config_snapshot=_safe_value(run.config_snapshot or {}),
        created_at=_utc(run.created_at), updated_at=_utc(run.updated_at),
    )
    plan_row = session.scalar(select(ResearchPlanRecord).where(ResearchPlanRecord.run_id == run_id))
    plan = None
    if plan_row:
        plan = ManifestPlan(
            id=plan_row.id, input_question=plan_row.input_question,
            input_scope=_safe_value(plan_row.input_scope),
            plan=_safe_value(plan_row.plan_json), schema_version=plan_row.schema_version,
            prompt_version=plan_row.prompt_version, model_name=plan_row.llm_model,
            created_at=_utc(plan_row.created_at),
        )
    else:
        gaps.add("No persisted ResearchPlan exists for this ResearchRun.")

    planned_rows = session.scalars(
        select(PlannedQuery).where(PlannedQuery.run_id == run_id).order_by(PlannedQuery.query_key, PlannedQuery.id)
    ).all()
    search_rows = session.scalars(
        select(SearchQuery).where(SearchQuery.run_id == run_id).order_by(SearchQuery.source, SearchQuery.query, SearchQuery.id)
    ).all()
    planned_queries = [ManifestPlannedQuery(
        id=row.id, plan_id=row.plan_id, query=row.query, query_key=row.query_key,
        purpose=row.purpose, variant=row.variant, origins=_safe_value(row.origins),
        generation_version=row.generation_version,
        executed_search_query_ids=sorted(
            (query.id for query in search_rows if query.planned_query_id == row.id), key=str
        ), created_at=_utc(row.created_at),
    ) for row in planned_rows]
    search_queries: list[ManifestSearchQuery] = []
    for row in search_rows:
        results = session.scalars(
            select(SearchResult).where(SearchResult.search_query_id == row.id).order_by(SearchResult.id)
        ).all()
        search_queries.append(ManifestSearchQuery(
            id=row.id, planned_query_id=row.planned_query_id, query=row.query, source=row.source,
            filters=_safe_value(row.filters or {}), returned_count=row.returned_count,
            skipped_count=row.skipped_count, scope_filtered_count=row.scope_filtered_count,
            executed_at=_utc(row.executed_at), results=[ManifestSearchResult(
                id=item.id, paper_id=item.paper_id, source_record_id=item.source_record_id,
                source_url=_safe_url(item.source_url), discovered_at=_utc(item.discovered_at),
            ) for item in results],
        ))
    data_sources = sorted({row.source for row in search_rows})

    selections = session.scalars(
        select(StudyRunSelection).where(StudyRunSelection.run_id == run_id).order_by(StudyRunSelection.study_id)
    ).all()
    paper_ids = set(session.scalars(
        select(SearchResult.paper_id).join(SearchQuery, SearchResult.search_query_id == SearchQuery.id)
        .where(SearchQuery.run_id == run_id)
    ).all())
    paper_ids.update(session.scalars(
        select(ScreeningResult.paper_id).where(ScreeningResult.run_id == run_id)
    ).all())
    paper_ids.update(session.scalars(
        select(FullTextScreeningResult.paper_id).where(FullTextScreeningResult.run_id == run_id)
    ).all())
    paper_ids.update(session.scalars(
        select(FullTextAcquisition.paper_id).where(FullTextAcquisition.run_id == run_id)
    ).all())
    study_members = session.scalars(
        select(StudyPaper).where(or_(
            StudyPaper.paper_id.in_(paper_ids) if paper_ids else False,
            StudyPaper.study_id.in_([item.study_id for item in selections]) if selections else False,
        ))
    ).all()
    study_by_paper = {member.paper_id: member.study_id for member in study_members}
    study_ids = {item.study_id for item in selections} | set(study_by_paper.values())
    studies: list[ManifestStudy] = []
    papers_for_studies: dict[uuid.UUID, list[StudyPaper]] = defaultdict(list)
    for member in study_members:
        papers_for_studies[member.study_id].append(member)
    fulltext_rows = session.scalars(
        select(FullTextScreeningResult).where(FullTextScreeningResult.run_id == run_id)
    ).all()
    final_by_study: dict[uuid.UUID, str] = {}
    selection_by_study = {item.study_id: item for item in selections}
    decisions_by_paper: dict[uuid.UUID, list[FullTextScreeningResult]] = defaultdict(list)
    for row in fulltext_rows:
        decisions_by_paper[row.paper_id].append(row)
    for study_id in sorted(study_ids, key=str):
        study = session.get(CanonicalStudy, study_id)
        if study is None:
            continue
        members = papers_for_studies[study_id]
        paper_map = {item.id: item for item in session.scalars(
            select(Paper).where(Paper.id.in_([member.paper_id for member in members]))
        ).all()} if members else {}
        decisions: list[ManifestPaperDecision] = []
        labels: list[str] = []
        for member in sorted(members, key=lambda item: str(item.paper_id)):
            paper = paper_map.get(member.paper_id)
            for decision in sorted(decisions_by_paper.get(member.paper_id, []), key=lambda item: (item.updated_at, str(item.id))):
                if decision.status == "success" and decision.label:
                    labels.append(decision.label)
                    if decision.label in {"include", "uncertain", "exclude"}:
                        decisions.append(ManifestPaperDecision(
                            paper_id=member.paper_id, title=paper.title if paper else "[missing paper]",
                            publication_role=member.publication_role,
                            decision=decision.label, status=decision.status, rationale=decision.rationale,
                        ))
        if "include" in labels:
            final_status = "include"
        elif "uncertain" in labels:
            final_status = "uncertain"
        elif labels:
            final_status = "exclude"
        elif study_id in selection_by_study:
            final_status = "selected_unresolved"
        else:
            continue
        final_by_study[study_id] = final_status
        decision_paper_ids = {item.paper_id for item in decisions}
        for member in sorted(members, key=lambda item: str(item.paper_id)):
            if member.paper_id not in decision_paper_ids:
                paper = paper_map.get(member.paper_id)
                decisions.append(ManifestPaperDecision(
                    paper_id=member.paper_id,
                    title=paper.title if paper else "[missing paper]",
                    publication_role=member.publication_role,
                    decision="not_screened", status="not_screened", rationale=None,
                ))
        decisions.sort(key=lambda item: (str(item.paper_id), item.decision))
        selection = selection_by_study.get(study_id)
        canonical_paper = session.get(Paper, study.canonical_paper_id)
        studies.append(ManifestStudy(
            id=study.id, canonical_paper_id=study.canonical_paper_id,
            canonical_title=canonical_paper.title if canonical_paper else "[missing canonical paper]",
            final_status=final_status,
            selection_reason=selection.selection_reason if selection else None,
            selection_policy_version=selection.policy_version if selection else None,
            preferred_paper_version_id=selection.preferred_paper_version_id if selection else None,
            papers=decisions,
        ))

    draft = session.scalar(
        select(SynthesisDraft).where(
            SynthesisDraft.run_id == run_id, SynthesisDraft.status == "success",
            SynthesisDraft.document_json.is_not(None),
        ).order_by(SynthesisDraft.created_at.desc(), SynthesisDraft.id.desc()).limit(1)
    )
    cited_evidence_ids: set[uuid.UUID] = set()
    draft_model: ManifestDraft | None = None
    if draft and draft.document_json:
        document = draft.document_json
        sentence_models: list[ManifestDraftSentence] = []
        for section_index, section in enumerate(document.get("sections", [])):
            for paragraph_index, paragraph in enumerate(section.get("paragraphs", [])):
                for sentence_index, sentence in enumerate(paragraph.get("sentences", [])):
                    evidence_ids = [uuid.UUID(value) for value in sentence.get("evidence_ids", [])]
                    cited_evidence_ids.update(evidence_ids)
                    sentence_models.append(ManifestDraftSentence(
                        section_index=section_index, paragraph_index=paragraph_index,
                        sentence_index=sentence_index, text=sentence.get("text", ""),
                        claim_ids=[uuid.UUID(value) for value in sentence.get("claim_ids", [])],
                        evidence_ids=evidence_ids,
                    ))
        draft_model = ManifestDraft(
            id=draft.id, status=draft.status, input_hash=draft.input_hash,
            claim_generation_id=draft.claim_generation_id, schema_version=draft.schema_version,
            prompt_version=draft.prompt_version, model_name=draft.llm_model,
            created_at=_utc(draft.created_at), title=document.get("title", ""),
            research_question=document.get("research_question", ""), sentences=sentence_models,
        )
    elif draft is None:
        gaps.add("No successful persisted synthesis draft exists for this ResearchRun.")

    spans: list[EvidenceSpan] = []
    if draft:
        draft_claim_ids = session.scalars(select(Claim.id).where(
            Claim.generation_id == draft.claim_generation_id
        )).all()
        if draft_claim_ids:
            draft_extraction_ids = session.scalars(select(EvidenceExtraction.id).where(
                EvidenceExtraction.claim_id.in_(draft_claim_ids)
            )).all()
            if draft_extraction_ids:
                spans = session.scalars(select(EvidenceSpan).where(
                    EvidenceSpan.extraction_id.in_(draft_extraction_ids)
                ).order_by(EvidenceSpan.id)).all()
    if not spans and cited_evidence_ids:
        spans = session.scalars(
            select(EvidenceSpan).where(EvidenceSpan.id.in_(cited_evidence_ids)).order_by(EvidenceSpan.id)
        ).all()
    claim_by_span: dict[uuid.UUID, uuid.UUID] = {}
    if spans:
        extraction_rows = session.scalars(select(EvidenceExtraction).where(
            EvidenceExtraction.id.in_({span.extraction_id for span in spans})
        )).all()
        claim_by_extraction = {row.id: row.claim_id for row in extraction_rows}
        claim_by_span = {span.id: claim_by_extraction[span.extraction_id]
                         for span in spans if span.extraction_id in claim_by_extraction}
    evidence_models = [ManifestEvidenceSpan(
        id=span.id, claim_id=claim_by_span[span.id], study_id=span.study_id,
        paper_version_id=span.paper_version_id, chunk_id=span.chunk_id,
        quote=span.quote, stance=span.stance, page_number=span.page_number,
        cited_by_current_draft=span.id in cited_evidence_ids,
        section=span.section, page_char_start=span.page_char_start,
        page_char_end=span.page_char_end, locator=_safe_value(span.locator),
    ) for span in spans if span.id in claim_by_span]

    acquisitions = session.scalars(select(FullTextAcquisition).where(FullTextAcquisition.run_id == run_id)).all()
    versions = _add_versions(
        session, study_by_paper=study_by_paper, selections=selections,
        acquisitions=acquisitions, fulltext_rows=fulltext_rows, spans=spans,
        cited_evidence_ids=cited_evidence_ids,
    )

    current_citation = None
    if draft:
        current_citation = session.scalar(
            select(CitationAudit).where(CitationAudit.run_id == run_id, CitationAudit.draft_id == draft.id)
            .order_by(CitationAudit.checked_at.desc(), CitationAudit.id.desc()).limit(1)
        )
    audit_summary = ManifestAuditSummary()
    citation_sentence_rows: list[CitationSentenceAudit] = []
    if current_citation:
        links = session.scalars(select(CitationAuditSentenceLink).where(
            CitationAuditSentenceLink.citation_audit_id == current_citation.id
        )).all()
        linked_ids = [link.sentence_audit_id for link in links]
        citation_sentence_rows = session.scalars(select(CitationSentenceAudit).where(
            CitationSentenceAudit.id.in_(linked_ids)
        ).order_by(CitationSentenceAudit.section_index, CitationSentenceAudit.paragraph_index,
                   CitationSentenceAudit.sentence_index)) .all() if linked_ids else []
        citation_sentences = [ManifestSentenceAudit(
            section_index=row.section_index, paragraph_index=row.paragraph_index,
            sentence_index=row.sentence_index, status=row.status, verdict=None,
            input_hash=row.input_hash, issues=_safe_value(row.issues_json),
        ) for row in citation_sentence_rows]
        citation_issues = sum(len(row.issues_json or []) for row in citation_sentence_rows)
        audit_summary.citation = _stage_audit(
            audit_id=current_citation.id, status=current_citation.status,
            input_hash=current_citation.input_hash,
            auditor_version=current_citation.auditor_version,
            sentence_rows=citation_sentences,
            sentence_count=current_citation.sentence_count,
            issue_count=citation_issues,
            citation_count=current_citation.citation_count,
            unique_evidence_count=current_citation.unique_evidence_count,
        )
        semantic_rows = session.scalars(select(SemanticCitationAudit).where(
            SemanticCitationAudit.run_id == run_id,
            SemanticCitationAudit.draft_id == draft.id,
            SemanticCitationAudit.citation_audit_id == current_citation.id,
        ).order_by(SemanticCitationAudit.section_index, SemanticCitationAudit.paragraph_index,
                   SemanticCitationAudit.sentence_index, SemanticCitationAudit.updated_at.desc(),
                   SemanticCitationAudit.id.desc())).all()
        semantic_latest: dict[tuple[int, int, int], SemanticCitationAudit] = {}
        for row in semantic_rows:
            semantic_latest.setdefault((row.section_index, row.paragraph_index, row.sentence_index), row)
        if semantic_latest:
            rows = list(semantic_latest.values())
            sem_sentences = [ManifestSentenceAudit(
                section_index=row.section_index, paragraph_index=row.paragraph_index,
                sentence_index=row.sentence_index, status=row.status, verdict=row.verdict,
                input_hash=row.input_hash, model_name=row.llm_model,
                prompt_version=row.prompt_version, rationale=row.rationale,
                minimal_revision=row.minimal_revision,
            ) for row in sorted(rows, key=lambda item: (item.section_index, item.paragraph_index, item.sentence_index))]
            audit_summary.semantic = _stage_audit(
                audit_id=current_citation.id, status="complete" if all(r.status == "success" for r in rows) else "partial",
                sentence_rows=sem_sentences, sentence_count=len(rows),
                prompt_versions={r.prompt_version for r in rows},
            )
        omission_rows = session.scalars(select(OmissionAudit).where(
            OmissionAudit.run_id == run_id, OmissionAudit.draft_id == draft.id,
            OmissionAudit.citation_audit_id == current_citation.id,
        ).order_by(OmissionAudit.section_index, OmissionAudit.paragraph_index,
                   OmissionAudit.sentence_index, OmissionAudit.updated_at.desc(),
                   OmissionAudit.id.desc())).all()
        omission_latest: dict[tuple[int, int, int], OmissionAudit] = {}
        for row in omission_rows:
            omission_latest.setdefault((row.section_index, row.paragraph_index, row.sentence_index), row)
        if omission_latest:
            rows = list(omission_latest.values())
            omission_sentences = [ManifestSentenceAudit(
                section_index=row.section_index, paragraph_index=row.paragraph_index,
                sentence_index=row.sentence_index, status=row.status, verdict=row.verdict,
                input_hash=row.input_hash, model_name=row.llm_model,
                prompt_version=row.prompt_version, rationale=row.rationale,
                omitted_evidence_ids=[uuid.UUID(value) for value in row.omitted_evidence_ids],
                impact_types=_safe_value(row.impact_types_json),
            ) for row in sorted(rows, key=lambda item: (item.section_index, item.paragraph_index, item.sentence_index))]
            audit_summary.omission = _stage_audit(
                audit_id=current_citation.id, status="complete" if all(r.status == "success" for r in rows) else "partial",
                sentence_rows=omission_sentences, sentence_count=len(rows),
                prompt_versions={r.prompt_version for r in rows},
            )

    models: list[ManifestModelUse] = []
    prompt_versions: list[ManifestPromptSchemaVersion] = []

    def add_stage(stage: str, task: str, rows: list[Any], prompt_attr: str = "prompt_version",
                  model_attr: str = "llm_model", schema_attr: str | None = None,
                  impl_attr: str | None = None) -> None:
        prompts = {getattr(row, prompt_attr) for row in rows if prompt_attr and getattr(row, prompt_attr, None)}
        schemas = {getattr(row, schema_attr) for row in rows if schema_attr and getattr(row, schema_attr, None) is not None}
        implementations = {getattr(row, impl_attr) for row in rows if impl_attr and getattr(row, impl_attr, None)}
        if rows:
            prompt_versions.append(ManifestPromptSchemaVersion(
                stage=stage, task=task, prompt_versions=sorted(prompts),
                schema_versions=sorted(schemas, key=str), schema_version_recorded=bool(schema_attr and schemas),
                implementation_versions=sorted(implementations), record_ids=sorted({r.id for r in rows}, key=str),
            ))
            if not schema_attr or not schemas:
                gaps.add(f"{stage} schema version was not persisted by the stage record.")
            for row in rows:
                model = getattr(row, model_attr, None) if model_attr else None
                if model:
                    _add_model_use(models, stage=stage, task=task, model_name=model,
                                   record_id=row.id, model_revision=getattr(row, "retrieval_model_revision", None))

    if plan_row:
        add_stage("planner", "scope_planning", [plan_row], schema_attr="schema_version")
    if planned_rows:
        prompt_versions.append(ManifestPromptSchemaVersion(
            stage="query_planner", task="query_generation", prompt_versions=[],
            schema_versions=[], schema_version_recorded=False,
            implementation_versions=sorted({row.generation_version for row in planned_rows}),
            record_ids=sorted({row.id for row in planned_rows}, key=str),
        ))
        gaps.add("Query-generation model/prompt were not persisted; the stored generator version is recorded.")
    screening_rows = session.scalars(select(ScreeningResult).where(ScreeningResult.run_id == run_id)).all()
    add_stage("screener", "title_abstract_screening", screening_rows)
    add_stage("screener", "fulltext_screening", fulltext_rows, impl_attr="retrieval_model_revision")
    claim_generations = session.scalars(select(ClaimGeneration).where(ClaimGeneration.run_id == run_id)).all()
    add_stage("evidence", "claim_generation", claim_generations)
    extraction_rows = session.scalars(select(EvidenceExtraction).where(
        EvidenceExtraction.claim_id.in_(select(Claim.id).where(
            Claim.generation_id.in_(select(ClaimGeneration.id).where(ClaimGeneration.run_id == run_id))
        ))
    )).all()
    add_stage("evidence", "evidence_extraction", extraction_rows, impl_attr="retrieval_model_revision")
    if draft:
        add_stage("synthesis", "synthesis_writer", [draft], schema_attr="schema_version")
    if current_citation:
        prompt_versions.append(ManifestPromptSchemaVersion(
            stage="citation_audit", task="deterministic_chain_audit", prompt_versions=[],
            schema_versions=[], schema_version_recorded=False,
            implementation_versions=[current_citation.auditor_version], record_ids=[current_citation.id],
        ))
        gaps.add("citation_audit schema version was not persisted; auditor version is recorded instead.")
    sem_records = session.scalars(select(SemanticCitationAudit).where(
        SemanticCitationAudit.run_id == run_id,
        SemanticCitationAudit.citation_audit_id == (current_citation.id if current_citation else uuid.UUID(int=0)),
    )).all()
    add_stage("semantic_audit", "sentence_semantics", sem_records)
    omission_records = session.scalars(select(OmissionAudit).where(
        OmissionAudit.run_id == run_id,
        OmissionAudit.citation_audit_id == (current_citation.id if current_citation else uuid.UUID(int=0)),
    )).all()
    add_stage("omission_audit", "omitted_evidence_check", omission_records)

    embedding_rows = session.scalars(select(ChunkEmbedding).join(Chunk).where(
        Chunk.paper_version_id.in_({version.id for version in versions})
    )).all() if versions else []
    embedding_groups: dict[tuple[str, str, str, str, str, int, str | None], list[ChunkEmbedding]] = defaultdict(list)
    for row in embedding_rows:
        embedding_groups[(row.provider, row.model_name, row.model_revision, row.source_revision,
                          row.encoder_version, row.dimensions, row.endpoint_url)].append(row)
        _add_model_use(models, stage="embedding", task="chunk_embedding", model_name=row.model_name,
                       provider=row.provider, endpoint_url=row.endpoint_url,
                       model_revision=row.model_revision, encoder_version=row.encoder_version,
                       dimensions=row.dimensions, record_id=row.id)
    embedding_configs = [ManifestEmbeddingConfig(
        provider=key[0], model_name=key[1], model_revision=key[2], source_revision=key[3],
        encoder_version=key[4], dimensions=key[5], endpoint_url=_safe_url(key[6]),
        successful_records=sum(row.status == "success" for row in rows),
        failed_records=sum(row.status == "failed" for row in rows),
    ) for key, rows in sorted(embedding_groups.items(), key=lambda item: item[0])]
    if not embedding_rows:
        gaps.add("No persisted embedding records were found for the PaperVersions used by this run.")

    if models:
        gaps.add("LLM provider/base URL was not persisted for historical calls; only recorded model names are shown.")

    timing_events: dict[str, list[datetime]] = defaultdict(list)

    def event(stage: str, value: datetime | None) -> None:
        if value is not None:
            timing_events[stage].append(_utc(value))

    event("planning", plan_row.created_at if plan_row else None)
    for row in planned_rows: event("query_planning", row.created_at)
    for row in search_rows: event("discovery", row.executed_at)
    for row in screening_rows: event("title_abstract_screening", row.created_at); event("title_abstract_screening", row.updated_at)
    for row in acquisitions: event("fulltext_acquisition", row.attempted_at)
    for row in fulltext_rows: event("fulltext_screening", row.attempted_at); event("fulltext_screening", row.updated_at)
    parse_records = session.scalars(select(PdfParseRecord).where(
        PdfParseRecord.paper_version_id.in_({version.id for version in versions})
    )).all() if versions else []
    if parse_records:
        prompt_versions.append(ManifestPromptSchemaVersion(
            stage="pdf_parser", task="page_aware_parsing", prompt_versions=[],
            schema_versions=[], schema_version_recorded=False,
            implementation_versions=sorted({row.parser_version for row in parse_records}),
            record_ids=sorted({row.id for row in parse_records}, key=str),
        ))
    for row in parse_records: event("pdf_parsing", row.attempted_at)
    for row in embedding_rows: event("embedding", row.attempted_at)
    for row in selections: event("study_selection", row.selected_at)
    for row in claim_generations: event("claim_generation", row.created_at); event("claim_generation", row.updated_at)
    for row in extraction_rows: event("evidence_extraction", row.created_at); event("evidence_extraction", row.updated_at)
    if draft: event("synthesis", draft.created_at); event("synthesis", draft.updated_at)
    if current_citation: event("citation_audit", current_citation.checked_at)
    for row in citation_sentence_rows: event("citation_audit", row.created_at)
    for row in sem_records: event("semantic_audit", row.created_at); event("semantic_audit", row.updated_at)
    for row in omission_records: event("omission_audit", row.created_at); event("omission_audit", row.updated_at)
    stage_timings = []
    for stage, events in sorted(timing_events.items()):
        first, last = min(events), max(events)
        stage_timings.append(ManifestStageTiming(
            stage=stage, first_event_at=first, last_event_at=last,
            observed_span_seconds=max(0.0, (last - first).total_seconds()),
            event_count=len(events),
        ))
    if stage_timings:
        gaps.add("Stage durations are spans between persisted event timestamps, not measured wall-clock execution times.")

    return RunManifest(
        research_run=run_data, research_plan=plan, planned_queries=planned_queries,
        search_queries=search_queries, data_sources=data_sources, studies=studies,
        final_included_study_ids=sorted((sid for sid, status in final_by_study.items() if status == "include"), key=str),
        paper_versions=versions, evidence_spans=evidence_models, current_draft=draft_model,
        audit_summary=audit_summary,
        model_uses=sorted(models, key=lambda item: (item.stage, item.task, item.model_name or "", item.provider or "")),
        prompt_schema_versions=sorted(prompt_versions, key=lambda item: (item.stage, item.task)),
        embedding_configs=embedding_configs, stage_timings=stage_timings,
        capture_gaps=sorted(gaps),
    )


def create_run_manifest(
    run_id: uuid.UUID,
    *,
    session_factory: sessionmaker[Session] | None = None,
) -> ManifestResult:
    """Create or reuse the canonical content-addressed manifest for current DB facts."""
    with session_scope(session_factory) as session:
        manifest = _build_manifest(session, run_id)
        data = _jsonable(manifest)
        serialized = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        content_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        from tracescholar.models import RunManifestRecord

        record = session.scalar(select(RunManifestRecord).where(
            RunManifestRecord.run_id == run_id, RunManifestRecord.content_hash == content_hash
        ))
        created = record is None
        if record is None:
            record = RunManifestRecord(
                run_id=run_id, manifest_version=manifest.manifest_version,
                content_hash=content_hash, manifest_json=data,
                created_at=datetime.now(UTC),
            )
            session.add(record)
            session.flush()
        return ManifestResult(record.id, run_id, record.content_hash, created, manifest)


def get_run_manifest(
    manifest_id: uuid.UUID,
    *,
    session_factory: sessionmaker[Session] | None = None,
) -> ManifestResult:
    """Read and validate a persisted immutable manifest by ID."""
    from tracescholar.models import RunManifestRecord

    with session_scope(session_factory) as session:
        row = session.get(RunManifestRecord, manifest_id)
        if row is None:
            raise LookupError(f"RunManifest {manifest_id} does not exist.")
        manifest = RunManifest.model_validate_json(json.dumps(row.manifest_json, ensure_ascii=False))
        return ManifestResult(row.id, row.run_id, row.content_hash, False, manifest)


def get_latest_run_manifest(
    run_id: uuid.UUID,
    *,
    session_factory: sessionmaker[Session] | None = None,
) -> ManifestResult:
    """Read the newest saved manifest for one run."""
    from tracescholar.models import RunManifestRecord

    with session_scope(session_factory) as session:
        row = session.scalar(select(RunManifestRecord).where(
            RunManifestRecord.run_id == run_id
        ).order_by(RunManifestRecord.created_at.desc(), RunManifestRecord.id.desc()).limit(1))
        if row is None:
            raise LookupError(f"No RunManifest exists for ResearchRun {run_id}.")
        manifest = RunManifest.model_validate_json(json.dumps(row.manifest_json, ensure_ascii=False))
        return ManifestResult(row.id, row.run_id, row.content_hash, False, manifest)
