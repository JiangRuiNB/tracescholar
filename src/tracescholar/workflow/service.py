"""Ordered, resumable orchestration over TraceScholar's existing stage services.

Each call advances at most one stage. Stage progress is inferred from persisted
domain records where possible and otherwise from the workflow execution log.
"""

from __future__ import annotations

import dataclasses
import json
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from tracescholar.database import get_session_factory, session_scope
from tracescholar.discovery import run_planned_discovery
from tracescholar.evidence import extract_evidence
from tracescholar.exports import build_grounded_review_export, write_grounded_review_export
from tracescholar.fulltext import acquire_fulltext
from tracescholar.fulltext_screening import screen_fulltext_research_run
from tracescholar.manifests import create_run_manifest
from tracescholar.models import (
    Chunk, ChunkEmbedding, Claim, ClaimGeneration, CitationAudit,
    CitationSentenceAudit, EvidenceExtraction, EvidenceSpan, FullTextAcquisition,
    FullTextScreeningResult, OmissionAudit, PdfParseRecord,
    PlannedQuery, ResearchPlanRecord, ResearchRun, RunManifestRecord,
    SearchQuery, SearchResult, ScreeningResult, SemanticCitationAudit,
    StudyPaper, StudyRunSelection, StudyVersionComparison, SynthesisDraft,
    WorkflowStageExecution,
)
from tracescholar.planning import plan_research_run
from tracescholar.pdf_parsing import parse_research_run
from tracescholar.retrieval import embed_research_run
from tracescholar.screening import screen_research_run
from tracescholar.studies import normalize_studies
from tracescholar.synthesis import (
    audit_omitted_counterevidence, audit_synthesis_citations,
    audit_synthesis_semantics, write_synthesis,
)
from tracescholar.synthesis.schemas import SynthesisDocument


class WorkflowStage(StrEnum):
    PLANNED = "planned"
    DISCOVERED = "discovered"
    SCREENED = "screened"
    ACQUIRED = "acquired"
    PARSED = "parsed"
    EMBEDDED = "embedded"
    FULLTEXT_SCREENED = "fulltext_screened"
    STUDIES_NORMALIZED = "studies_normalized"
    EVIDENCE_EXTRACTED = "evidence_extracted"
    SYNTHESIZED = "synthesized"
    AUDITED = "audited"
    MANIFESTED = "manifested"
    EXPORTED = "exported"


STAGE_ORDER: tuple[WorkflowStage, ...] = tuple(WorkflowStage)


@dataclass(frozen=True, slots=True)
class WorkflowStageState:
    stage: WorkflowStage
    status: str
    attempt_count: int
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_seconds: float | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class WorkflowSnapshot:
    run_id: uuid.UUID
    stages: tuple[WorkflowStageState, ...]
    completed_stages: tuple[WorkflowStage, ...]
    next_stage: WorkflowStage | None
    last_failure_stage: WorkflowStage | None
    last_failure_reason: str | None

    def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
        """Return a JSON-friendly status document for API and CLI callers."""
        def stamp(value: datetime | None) -> str | None:
            return value.isoformat() if value is not None else None

        return {
            "run_id": str(self.run_id),
            "completed_stages": [stage.value for stage in self.completed_stages],
            "next_stage": self.next_stage.value if self.next_stage else None,
            "last_failure": ({
                "stage": self.last_failure_stage.value,
                "reason": self.last_failure_reason,
            } if self.last_failure_stage else None),
            "stages": [{
                "stage": item.stage.value,
                "status": item.status,
                "attempt_count": item.attempt_count,
                "started_at": stamp(item.started_at),
                "finished_at": stamp(item.finished_at),
                "duration_seconds": item.duration_seconds,
                "detail": item.detail,
            } for item in self.stages],
        }


@dataclass(frozen=True, slots=True)
class WorkflowStepResult:
    run_id: uuid.UUID
    stage: WorkflowStage | None
    status: str
    skipped: bool
    result: dict[str, Any] | None
    detail: str | None
    snapshot: WorkflowSnapshot


@dataclass(frozen=True, slots=True)
class WorkflowRunResult:
    """Terminal result of a synchronous run through the remaining workflow."""

    run_id: uuid.UUID
    status: str
    steps: tuple[WorkflowStepResult, ...]
    snapshot: WorkflowSnapshot

    def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
        return {
            "run_id": str(self.run_id),
            "status": self.status,
            "steps": [{
                "stage": step.stage.value if step.stage else None,
                "status": step.status,
                "skipped": step.skipped,
                "detail": step.detail,
            } for step in self.steps],
            "workflow": self.snapshot.model_dump(mode=mode),
        }


class WorkflowOrderError(ValueError):
    """Raised when a caller requests a stage other than the next eligible one."""


def _latest_attempts(session: Session, run_id: uuid.UUID) -> dict[WorkflowStage, WorkflowStageExecution]:
    rows = session.scalars(select(WorkflowStageExecution).where(
        WorkflowStageExecution.run_id == run_id,
    ).order_by(WorkflowStageExecution.attempt.desc()))
    latest: dict[WorkflowStage, WorkflowStageExecution] = {}
    for row in rows:
        latest.setdefault(WorkflowStage(row.stage), row)
    return latest


def _run_paper_ids(session: Session, run_id: uuid.UUID) -> set[uuid.UUID]:
    return set(session.scalars(select(SearchResult.paper_id).join(
        SearchQuery, SearchQuery.id == SearchResult.search_query_id,
    ).where(SearchQuery.run_id == run_id).distinct()))


def _latest_generation(session: Session, run_id: uuid.UUID) -> ClaimGeneration | None:
    return session.scalar(select(ClaimGeneration).where(
        ClaimGeneration.run_id == run_id,
    ).order_by(ClaimGeneration.created_at.desc(), ClaimGeneration.id.desc()).limit(1))


def _evidence_is_complete(session: Session, run_id: uuid.UUID) -> bool:
    """Check the persisted Claim × included Study × acquired PDF task ledger."""
    generation = _latest_generation(session, run_id)
    if generation is None or generation.status != "success":
        return False
    claims = list(session.scalars(select(Claim).where(Claim.generation_id == generation.id)))
    if not claims:
        return True
    included_study_ids = set(session.scalars(select(StudyPaper.study_id).join(
        FullTextScreeningResult, FullTextScreeningResult.paper_id == StudyPaper.paper_id,
    ).where(FullTextScreeningResult.run_id == run_id,
            FullTextScreeningResult.status == "success",
            FullTextScreeningResult.label == "include").distinct()))
    selections = list(session.scalars(select(StudyRunSelection).where(
        StudyRunSelection.run_id == run_id,
        StudyRunSelection.study_id.in_(included_study_ids),
    ))) if included_study_ids else []
    if not selections:
        return False
    expected: set[tuple[uuid.UUID, uuid.UUID]] = set()
    for claim in claims:
        for selection in selections:
            if claim.scope_kind == "study_specific" and claim.basis_study_id != selection.study_id:
                continue
            version_ids: set[uuid.UUID] = set()
            if selection.preferred_paper_version_id:
                version_ids.add(selection.preferred_paper_version_id)
                for comparison in session.scalars(select(StudyVersionComparison).where(
                    StudyVersionComparison.study_id == selection.study_id,
                    StudyVersionComparison.result_relation == "changed",
                )):
                    version_ids.update((comparison.version_a_id, comparison.version_b_id))
            acquired_ids = set(session.scalars(select(FullTextAcquisition.paper_version_id).where(
                FullTextAcquisition.run_id == run_id,
                FullTextAcquisition.paper_version_id.in_(version_ids),
                FullTextAcquisition.status.in_(("downloaded", "cached")),
            ))) if version_ids else set()
            expected.update((claim.id, version_id) for version_id in acquired_ids)
    if not expected:
        return False
    successful = set(session.execute(select(
        EvidenceExtraction.claim_id, EvidenceExtraction.paper_version_id,
    ).where(EvidenceExtraction.claim_id.in_([claim.id for claim in claims]),
            EvidenceExtraction.status == "success")))
    return expected.issubset(successful)


def _sentence_count(document_json: dict[str, Any] | None) -> int:
    if not document_json:
        return 0
    document = SynthesisDocument.model_validate_json(
        json.dumps(document_json, ensure_ascii=False),
    )
    return sum(len(paragraph.sentences)
               for section in document.sections for paragraph in section.paragraphs)


def _fact_completion(session: Session, run_id: uuid.UUID) -> tuple[dict[WorkflowStage, bool], dict[WorkflowStage, str]]:
    """Infer stage completion from durable outputs, including valid no-op stages."""
    run = session.get(ResearchRun, run_id)
    if run is None:
        raise LookupError(f"ResearchRun {run_id} does not exist.")

    done: dict[WorkflowStage, bool] = {stage: False for stage in STAGE_ORDER}
    detail: dict[WorkflowStage, str] = {}

    plan = session.scalar(select(ResearchPlanRecord).where(ResearchPlanRecord.run_id == run_id))
    done[WorkflowStage.PLANNED] = plan is not None

    queries = list(session.scalars(select(SearchQuery).where(SearchQuery.run_id == run_id)))
    planned = list(session.scalars(select(PlannedQuery).where(PlannedQuery.run_id == run_id)))
    if planned:
        executed_sources: dict[uuid.UUID, set[str]] = {}
        for query in queries:
            if query.planned_query_id:
                executed_sources.setdefault(query.planned_query_id, set()).add(query.source)
        expected_sources = {"openalex", "crossref"}
        query_coverage = all(
            expected_sources.issubset(executed_sources.get(item.id, set()))
            for item in planned
        )
        done[WorkflowStage.DISCOVERED] = done[WorkflowStage.PLANNED] and bool(queries) and query_coverage
    else:
        done[WorkflowStage.DISCOVERED] = done[WorkflowStage.PLANNED] and bool(queries)
    if done[WorkflowStage.DISCOVERED] and not _run_paper_ids(session, run_id):
        detail[WorkflowStage.DISCOVERED] = "Search completed; no paper records were returned."

    paper_ids = _run_paper_ids(session, run_id)
    screened_ids = set(session.scalars(select(ScreeningResult.paper_id).where(
        ScreeningResult.run_id == run_id,
    )))
    done[WorkflowStage.SCREENED] = (
        done[WorkflowStage.DISCOVERED]
        and (not paper_ids or paper_ids.issubset(screened_ids))
    )

    acquisition_candidates = set(session.scalars(select(ScreeningResult.paper_id).where(
        ScreeningResult.run_id == run_id,
        ScreeningResult.label.in_(("include", "maybe")),
    )))
    terminal_acquisition_ids = set(session.scalars(select(FullTextAcquisition.paper_id).where(
        FullTextAcquisition.run_id == run_id,
        FullTextAcquisition.status.in_(("downloaded", "cached", "unavailable")),
    )))
    done[WorkflowStage.ACQUIRED] = (
        done[WorkflowStage.SCREENED]
        and (not acquisition_candidates or acquisition_candidates.issubset(terminal_acquisition_ids))
    )

    run_versions = set(session.scalars(select(FullTextAcquisition.paper_version_id).where(
        FullTextAcquisition.run_id == run_id,
        FullTextAcquisition.status.in_(("downloaded", "cached")),
        FullTextAcquisition.paper_version_id.is_not(None),
    )))
    parsed_ids = set(session.scalars(select(PdfParseRecord.paper_version_id).where(
        PdfParseRecord.paper_version_id.in_(run_versions),
        PdfParseRecord.status == "success",
    ))) if run_versions else set()
    done[WorkflowStage.PARSED] = (
        done[WorkflowStage.ACQUIRED]
        and (not run_versions or run_versions.issubset(parsed_ids))
    )
    if done[WorkflowStage.PARSED] and not run_versions:
        detail[WorkflowStage.PARSED] = "No acquired PDF versions; parsing is not applicable."

    run_chunks = list(session.scalars(select(Chunk).join(
        FullTextAcquisition,
        FullTextAcquisition.paper_version_id == Chunk.paper_version_id,
    ).where(FullTextAcquisition.run_id == run_id,
            FullTextAcquisition.status.in_(("downloaded", "cached")))))
    embedded_chunk_ids = set(session.scalars(select(ChunkEmbedding.chunk_id).where(
        ChunkEmbedding.chunk_id.in_([chunk.id for chunk in run_chunks]),
        ChunkEmbedding.status == "success",
    ))) if run_chunks else set()
    done[WorkflowStage.EMBEDDED] = (
        done[WorkflowStage.PARSED]
        and (not run_chunks or {chunk.id for chunk in run_chunks}.issubset(embedded_chunk_ids))
    )
    if done[WorkflowStage.EMBEDDED] and not run_chunks:
        detail[WorkflowStage.EMBEDDED] = "No parsed chunks; embedding is not applicable."

    fulltext_candidates = set(session.scalars(select(FullTextAcquisition.paper_id).where(
        FullTextAcquisition.run_id == run_id,
        FullTextAcquisition.status.in_(("downloaded", "cached")),
    )))
    fulltext_screened_ids = set(session.scalars(select(FullTextScreeningResult.paper_id).where(
        FullTextScreeningResult.run_id == run_id,
        FullTextScreeningResult.status == "success",
    )))
    done[WorkflowStage.FULLTEXT_SCREENED] = (
        done[WorkflowStage.EMBEDDED]
        and (not fulltext_candidates or fulltext_candidates.issubset(fulltext_screened_ids))
    )
    if done[WorkflowStage.FULLTEXT_SCREENED] and not fulltext_candidates:
        detail[WorkflowStage.FULLTEXT_SCREENED] = "No acquired PDFs to screen at full text; stage is not applicable."

    final_papers = set(session.scalars(select(FullTextScreeningResult.paper_id).where(
        FullTextScreeningResult.run_id == run_id,
        FullTextScreeningResult.status == "success",
        FullTextScreeningResult.label.in_(("include", "uncertain")),
    )))
    memberships = set(session.scalars(select(StudyPaper.paper_id).where(
        StudyPaper.paper_id.in_(final_papers),
    ))) if final_papers else set()
    final_study_ids = set(session.scalars(select(StudyPaper.study_id).where(
        StudyPaper.paper_id.in_(final_papers),
    ))) if final_papers else set()
    selected_study_ids = set(session.scalars(select(StudyRunSelection.study_id).where(
        StudyRunSelection.run_id == run_id,
        StudyRunSelection.study_id.in_(final_study_ids),
    ))) if final_study_ids else set()
    done[WorkflowStage.STUDIES_NORMALIZED] = (
        done[WorkflowStage.FULLTEXT_SCREENED]
        and (not final_papers or final_papers.issubset(memberships))
        and final_study_ids.issubset(selected_study_ids)
    )

    generation = _latest_generation(session, run_id)
    done[WorkflowStage.EVIDENCE_EXTRACTED] = (
        done[WorkflowStage.STUDIES_NORMALIZED] and _evidence_is_complete(session, run_id)
    )
    if generation is not None and generation.status == "success":
        detail[WorkflowStage.EVIDENCE_EXTRACTED] = "Claim and evidence task records are complete."

    draft = session.scalar(select(SynthesisDraft).where(
        SynthesisDraft.run_id == run_id,
    ).order_by(SynthesisDraft.created_at.desc(), SynthesisDraft.id.desc()).limit(1))
    done[WorkflowStage.SYNTHESIZED] = (
        done[WorkflowStage.EVIDENCE_EXTRACTED] and draft is not None and draft.status == "success"
    )
    if draft is not None:
        sentence_total = _sentence_count(draft.document_json)
        citation_positions = set(session.execute(select(
            CitationSentenceAudit.section_index,
            CitationSentenceAudit.paragraph_index,
            CitationSentenceAudit.sentence_index,
        ).where(CitationSentenceAudit.draft_id == draft.id)))
        citation_coverage = len(citation_positions)
        semantic_success = session.scalar(select(func.count()).select_from(
            SemanticCitationAudit).where(SemanticCitationAudit.draft_id == draft.id,
                                         SemanticCitationAudit.status == "success")) or 0
        omission_success = session.scalar(select(func.count()).select_from(OmissionAudit).where(
            OmissionAudit.draft_id == draft.id, OmissionAudit.status == "success")) or 0
        audit_rows = session.scalar(select(func.count()).select_from(CitationAudit).where(
            CitationAudit.draft_id == draft.id)) or 0
        done[WorkflowStage.AUDITED] = (
            done[WorkflowStage.SYNTHESIZED] and audit_rows > 0
            and citation_coverage >= sentence_total
            and semantic_success >= sentence_total
            and omission_success >= sentence_total
        )
    else:
        done[WorkflowStage.AUDITED] = False

    manifest = session.scalar(select(RunManifestRecord).where(
        RunManifestRecord.run_id == run_id,
    ).order_by(RunManifestRecord.created_at.desc(), RunManifestRecord.id.desc()).limit(1))
    manifest_matches_draft = bool(
        manifest and draft
        and (manifest.manifest_json.get("current_draft") or {}).get("id") == str(draft.id)
    )
    done[WorkflowStage.MANIFESTED] = done[WorkflowStage.AUDITED] and manifest_matches_draft

    export_dir = Path("data") / "exports" / str(run_id)
    export_exists = (export_dir / "report.md").is_file() and (export_dir / "report.json").is_file()
    done[WorkflowStage.EXPORTED] = done[WorkflowStage.MANIFESTED] and export_exists

    return done, detail


def inspect_workflow(
    run_id: uuid.UUID, *, session_factory: sessionmaker[Session] | None = None,
) -> WorkflowSnapshot:
    """Read persisted stage outputs and report the first unfinished stage."""
    factory = session_factory or get_session_factory()
    with factory() as session:
        fact_done, details = _fact_completion(session, run_id)
        latest = _latest_attempts(session, run_id)
    states: list[WorkflowStageState] = []
    completed: list[WorkflowStage] = []
    last_failure: WorkflowStage | None = None
    last_reason: str | None = None
    next_stage: WorkflowStage | None = None

    for stage in STAGE_ORDER:
        attempt = latest.get(stage)
        if attempt and attempt.status in {"failed", "blocked", "running"}:
            # A terminal execution error must remain actionable even when a
            # service committed some outputs before raising. Its next retry
            # relies on that service's own idempotency checks.
            is_done = False
            status = attempt.status
        else:
            is_done = fact_done[stage] or bool(attempt and attempt.status == "completed")
            status = "completed" if is_done else "pending"
        if is_done:
            status = "completed"
            completed.append(stage)
        if attempt and attempt.status == "failed" and (
            last_failure is None or (attempt.finished_at or attempt.started_at)
            >= (latest[last_failure].finished_at or latest[last_failure].started_at)
        ):
            last_failure, last_reason = stage, attempt.failure_reason
        states.append(WorkflowStageState(
            stage=stage,
            status=status,
            attempt_count=attempt.attempt if attempt else 0,
            started_at=attempt.started_at if attempt else None,
            finished_at=attempt.finished_at if attempt else None,
            duration_seconds=attempt.duration_seconds if attempt else None,
            detail=details.get(stage) or (attempt.failure_reason if attempt else None),
        ))
        if next_stage is None and status != "completed":
            next_stage = stage

    return WorkflowSnapshot(run_id, tuple(states), tuple(completed), next_stage,
                            last_failure, last_reason)


def _json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _json_safe(dataclasses.asdict(value))
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (uuid.UUID, datetime)):
        return str(value) if isinstance(value, uuid.UUID) else value.isoformat()
    if isinstance(value, StrEnum):
        return value.value
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _result_incomplete(result: Any) -> str | None:
    """Treat retryable per-item failures as a failed stage, not a lost result."""
    if isinstance(result, dict):
        nested = [f"{key}: {reason}" for key, value in result.items()
                  if (reason := _result_incomplete(value))]
        return "; ".join(nested) if nested else None
    failures = getattr(result, "failures", ()) or ()
    failed = getattr(result, "failed", 0) or 0
    pending = getattr(result, "pending", 0) or 0
    counts = getattr(result, "counts", {}) or {}
    audit_failures = counts.get("failed", 0) if isinstance(counts, dict) else 0
    if audit_failures:
        failed += audit_failures
    if failures or failed or pending:
        parts = []
        if failed:
            parts.append(f"failed items: {failed}")
        if pending:
            parts.append(f"pending items: {pending}")
        if failures:
            parts.append(f"reported failures: {len(failures)}")
        return "; ".join(parts)
    return None


def _call_stage(
    stage: WorkflowStage,
    run_id: uuid.UUID,
    *,
    session_factory: sessionmaker[Session],
) -> Any:
    """Dispatch through existing service entrypoints; no stage logic is copied here."""
    if stage == WorkflowStage.PLANNED:
        return plan_research_run(run_id, session_factory=session_factory)
    if stage == WorkflowStage.DISCOVERED:
        return run_planned_discovery(run_id, session_factory=session_factory)
    if stage == WorkflowStage.SCREENED:
        return screen_research_run(run_id, session_factory=session_factory)
    if stage == WorkflowStage.ACQUIRED:
        return acquire_fulltext(run_id, session_factory=session_factory)
    if stage == WorkflowStage.PARSED:
        return parse_research_run(run_id, session_factory=session_factory)
    if stage == WorkflowStage.EMBEDDED:
        return embed_research_run(run_id, session_factory=session_factory)
    if stage == WorkflowStage.FULLTEXT_SCREENED:
        return screen_fulltext_research_run(run_id, session_factory=session_factory)
    if stage == WorkflowStage.STUDIES_NORMALIZED:
        return normalize_studies(run_id, session_factory=session_factory)
    if stage == WorkflowStage.EVIDENCE_EXTRACTED:
        return extract_evidence(run_id, session_factory=session_factory)
    if stage == WorkflowStage.SYNTHESIZED:
        return write_synthesis(run_id, session_factory=session_factory)
    if stage == WorkflowStage.AUDITED:
        citation = audit_synthesis_citations(run_id, session_factory=session_factory)
        semantic = audit_synthesis_semantics(run_id, session_factory=session_factory)
        omission = audit_omitted_counterevidence(run_id, session_factory=session_factory)
        return {"citation": citation, "semantic": semantic, "omission": omission}
    if stage == WorkflowStage.MANIFESTED:
        return create_run_manifest(run_id, session_factory=session_factory)
    if stage == WorkflowStage.EXPORTED:
        exported = build_grounded_review_export(run_id, session_factory=session_factory)
        output_dir = Path("data") / "exports" / str(run_id)
        markdown_path, json_path = write_grounded_review_export(exported, output_dir)
        return {"markdown_path": str(markdown_path), "json_path": str(json_path),
                "manifest_id": str(exported.manifest_id), "draft_id": str(exported.draft_id)}
    raise ValueError(f"Unsupported workflow stage: {stage.value}")


def _precondition_reason(session: Session, run_id: uuid.UUID, stage: WorkflowStage) -> str | None:
    done, _ = _fact_completion(session, run_id)
    history = _latest_attempts(session, run_id)
    for prior_stage, attempt in history.items():
        if attempt.status == "completed":
            done[prior_stage] = True
    index = STAGE_ORDER.index(stage)
    if index and not done[STAGE_ORDER[index - 1]]:
        return f"Stage '{stage.value}' requires '{STAGE_ORDER[index - 1].value}' to be complete."
    if stage == WorkflowStage.DISCOVERED and session.scalar(select(ResearchPlanRecord.id).where(
        ResearchPlanRecord.run_id == run_id,
    )) is None:
        return "A frozen ResearchPlan is required before Discovery."
    if stage == WorkflowStage.PARSED:
        pdf_count = session.scalar(select(func.count()).select_from(FullTextAcquisition).where(
            FullTextAcquisition.run_id == run_id,
            FullTextAcquisition.status.in_(("downloaded", "cached")),
            FullTextAcquisition.paper_version_id.is_not(None),
        )) or 0
        if not pdf_count:
            return "PDF parsing requires at least one acquired PDF version."
    if stage == WorkflowStage.EMBEDDED:
        chunk_count = session.scalar(select(func.count()).select_from(Chunk).join(
            FullTextAcquisition,
            FullTextAcquisition.paper_version_id == Chunk.paper_version_id,
        ).where(FullTextAcquisition.run_id == run_id)) or 0
        if not chunk_count:
            return "Embedding requires parsed PDF chunks."
    if stage == WorkflowStage.SYNTHESIZED and not _evidence_is_complete(session, run_id):
        return "Synthesis requires a complete persisted Evidence Ledger."
    if stage == WorkflowStage.SYNTHESIZED:
        saved_evidence_id = session.scalar(select(EvidenceSpan.id).join(
            EvidenceExtraction, EvidenceExtraction.id == EvidenceSpan.extraction_id,
        ).join(Claim, Claim.id == EvidenceExtraction.claim_id).join(
            ClaimGeneration, ClaimGeneration.id == Claim.generation_id,
        ).where(ClaimGeneration.run_id == run_id,
                ClaimGeneration.status == "success",
                EvidenceExtraction.status == "success").limit(1))
        if saved_evidence_id is None:
            return "Synthesis requires at least one Claim with a saved EvidenceSpan."
    return None


def _new_attempt(
    session: Session, run_id: uuid.UUID, stage: WorkflowStage, *, status: str,
    reason: str | None = None,
) -> WorkflowStageExecution:
    previous = session.scalar(select(WorkflowStageExecution).where(
        WorkflowStageExecution.run_id == run_id,
        WorkflowStageExecution.stage == stage.value,
    ).order_by(WorkflowStageExecution.attempt.desc()).limit(1))
    latest_attempt = previous.attempt if previous else 0
    now = datetime.now(UTC)
    if previous is not None and previous.status == "running":
        previous.status = "failed"
        previous.finished_at = now
        previous.duration_seconds = max(0.0, (now - previous.started_at).total_seconds())
        previous.failure_type = "InterruptedWorkflowStage"
        previous.failure_reason = "Previous process ended before the stage recorded a terminal status."
    row = WorkflowStageExecution(
        run_id=run_id, stage=stage.value, attempt=latest_attempt + 1,
        status=status, started_at=now,
        finished_at=now if status != "running" else None,
        failure_type="WorkflowOrderError" if status == "blocked" else None,
        failure_reason=reason,
    )
    session.add(row)
    session.flush()
    return row


def run_next_stage(
    run_id: uuid.UUID,
    *,
    stage: WorkflowStage | str | None = None,
    session_factory: sessionmaker[Session] | None = None,
) -> WorkflowStepResult:
    """Execute exactly one next stage, or record why an explicit stage is blocked."""
    factory = session_factory or get_session_factory()
    before = inspect_workflow(run_id, session_factory=factory)
    requested = WorkflowStage(stage) if stage is not None else before.next_stage
    if requested is None:
        return WorkflowStepResult(run_id, None, "completed", True, None,
                                  "All workflow stages are complete.", before)
    if requested in before.completed_stages:
        return WorkflowStepResult(run_id, requested, "completed", True, None,
                                  "Stage already completed; no service was called.", before)
    if requested != before.next_stage:
        raise WorkflowOrderError(
            f"Next eligible stage is '{before.next_stage.value if before.next_stage else 'none'}'; "
            f"cannot run '{requested.value}'."
        )

    with session_scope(factory) as session:
        if session.get(ResearchRun, run_id) is None:
            raise LookupError(f"ResearchRun {run_id} does not exist.")
        reason = _precondition_reason(session, run_id, requested)
        if reason:
            attempt = _new_attempt(session, run_id, requested, status="blocked", reason=reason)
            attempt.finished_at = datetime.now(UTC)
        else:
            attempt = _new_attempt(session, run_id, requested, status="running")
        attempt_id = attempt.id

    if reason:
        snapshot = inspect_workflow(run_id, session_factory=factory)
        return WorkflowStepResult(run_id, requested, "blocked", False, None, reason, snapshot)

    started = time.monotonic()
    try:
        raw_result = _call_stage(requested, run_id, session_factory=factory)
        incomplete_reason = _result_incomplete(raw_result)
        safe_result = _json_safe(raw_result)
        status = "failed" if incomplete_reason else "completed"
        failure_type = "IncompleteStageResult" if incomplete_reason else None
        failure_reason = incomplete_reason
    except Exception as error:  # persist stage failure without rolling back prior services' commits
        raw_result = None
        safe_result = None
        status = "failed"
        failure_type = type(error).__name__
        failure_reason = str(error)[:4000] or type(error).__name__

    finished = datetime.now(UTC)
    with session_scope(factory) as session:
        attempt = session.get(WorkflowStageExecution, attempt_id)
        if attempt is not None:
            attempt.status = status
            attempt.finished_at = finished
            attempt.duration_seconds = round(time.monotonic() - started, 6)
            attempt.result_json = safe_result
            attempt.failure_type = failure_type
            attempt.failure_reason = failure_reason

    snapshot = inspect_workflow(run_id, session_factory=factory)
    return WorkflowStepResult(run_id, requested, status, False, safe_result,
                              failure_reason, snapshot)


def run_workflow(
    run_id: uuid.UUID,
    *,
    session_factory: sessionmaker[Session] | None = None,
    on_step: Callable[[WorkflowStepResult], None] | None = None,
) -> WorkflowRunResult:
    """Synchronously advance until every stage is complete or one cannot proceed.

    Each iteration delegates exactly one stage to ``run_next_stage``. A failed
    or blocked stage ends this invocation; calling the function again resumes
    from that stage using the existing services' idempotency behavior.
    """
    factory = session_factory or get_session_factory()
    snapshot = inspect_workflow(run_id, session_factory=factory)
    steps: list[WorkflowStepResult] = []

    while snapshot.next_stage is not None:
        current_stage = snapshot.next_stage
        step = run_next_stage(run_id, stage=current_stage, session_factory=factory)
        steps.append(step)
        if on_step is not None:
            on_step(step)
        snapshot = step.snapshot

        if step.status in {"failed", "blocked"}:
            return WorkflowRunResult(run_id, step.status, tuple(steps), snapshot)
        if step.status != "completed" or snapshot.next_stage == current_stage:
            # Defensive stop if a step did not produce a terminal state change.
            return WorkflowRunResult(run_id, "blocked", tuple(steps), snapshot)

    return WorkflowRunResult(run_id, "completed", tuple(steps), snapshot)
