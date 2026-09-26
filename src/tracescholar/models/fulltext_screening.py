"""Auditable, retryable full-text decisions and their cited PDF chunks."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    JSON, CheckConstraint, DateTime, Float, ForeignKey, Index, Integer, String,
    Text, UniqueConstraint, Uuid, func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from tracescholar.database.base import Base

if TYPE_CHECKING:
    from tracescholar.models.chunk import Chunk
    from tracescholar.models.paper import Paper
    from tracescholar.models.paper_version import PaperVersion
    from tracescholar.models.research_plan import ResearchPlanRecord
    from tracescholar.models.research_run import ResearchRun


json_type = JSON().with_variant(JSONB(), "postgresql")


class FullTextScreeningResult(Base):
    """One current full-text decision or retryable failure per run and paper."""

    __tablename__ = "fulltext_screening_results"
    __table_args__ = (
        UniqueConstraint("run_id", "paper_id", name="uq_fulltext_screening_run_paper"),
        Index("ix_fulltext_screening_run_status", "run_id", "status"),
        CheckConstraint("status IN ('success', 'failed')", name="status_valid"),
        CheckConstraint("label IS NULL OR label IN ('include', 'exclude', 'uncertain')", name="label_valid"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("research_runs.id", ondelete="CASCADE"), nullable=False,
    )
    paper_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False,
    )
    paper_version_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("paper_versions.id", ondelete="CASCADE"), nullable=True,
    )
    plan_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("research_plans.id", ondelete="CASCADE"), nullable=False,
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    label: Mapped[str | None] = mapped_column(String(16))
    rationale: Mapped[str | None] = mapped_column(Text)
    matched_inclusion_criteria: Mapped[list[str]] = mapped_column(json_type, nullable=False, default=list)
    matched_exclusion_criteria: Mapped[list[str]] = mapped_column(json_type, nullable=False, default=list)
    supported_sub_question_indices: Mapped[list[int]] = mapped_column(json_type, nullable=False, default=list)
    evidence_role: Mapped[str | None] = mapped_column(String(40))
    quality_warnings: Mapped[list[str]] = mapped_column(json_type, nullable=False, default=list)
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    input_snapshot: Mapped[dict[str, Any]] = mapped_column(json_type, nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(64), nullable=False)
    llm_model: Mapped[str] = mapped_column(String(128), nullable=False)
    retrieval_model_revision: Mapped[str] = mapped_column(String(64), nullable=False)
    failure_code: Mapped[str | None] = mapped_column(String(64))
    failure_detail: Mapped[str | None] = mapped_column(Text)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    attempted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now(),
    )

    research_run: Mapped[ResearchRun] = relationship(back_populates="fulltext_screening_results")
    paper: Mapped[Paper] = relationship(back_populates="fulltext_screening_results")
    paper_version: Mapped[PaperVersion | None] = relationship(back_populates="fulltext_screening_results")
    research_plan: Mapped[ResearchPlanRecord] = relationship(back_populates="fulltext_screening_results")
    evidence: Mapped[list[FullTextScreeningEvidence]] = relationship(
        back_populates="screening_result", cascade="all, delete-orphan",
    )


class FullTextScreeningEvidence(Base):
    """A cited candidate passage, with its retrieval provenance and rank."""

    __tablename__ = "fulltext_screening_evidence"
    __table_args__ = (
        UniqueConstraint("screening_result_id", "chunk_id", name="uq_fulltext_screening_evidence_chunk"),
        CheckConstraint("rank >= 1", name="rank_positive"),
        CheckConstraint("similarity >= -1 AND similarity <= 1", name="similarity_range"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    screening_result_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("fulltext_screening_results.id", ondelete="CASCADE"), nullable=False,
    )
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False,
    )
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    similarity: Mapped[float] = mapped_column(Float, nullable=False)
    retrieval_sub_question_indices: Mapped[list[int]] = mapped_column(json_type, nullable=False)

    screening_result: Mapped[FullTextScreeningResult] = relationship(back_populates="evidence")
    chunk: Mapped[Chunk] = relationship(back_populates="fulltext_screening_evidence")
