"""Grounded claim candidates and page-verifiable evidence ledger."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON, CheckConstraint, DateTime, Float, ForeignKey, Index, Integer, String,
    Text, UniqueConstraint, Uuid, func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from tracescholar.database.base import Base


json_type = JSON().with_variant(JSONB(), "postgresql")


class ClaimGeneration(Base):
    """One versioned candidate-generation outcome for a frozen plan and corpus."""

    __tablename__ = "claim_generations"
    __table_args__ = (
        UniqueConstraint("run_id", "input_hash", name="uq_claim_generation_input"),
        CheckConstraint("status IN ('success', 'failed')", name="status_valid"),
        Index("ix_claim_generations_run_created", "run_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("research_runs.id", ondelete="CASCADE"), nullable=False)
    plan_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("research_plans.id", ondelete="CASCADE"), nullable=False)
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    input_snapshot: Mapped[dict[str, Any]] = mapped_column(json_type, nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(64), nullable=False)
    llm_model: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failure_code: Mapped[str | None] = mapped_column(String(64))
    failure_detail: Mapped[str | None] = mapped_column(Text)
    no_claim_reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    claims: Mapped[list[Claim]] = relationship(back_populates="generation", cascade="all, delete-orphan")


class Claim(Base):
    """One atomic, evidence-seeking assertion tied to a plan sub-question."""

    __tablename__ = "claims"
    __table_args__ = (
        UniqueConstraint("generation_id", "sub_question_index", name="uq_claim_generation_subquestion"),
        CheckConstraint("sub_question_index >= 0", name="subquestion_nonnegative"),
        CheckConstraint("scope_kind IN ('study_specific', 'cross_study')", name="scope_kind_valid"),
        CheckConstraint("basis_chunk_char_end > basis_chunk_char_start", name="basis_range_valid"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    generation_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("claim_generations.id", ondelete="CASCADE"), nullable=False)
    sub_question_index: Mapped[int] = mapped_column(Integer, nullable=False)
    statement: Mapped[str] = mapped_column(Text, nullable=False)
    scope_kind: Mapped[str] = mapped_column(String(20), nullable=False, server_default="cross_study")
    basis_study_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("canonical_studies.id", ondelete="RESTRICT"))
    basis_chunk_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("chunks.id", ondelete="RESTRICT"), nullable=False)
    basis_quote: Mapped[str] = mapped_column(Text, nullable=False)
    basis_chunk_char_start: Mapped[int] = mapped_column(Integer, nullable=False)
    basis_chunk_char_end: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now())

    generation: Mapped[ClaimGeneration] = relationship(back_populates="claims")
    attempts: Mapped[list[EvidenceExtraction]] = relationship(back_populates="claim")


class EvidenceExtraction(Base):
    """One retryable claim × study × PDF outcome, including explicit no-evidence."""

    __tablename__ = "evidence_extractions"
    __table_args__ = (
        UniqueConstraint("claim_id", "study_id", "paper_version_id", "input_hash",
                         name="uq_evidence_extraction_input"),
        CheckConstraint("status IN ('success', 'failed')", name="status_valid"),
        CheckConstraint("disposition IS NULL OR disposition IN ('evidence', 'no_evidence')",
                        name="disposition_valid"),
        Index("ix_evidence_extractions_claim_study", "claim_id", "study_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    claim_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("claims.id", ondelete="CASCADE"), nullable=False)
    study_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("canonical_studies.id", ondelete="RESTRICT"), nullable=False)
    paper_version_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("paper_versions.id", ondelete="RESTRICT"), nullable=False)
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    input_snapshot: Mapped[dict[str, Any]] = mapped_column(json_type, nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(64), nullable=False)
    llm_model: Mapped[str] = mapped_column(String(128), nullable=False)
    retrieval_model_revision: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    disposition: Mapped[str | None] = mapped_column(String(16))
    no_evidence_reason: Mapped[str | None] = mapped_column(Text)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failure_code: Mapped[str | None] = mapped_column(String(64))
    failure_detail: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    claim: Mapped[Claim] = relationship(back_populates="attempts")
    spans: Mapped[list[EvidenceSpan]] = relationship(back_populates="extraction", cascade="all, delete-orphan")


class EvidenceSpan(Base):
    """A verbatim, page-local excerpt whose offsets are checked against ParsedPage."""

    __tablename__ = "evidence_spans"
    __table_args__ = (
        UniqueConstraint("extraction_id", "chunk_id", "chunk_char_start", "chunk_char_end",
                         name="uq_evidence_span_exact_quote"),
        CheckConstraint("stance IN ('supports', 'contradicts', 'qualifies', 'unrelated')",
                        name="stance_valid"),
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="confidence_range"),
        CheckConstraint("page_number >= 1", name="page_positive"),
        CheckConstraint("chunk_char_end > chunk_char_start", name="chunk_range_valid"),
        CheckConstraint("page_char_end > page_char_start", name="page_range_valid"),
        Index("ix_evidence_spans_study_stance", "study_id", "stance"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    extraction_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("evidence_extractions.id", ondelete="CASCADE"), nullable=False)
    study_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("canonical_studies.id", ondelete="RESTRICT"), nullable=False)
    paper_version_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("paper_versions.id", ondelete="RESTRICT"), nullable=False)
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("chunks.id", ondelete="RESTRICT"), nullable=False)
    quote: Mapped[str] = mapped_column(Text, nullable=False)
    stance: Mapped[str] = mapped_column(String(16), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    study_context: Mapped[str] = mapped_column(Text, nullable=False)
    limitations: Mapped[str] = mapped_column(Text, nullable=False)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    section: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_char_start: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_char_end: Mapped[int] = mapped_column(Integer, nullable=False)
    page_char_start: Mapped[int] = mapped_column(Integer, nullable=False)
    page_char_end: Mapped[int] = mapped_column(Integer, nullable=False)
    locator: Mapped[dict[str, Any]] = mapped_column(json_type, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now())

    extraction: Mapped[EvidenceExtraction] = relationship(back_populates="spans")
