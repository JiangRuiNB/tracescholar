"""Persisted single-sentence omitted-counterevidence checks."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, CheckConstraint, DateTime, ForeignKey, Index, Integer, String, Text, Uuid, UniqueConstraint, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from tracescholar.database.base import Base


json_type = JSON().with_variant(JSONB(), "postgresql")


class OmissionAudit(Base):
    """One retryable assessment of all omitted spans and their sentence impact."""

    __tablename__ = "omission_audits"
    __table_args__ = (
        UniqueConstraint("draft_id", "section_index", "paragraph_index", "sentence_index",
                         "input_hash", name="uq_omission_audits_sentence_input"),
        CheckConstraint("status IN ('success', 'failed')", name="status_valid"),
        CheckConstraint("verdict IS NULL OR verdict IN ('pass', 'revise', 'flag')",
                        name="verdict_valid"),
        CheckConstraint("section_index >= 0 AND paragraph_index >= 0 AND sentence_index >= 0",
                        name="indices_nonnegative"),
        Index("ix_omission_audits_run_draft", "run_id", "draft_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("research_runs.id", ondelete="CASCADE"), nullable=False)
    draft_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("synthesis_drafts.id", ondelete="CASCADE"), nullable=False)
    citation_audit_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("citation_audits.id", ondelete="CASCADE"), nullable=False)
    section_index: Mapped[int] = mapped_column(Integer, nullable=False)
    paragraph_index: Mapped[int] = mapped_column(Integer, nullable=False)
    sentence_index: Mapped[int] = mapped_column(Integer, nullable=False)
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    input_snapshot: Mapped[dict[str, Any]] = mapped_column(json_type, nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(64), nullable=False)
    llm_model: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    verdict: Mapped[str | None] = mapped_column(String(16))
    omitted_evidence_ids: Mapped[list[str]] = mapped_column(json_type, nullable=False, default=list)
    impact_types_json: Mapped[list[dict[str, str]]] = mapped_column(
        json_type, nullable=False, default=list, server_default=text("'[]'"))
    rationale: Mapped[str | None] = mapped_column(Text)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failure_code: Mapped[str | None] = mapped_column(String(64))
    failure_detail: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
