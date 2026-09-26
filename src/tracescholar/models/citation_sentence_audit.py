"""Immutable per-sentence deterministic citation-chain audit records."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON, CheckConstraint, DateTime, ForeignKey, Index, Integer, String,
    UniqueConstraint, Uuid, func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from tracescholar.database.base import Base


json_type = JSON().with_variant(JSONB(), "postgresql")


class CitationSentenceAudit(Base):
    """One content-addressed citation-chain check for a draft sentence."""

    __tablename__ = "citation_sentence_audits"
    __table_args__ = (
        UniqueConstraint(
            "draft_id", "auditor_version", "section_index", "paragraph_index",
            "sentence_index", "input_hash", name="uq_citation_sentence_audit_input",
        ),
        CheckConstraint("status IN ('passed', 'failed')", name="status_valid"),
        CheckConstraint(
            "section_index >= 0 AND paragraph_index >= 0 AND sentence_index >= 0",
            name="indices_nonnegative",
        ),
        Index("ix_citation_sentence_audits_run_draft", "run_id", "draft_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("research_runs.id", ondelete="CASCADE"), nullable=False)
    draft_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("synthesis_drafts.id", ondelete="CASCADE"), nullable=False)
    auditor_version: Mapped[str] = mapped_column(String(64), nullable=False)
    section_index: Mapped[int] = mapped_column(Integer, nullable=False)
    paragraph_index: Mapped[int] = mapped_column(Integer, nullable=False)
    sentence_index: Mapped[int] = mapped_column(Integer, nullable=False)
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    input_snapshot: Mapped[dict[str, Any]] = mapped_column(json_type, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    issues_json: Mapped[list[dict[str, Any]]] = mapped_column(json_type, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now())


class CitationAuditSentenceLink(Base):
    """The exact per-sentence outcomes composing one draft-level summary."""

    __tablename__ = "citation_audit_sentence_links"
    __table_args__ = (
        Index("ix_citation_audit_sentence_links_sentence", "sentence_audit_id"),
    )

    citation_audit_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("citation_audits.id", ondelete="CASCADE"),
        primary_key=True)
    sentence_audit_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("citation_sentence_audits.id", ondelete="RESTRICT"),
        primary_key=True)
