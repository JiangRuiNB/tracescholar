"""Immutable deterministic audits of saved synthesis citation chains."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, CheckConstraint, DateTime, ForeignKey, Index, Integer, String, Uuid, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from tracescholar.database.base import Base


json_type = JSON().with_variant(JSONB(), "postgresql")


class CitationAudit(Base):
    """One reproducible, read-only chain check against a draft and DB snapshot."""

    __tablename__ = "citation_audits"
    __table_args__ = (
        UniqueConstraint("draft_id", "auditor_version", "input_hash",
                         name="uq_citation_audits_draft_version_input"),
        CheckConstraint("status IN ('passed', 'failed')", name="status_valid"),
        Index("ix_citation_audits_run_checked", "run_id", "checked_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("research_runs.id", ondelete="CASCADE"), nullable=False)
    draft_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("synthesis_drafts.id", ondelete="CASCADE"), nullable=False)
    auditor_version: Mapped[str] = mapped_column(String(64), nullable=False)
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    sentence_count: Mapped[int] = mapped_column(Integer, nullable=False)
    citation_count: Mapped[int] = mapped_column(Integer, nullable=False)
    unique_evidence_count: Mapped[int] = mapped_column(Integer, nullable=False)
    issues_json: Mapped[list[dict[str, Any]]] = mapped_column(json_type, nullable=False)
    checked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now())
