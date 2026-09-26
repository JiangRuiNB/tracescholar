"""Versioned, structured synthesis drafts for one research run."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, CheckConstraint, DateTime, ForeignKey, Index, Integer, String, Text, Uuid, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from tracescholar.database.base import Base


json_type = JSON().with_variant(JSONB(), "postgresql")


class SynthesisDraft(Base):
    """One reproducible Writer attempt and its validated SynthesisDocument."""

    __tablename__ = "synthesis_drafts"
    __table_args__ = (
        UniqueConstraint("run_id", "input_hash", name="uq_synthesis_drafts_run_input"),
        CheckConstraint("status IN ('success', 'failed')", name="status_valid"),
        Index("ix_synthesis_drafts_run_created", "run_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("research_runs.id", ondelete="CASCADE"), nullable=False)
    claim_generation_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("claim_generations.id", ondelete="CASCADE"), nullable=False)
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    input_snapshot: Mapped[dict[str, Any]] = mapped_column(json_type, nullable=False)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    prompt_version: Mapped[str] = mapped_column(String(64), nullable=False)
    llm_model: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    document_json: Mapped[dict[str, Any] | None] = mapped_column(json_type)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failure_code: Mapped[str | None] = mapped_column(String(64))
    failure_detail: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
