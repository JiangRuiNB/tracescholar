"""Immutable, content-addressed reproducibility manifests for ResearchRuns."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, Index, Integer, String, UniqueConstraint, Uuid, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from tracescholar.database.base import Base


json_type = JSON().with_variant(JSONB(), "postgresql")


class RunManifestRecord(Base):
    """One immutable JSON snapshot of the database facts for a ResearchRun."""

    __tablename__ = "run_manifests"
    __table_args__ = (
        UniqueConstraint("run_id", "content_hash", name="uq_run_manifests_run_content_hash"),
        Index("ix_run_manifests_run_created", "run_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("research_runs.id", ondelete="CASCADE"), nullable=False)
    manifest_version: Mapped[int] = mapped_column(Integer, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    manifest_json: Mapped[dict[str, Any]] = mapped_column(json_type, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now())
