"""Durable history for one-stage-at-a-time ResearchRun orchestration."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    JSON, CheckConstraint, DateTime, Float, ForeignKey, Index, Integer,
    String, Text, UniqueConstraint, Uuid, func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from tracescholar.database.base import Base

if TYPE_CHECKING:
    from tracescholar.models.research_run import ResearchRun


json_type = JSON().with_variant(JSONB(), "postgresql")


class WorkflowStageExecution(Base):
    """One attempt to execute a single workflow stage for a ResearchRun."""

    __tablename__ = "workflow_stage_executions"
    __table_args__ = (
        UniqueConstraint("run_id", "stage", "attempt", name="uq_workflow_stage_attempt"),
        CheckConstraint(
            "status IN ('running', 'completed', 'failed', 'blocked')",
            name="status_valid",
        ),
        CheckConstraint("attempt >= 1", name="attempt_positive"),
        Index("ix_workflow_stage_run_stage_started", "run_id", "stage", "started_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("research_runs.id", ondelete="CASCADE"), nullable=False,
    )
    stage: Mapped[str] = mapped_column(String(32), nullable=False)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    result_json: Mapped[dict[str, Any] | None] = mapped_column(json_type)
    failure_type: Mapped[str | None] = mapped_column(String(128))
    failure_reason: Mapped[str | None] = mapped_column(Text)

    research_run: Mapped[ResearchRun] = relationship(back_populates="workflow_executions")
