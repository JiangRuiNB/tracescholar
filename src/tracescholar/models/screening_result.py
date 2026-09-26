"""Auditable title/abstract screening decision for one run and paper."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    JSON, Boolean, CheckConstraint, DateTime, Float, ForeignKey, Integer,
    String, Text, UniqueConstraint, Uuid, func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from tracescholar.database.base import Base

if TYPE_CHECKING:
    from tracescholar.models.paper import Paper
    from tracescholar.models.research_plan import ResearchPlanRecord
    from tracescholar.models.research_run import ResearchRun


json_type = JSON().with_variant(JSONB(), "postgresql")


class ScreeningResult(Base):
    """Current reproducible decision; excluded papers remain queryable."""

    __tablename__ = "screening_results"
    __table_args__ = (
        UniqueConstraint("run_id", "paper_id", name="uq_screening_results_run_paper"),
        CheckConstraint("label IN ('include', 'maybe', 'exclude')", name="label_valid"),
        CheckConstraint("relevance_score >= 0 AND relevance_score <= 1", name="score_range"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("research_runs.id", ondelete="CASCADE"), nullable=False
    )
    paper_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False
    )
    plan_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("research_plans.id", ondelete="CASCADE"), nullable=False
    )
    label: Mapped[str] = mapped_column(String(16), nullable=False)
    relevance_score: Mapped[float] = mapped_column(Float, nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    matched_inclusion_criteria: Mapped[list[str]] = mapped_column(json_type, nullable=False)
    matched_exclusion_criteria: Mapped[list[str]] = mapped_column(json_type, nullable=False)
    needs_full_text: Mapped[bool] = mapped_column(Boolean, nullable=False)
    sub_question_index: Mapped[int | None] = mapped_column(Integer)
    evidence_role: Mapped[str] = mapped_column(String(32), nullable=False)
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    input_snapshot: Mapped[dict[str, Any]] = mapped_column(json_type, nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(64), nullable=False)
    llm_model: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    research_run: Mapped[ResearchRun] = relationship(back_populates="screening_results")
    paper: Mapped[Paper] = relationship(back_populates="screening_results")
    research_plan: Mapped[ResearchPlanRecord] = relationship(back_populates="screening_results")
