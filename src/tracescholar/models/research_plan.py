"""Frozen, versioned ResearchPlan attached one-to-one to a ResearchRun."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, Uuid, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from tracescholar.database.base import Base

if TYPE_CHECKING:
    from tracescholar.models.planned_query import PlannedQuery
    from tracescholar.models.research_run import ResearchRun
    from tracescholar.models.screening_result import ScreeningResult
    from tracescholar.models.fulltext_screening import FullTextScreeningResult


json_type = JSON().with_variant(JSONB(), "postgresql")


class ResearchPlanRecord(Base):
    """Stored plan plus inputs and versions needed to reproduce its scope."""

    __tablename__ = "research_plans"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("research_runs.id", ondelete="CASCADE"),
        nullable=False, unique=True,
    )
    plan_json: Mapped[dict[str, Any]] = mapped_column(json_type, nullable=False)
    input_question: Mapped[str] = mapped_column(Text, nullable=False)
    input_scope: Mapped[dict[str, Any]] = mapped_column(json_type, nullable=False)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(64), nullable=False)
    llm_model: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    research_run: Mapped[ResearchRun] = relationship(back_populates="research_plan")
    planned_queries: Mapped[list[PlannedQuery]] = relationship(back_populates="research_plan")
    screening_results: Mapped[list[ScreeningResult]] = relationship(back_populates="research_plan")
    fulltext_screening_results: Mapped[list[FullTextScreeningResult]] = relationship(
        back_populates="research_plan"
    )
