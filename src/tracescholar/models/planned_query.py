"""Bounded generated query inventory linked to executed provider searches."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, DateTime, ForeignKey, String, Text, UniqueConstraint, Uuid, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from tracescholar.database.base import Base

if TYPE_CHECKING:
    from tracescholar.models.research_plan import ResearchPlanRecord
    from tracescholar.models.research_run import ResearchRun
    from tracescholar.models.search_query import SearchQuery


json_type = JSON().with_variant(JSONB(), "postgresql")


class PlannedQuery(Base):
    """One generated query, deduplicated within a run before provider calls."""

    __tablename__ = "planned_queries"
    __table_args__ = (UniqueConstraint("run_id", "query_key", name="uq_planned_queries_run_query_key"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("research_runs.id", ondelete="CASCADE"), nullable=False
    )
    plan_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("research_plans.id", ondelete="CASCADE"), nullable=False
    )
    query: Mapped[str] = mapped_column(Text, nullable=False)
    query_key: Mapped[str] = mapped_column(Text, nullable=False)
    purpose: Mapped[str] = mapped_column(String(32), nullable=False)
    variant: Mapped[str] = mapped_column(String(32), nullable=False)
    origins: Mapped[list[dict[str, Any]]] = mapped_column(json_type, nullable=False)
    generation_version: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    research_run: Mapped[ResearchRun] = relationship(back_populates="planned_queries")
    research_plan: Mapped[ResearchPlanRecord] = relationship(back_populates="planned_queries")
    executions: Mapped[list[SearchQuery]] = relationship(back_populates="planned_query")
