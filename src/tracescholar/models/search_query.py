"""Executed search queries and their paper hits."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, CheckConstraint, DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint, Uuid, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from tracescholar.database.base import Base

if TYPE_CHECKING:
    from tracescholar.models.paper import Paper
    from tracescholar.models.planned_query import PlannedQuery
    from tracescholar.models.research_run import ResearchRun


json_type = JSON().with_variant(JSONB(), "postgresql")


class SearchQuery(Base):
    """A search expression executed against a named metadata source."""

    __tablename__ = "search_queries"
    __table_args__ = (
        CheckConstraint("length(trim(query)) > 0", name="query_not_blank"),
        CheckConstraint("length(trim(source)) > 0", name="source_not_blank"),
        Index("ix_search_queries_run_executed_at", "run_id", "executed_at"),
        UniqueConstraint("planned_query_id", "source", name="uq_search_queries_planned_source"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("research_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    planned_query_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("planned_queries.id", ondelete="CASCADE")
    )
    query: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    filters: Mapped[dict[str, Any]] = mapped_column(json_type, nullable=False, default=dict)
    returned_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    skipped_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    scope_filtered_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    executed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    research_run: Mapped[ResearchRun] = relationship(back_populates="search_queries")
    planned_query: Mapped[PlannedQuery | None] = relationship(back_populates="executions")
    results: Mapped[list[SearchResult]] = relationship(
        back_populates="search_query",
        cascade="all, delete-orphan",
    )


class SearchResult(Base):
    """One provider record returned by a query, linked to a canonical paper."""

    __tablename__ = "search_results"
    __table_args__ = (
        Index("ix_search_results_paper_id", "paper_id"),
        Index(
            "uq_search_results_query_source_record",
            "search_query_id",
            "source_record_id",
            unique=True,
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    search_query_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("search_queries.id", ondelete="CASCADE"),
        nullable=False,
    )
    paper_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("papers.id", ondelete="CASCADE"),
        nullable=False,
    )
    source_record_id: Mapped[str | None] = mapped_column(String(255))
    source_url: Mapped[str | None] = mapped_column(Text)
    discovered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    search_query: Mapped[SearchQuery] = relationship(back_populates="results")
    paper: Mapped[Paper] = relationship(back_populates="search_results")
