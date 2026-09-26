"""ResearchRun persistence model."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import JSON, CheckConstraint, DateTime, Enum, Index, Text, Uuid, func, join
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from tracescholar.database.base import Base
from tracescholar.models.paper import Paper
from tracescholar.models.fulltext_acquisition import FullTextAcquisition
from tracescholar.models.fulltext_screening import FullTextScreeningResult
from tracescholar.models.planned_query import PlannedQuery
from tracescholar.models.research_plan import ResearchPlanRecord
from tracescholar.models.search_query import SearchQuery, SearchResult
from tracescholar.models.screening_result import ScreeningResult
from tracescholar.models.study import StudyRunSelection
from tracescholar.models.workflow_execution import WorkflowStageExecution


class ResearchRunStatus(StrEnum):
    """Lifecycle states for a research run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


json_type = JSON().with_variant(JSONB(), "postgresql")


class ResearchRun(Base):
    """A durable record of one user-requested literature research run."""

    __tablename__ = "research_runs"
    __table_args__ = (
        CheckConstraint("length(trim(question)) > 0", name="question_not_blank"),
        Index("ix_research_runs_status_created_at", "status", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[ResearchRunStatus] = mapped_column(
        Enum(
            ResearchRunStatus,
            name="research_run_status",
            native_enum=False,
            create_constraint=True,
            values_callable=lambda enum: [item.value for item in enum],
        ),
        nullable=False,
        default=ResearchRunStatus.PENDING,
    )
    scope: Mapped[dict[str, Any]] = mapped_column(
        json_type,
        nullable=False,
        default=dict,
    )
    config_snapshot: Mapped[dict[str, Any]] = mapped_column(
        json_type,
        nullable=False,
        default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    search_queries: Mapped[list[SearchQuery]] = relationship(
        back_populates="research_run",
        cascade="all, delete-orphan",
    )
    research_plan: Mapped[ResearchPlanRecord | None] = relationship(
        back_populates="research_run",
        uselist=False,
        cascade="all, delete-orphan",
    )
    planned_queries: Mapped[list[PlannedQuery]] = relationship(
        back_populates="research_run", cascade="all, delete-orphan"
    )
    screening_results: Mapped[list[ScreeningResult]] = relationship(
        back_populates="research_run", cascade="all, delete-orphan"
    )
    fulltext_acquisitions: Mapped[list[FullTextAcquisition]] = relationship(
        back_populates="research_run", cascade="all, delete-orphan"
    )
    fulltext_screening_results: Mapped[list[FullTextScreeningResult]] = relationship(
        back_populates="research_run", cascade="all, delete-orphan"
    )
    workflow_executions: Mapped[list[WorkflowStageExecution]] = relationship(
        back_populates="research_run", cascade="all, delete-orphan",
        order_by="WorkflowStageExecution.started_at",
    )
    study_selections: Mapped[list[StudyRunSelection]] = relationship(
        back_populates="research_run", cascade="all, delete-orphan"
    )
    papers: Mapped[list[Paper]] = relationship(
        secondary=lambda: join(
            SearchQuery.__table__,
            SearchResult.__table__,
            SearchQuery.id == SearchResult.search_query_id,
        ),
        primaryjoin=lambda: ResearchRun.id == SearchQuery.run_id,
        secondaryjoin=lambda: Paper.id == SearchResult.paper_id,
        viewonly=True,
    )

    def __repr__(self) -> str:
        return f"ResearchRun(id={self.id!r}, status={self.status.value!r})"
