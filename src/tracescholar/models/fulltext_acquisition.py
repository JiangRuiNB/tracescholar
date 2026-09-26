"""Per-run full-text acquisition state, including unavailable and failed outcomes."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from tracescholar.database.base import Base

if TYPE_CHECKING:
    from tracescholar.models.paper import Paper
    from tracescholar.models.paper_version import PaperVersion
    from tracescholar.models.research_run import ResearchRun


class FullTextAcquisition(Base):
    """Latest acquisition outcome for a screened paper in one research run."""

    __tablename__ = "fulltext_acquisitions"
    __table_args__ = (
        UniqueConstraint("run_id", "paper_id", name="uq_fulltext_acquisitions_run_paper"),
        CheckConstraint(
            "status IN ('located', 'downloaded', 'cached', 'unavailable', 'failed')",
            name="status_valid",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("research_runs.id", ondelete="CASCADE"), nullable=False
    )
    paper_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False
    )
    paper_version_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("paper_versions.id", ondelete="SET NULL")
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    source_url: Mapped[str | None] = mapped_column(Text)
    source_name: Mapped[str | None] = mapped_column(String(128))
    license: Mapped[str | None] = mapped_column(String(255))
    failure_code: Mapped[str | None] = mapped_column(String(64))
    failure_detail: Mapped[str | None] = mapped_column(Text)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    attempted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    retry_after: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    research_run: Mapped[ResearchRun] = relationship(back_populates="fulltext_acquisitions")
    paper: Mapped[Paper] = relationship(back_populates="fulltext_acquisitions")
    paper_version: Mapped[PaperVersion | None] = relationship(back_populates="acquisitions")
