"""Canonical scientific paper metadata."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import JSON, CheckConstraint, Index, Integer, String, Text, Uuid, join
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from tracescholar.database.base import Base

if TYPE_CHECKING:
    from tracescholar.models.fulltext_acquisition import FullTextAcquisition
    from tracescholar.models.fulltext_screening import FullTextScreeningResult
    from tracescholar.models.paper_version import PaperVersion
    from tracescholar.models.research_run import ResearchRun
    from tracescholar.models.screening_result import ScreeningResult
    from tracescholar.models.study import StudyPaper

from tracescholar.models.search_query import SearchQuery, SearchResult


authors_type = JSON().with_variant(JSONB(), "postgresql")


def _research_run_secondary_join():
    from tracescholar.models.research_run import ResearchRun

    return ResearchRun.id == SearchQuery.run_id


class Paper(Base):
    """A canonical paper shared across research runs and search providers."""

    __tablename__ = "papers"
    __table_args__ = (
        CheckConstraint("length(trim(title)) > 0", name="title_not_blank"),
        CheckConstraint("year IS NULL OR year BETWEEN 1000 AND 9999", name="year_range"),
        Index("ix_papers_normalized_title_year", "normalized_title", "year"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_title: Mapped[str] = mapped_column(Text, nullable=False)
    doi: Mapped[str | None] = mapped_column(Text, unique=True)
    arxiv_id: Mapped[str | None] = mapped_column(String(64), unique=True)
    year: Mapped[int | None] = mapped_column(Integer)
    venue: Mapped[str | None] = mapped_column(Text)
    authors: Mapped[list[str]] = mapped_column(authors_type, nullable=False, default=list)
    abstract: Mapped[str | None] = mapped_column(Text)
    language: Mapped[str | None] = mapped_column(String(32))

    search_results: Mapped[list[SearchResult]] = relationship(back_populates="paper")
    screening_results: Mapped[list[ScreeningResult]] = relationship(back_populates="paper")
    versions: Mapped[list[PaperVersion]] = relationship(back_populates="paper")
    fulltext_acquisitions: Mapped[list[FullTextAcquisition]] = relationship(back_populates="paper")
    fulltext_screening_results: Mapped[list[FullTextScreeningResult]] = relationship(back_populates="paper")
    study_membership: Mapped[StudyPaper | None] = relationship(back_populates="paper", uselist=False)
    research_runs: Mapped[list[ResearchRun]] = relationship(
        secondary=lambda: join(
            SearchQuery.__table__,
            SearchResult.__table__,
            SearchQuery.id == SearchResult.search_query_id,
        ),
        primaryjoin=lambda: Paper.id == SearchResult.paper_id,
        secondaryjoin=_research_run_secondary_join,
        viewonly=True,
    )
