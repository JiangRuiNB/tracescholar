"""Canonical studies, bibliographic versions, and PDF result comparisons."""

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
    from tracescholar.models.paper import Paper
    from tracescholar.models.paper_version import PaperVersion
    from tracescholar.models.research_run import ResearchRun


json_type = JSON().with_variant(JSONB(), "postgresql")


class CanonicalStudy(Base):
    """One independent research contribution, possibly represented by many Papers."""

    __tablename__ = "canonical_studies"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    canonical_paper_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("papers.id", ondelete="RESTRICT"), nullable=False, unique=True,
    )
    canonical_reason: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now(),
    )

    canonical_paper: Mapped[Paper] = relationship(foreign_keys=[canonical_paper_id])
    members: Mapped[list[StudyPaper]] = relationship(back_populates="study")
    run_selections: Mapped[list[StudyRunSelection]] = relationship(back_populates="study")
    version_comparisons: Mapped[list[StudyVersionComparison]] = relationship(back_populates="study")


class StudyPaper(Base):
    """Non-destructive membership of a source Paper record in one study."""

    __tablename__ = "study_papers"
    __table_args__ = (
        Index("ix_study_papers_study_role", "study_id", "publication_role"),
        CheckConstraint(
            "publication_role IN ('preprint', 'conference', 'journal', 'other')",
            name="publication_role_valid",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    study_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("canonical_studies.id", ondelete="CASCADE"), nullable=False,
    )
    paper_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False, unique=True,
    )
    publication_role: Mapped[str] = mapped_column(String(16), nullable=False)
    relationship_reason: Mapped[str] = mapped_column(Text, nullable=False)
    linked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )

    study: Mapped[CanonicalStudy] = relationship(back_populates="members")
    paper: Mapped[Paper] = relationship(back_populates="study_membership")


class StudyLinkCandidate(Base):
    """A confirmed or unresolved possible same-study relationship between Papers."""

    __tablename__ = "study_link_candidates"
    __table_args__ = (
        UniqueConstraint("paper_a_id", "paper_b_id", name="uq_study_link_candidate_pair"),
        Index("ix_study_link_candidates_status", "status"),
        CheckConstraint("paper_a_id <> paper_b_id", name="different_papers"),
        CheckConstraint("status IN ('confirmed', 'unresolved', 'rejected')", name="status_valid"),
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="confidence_range"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    paper_a_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False,
    )
    paper_b_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False,
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    match_reason: Mapped[str] = mapped_column(Text, nullable=False)
    signals: Mapped[dict[str, Any]] = mapped_column(json_type, nullable=False)
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    rule_version: Mapped[str] = mapped_column(String(64), nullable=False)
    decision_source: Mapped[str] = mapped_column(String(16), nullable=False, default="rule")
    evaluated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )

    paper_a: Mapped[Paper] = relationship(foreign_keys=[paper_a_id])
    paper_b: Mapped[Paper] = relationship(foreign_keys=[paper_b_id])


class StudyRunSelection(Base):
    """Reproducible default PDF choice for extraction within one ResearchRun."""

    __tablename__ = "study_run_selections"
    __table_args__ = (
        UniqueConstraint("run_id", "study_id", name="uq_study_run_selection"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("research_runs.id", ondelete="CASCADE"), nullable=False,
    )
    study_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("canonical_studies.id", ondelete="CASCADE"), nullable=False,
    )
    preferred_paper_version_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("paper_versions.id", ondelete="SET NULL"),
    )
    selection_reason: Mapped[str] = mapped_column(Text, nullable=False)
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    policy_version: Mapped[str] = mapped_column(String(64), nullable=False)
    selected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )

    research_run: Mapped[ResearchRun] = relationship(back_populates="study_selections")
    study: Mapped[CanonicalStudy] = relationship(back_populates="run_selections")
    preferred_version: Mapped[PaperVersion | None] = relationship(foreign_keys=[preferred_paper_version_id])


class StudyVersionComparison(Base):
    """Whether two preserved PDFs have identical or changed experimental results."""

    __tablename__ = "study_version_comparisons"
    __table_args__ = (
        UniqueConstraint("study_id", "version_a_id", "version_b_id", name="uq_study_version_pair"),
        CheckConstraint("version_a_id <> version_b_id", name="different_versions"),
        CheckConstraint(
            "result_relation IN ('identical_content', 'not_assessed', 'equivalent', 'changed')",
            name="result_relation_valid",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    study_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("canonical_studies.id", ondelete="CASCADE"), nullable=False,
    )
    version_a_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("paper_versions.id", ondelete="CASCADE"), nullable=False,
    )
    version_b_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("paper_versions.id", ondelete="CASCADE"), nullable=False,
    )
    result_relation: Mapped[str] = mapped_column(String(24), nullable=False)
    assessment_source: Mapped[str] = mapped_column(String(16), nullable=False)
    assessment_note: Mapped[str] = mapped_column(Text, nullable=False)
    assessed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )

    study: Mapped[CanonicalStudy] = relationship(back_populates="version_comparisons")
    version_a: Mapped[PaperVersion] = relationship(foreign_keys=[version_a_id])
    version_b: Mapped[PaperVersion] = relationship(foreign_keys=[version_b_id])
