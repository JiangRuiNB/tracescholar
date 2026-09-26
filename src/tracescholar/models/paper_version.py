"""Immutable PDF content version associated with a canonical paper."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from tracescholar.database.base import Base

if TYPE_CHECKING:
    from tracescholar.models.chunk import Chunk
    from tracescholar.models.paper import Paper
    from tracescholar.models.fulltext_acquisition import FullTextAcquisition
    from tracescholar.models.fulltext_screening import FullTextScreeningResult
    from tracescholar.models.parsed_page import ParsedPage
    from tracescholar.models.pdf_parse_record import PdfParseRecord


class PaperVersion(Base):
    """A paper-specific reference to a content-addressed local PDF."""

    __tablename__ = "paper_versions"
    __table_args__ = (
        UniqueConstraint("paper_id", "content_hash", name="uq_paper_versions_paper_hash"),
        Index("ix_paper_versions_content_hash", "content_hash"),
        Index("ix_paper_versions_source_url", "source_url"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    paper_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False
    )
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    storage_path: Mapped[str] = mapped_column(Text, nullable=False)
    content_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    source_name: Mapped[str] = mapped_column(String(128), nullable=False)
    license: Mapped[str | None] = mapped_column(String(255))
    version_label: Mapped[str | None] = mapped_column(String(64))
    retrieved_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    paper: Mapped[Paper] = relationship(back_populates="versions")
    acquisitions: Mapped[list[FullTextAcquisition]] = relationship(back_populates="paper_version")
    parsed_pages: Mapped[list[ParsedPage]] = relationship(
        back_populates="paper_version", cascade="all, delete-orphan"
    )
    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="paper_version", cascade="all, delete-orphan"
    )
    parse_record: Mapped[PdfParseRecord | None] = relationship(
        back_populates="paper_version", uselist=False, cascade="all, delete-orphan"
    )
    fulltext_screening_results: Mapped[list[FullTextScreeningResult]] = relationship(
        back_populates="paper_version"
    )
