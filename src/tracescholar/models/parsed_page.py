"""Normalized page text whose offsets are referenced by chunk locators."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import JSON, CheckConstraint, DateTime, Float, ForeignKey, Integer, Text, UniqueConstraint, Uuid, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from tracescholar.database.base import Base

if TYPE_CHECKING:
    from tracescholar.models.paper_version import PaperVersion


json_type = JSON().with_variant(JSONB(), "postgresql")


class ParsedPage(Base):
    """One physical PDF page; text may be empty for a blank/reference page."""

    __tablename__ = "parsed_pages"
    __table_args__ = (
        UniqueConstraint("paper_version_id", "page_number", name="uq_parsed_pages_version_page"),
        CheckConstraint("page_number >= 1", name="page_number_positive"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    paper_version_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("paper_versions.id", ondelete="CASCADE"), nullable=False
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False)
    width: Mapped[float] = mapped_column(Float, nullable=False)
    height: Mapped[float] = mapped_column(Float, nullable=False)
    column_count: Mapped[int] = mapped_column(Integer, nullable=False)
    quality_flags: Mapped[list[str]] = mapped_column(json_type, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    paper_version: Mapped[PaperVersion] = relationship(back_populates="parsed_pages")
