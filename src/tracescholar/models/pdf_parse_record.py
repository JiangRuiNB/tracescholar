"""Durable success/failure state for parsing one immutable PDF version."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import JSON, CheckConstraint, DateTime, ForeignKey, Integer, String, Text, Uuid, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from tracescholar.database.base import Base

if TYPE_CHECKING:
    from tracescholar.models.paper_version import PaperVersion


json_type = JSON().with_variant(JSONB(), "postgresql")


class PdfParseRecord(Base):
    """One current parser outcome per PaperVersion; failures remain retryable."""

    __tablename__ = "pdf_parse_records"
    __table_args__ = (
        CheckConstraint("status IN ('success', 'failed')", name="status_valid"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    paper_version_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("paper_versions.id", ondelete="CASCADE"),
        nullable=False, unique=True,
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    parser_version: Mapped[str] = mapped_column(String(64), nullable=False)
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    page_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    text_page_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_char_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    quality_flags: Mapped[list[str]] = mapped_column(json_type, nullable=False, default=list)
    failure_code: Mapped[str | None] = mapped_column(String(64))
    failure_detail: Mapped[str | None] = mapped_column(Text)
    attempted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    paper_version: Mapped[PaperVersion] = relationship(back_populates="parse_record")
