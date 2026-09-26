"""Page-local, section-aware text chunks with verifiable character locators."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, CheckConstraint, DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint, Uuid, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from tracescholar.database.base import Base

if TYPE_CHECKING:
    from tracescholar.models.chunk_embedding import ChunkEmbedding
    from tracescholar.models.fulltext_screening import FullTextScreeningEvidence
    from tracescholar.models.paper_version import PaperVersion


json_type = JSON().with_variant(JSONB(), "postgresql")


class Chunk(Base):
    """Exact substring of one ParsedPage, ordered across the PDF body."""

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("paper_version_id", "ordinal", name="uq_chunks_version_ordinal"),
        Index("ix_chunks_version_page", "paper_version_id", "page_start"),
        CheckConstraint("page_start >= 1 AND page_end >= page_start", name="page_range_valid"),
        CheckConstraint("char_count > 0", name="char_count_positive"),
        CheckConstraint("document_char_end > document_char_start", name="document_range_valid"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    paper_version_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("paper_versions.id", ondelete="CASCADE"), nullable=False
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    page_start: Mapped[int] = mapped_column(Integer, nullable=False)
    page_end: Mapped[int] = mapped_column(Integer, nullable=False)
    section: Mapped[str] = mapped_column(Text, nullable=False)
    document_char_start: Mapped[int] = mapped_column(Integer, nullable=False)
    document_char_end: Mapped[int] = mapped_column(Integer, nullable=False)
    locator: Mapped[dict[str, Any]] = mapped_column(json_type, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    parser_version: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    paper_version: Mapped[PaperVersion] = relationship(back_populates="chunks")
    embeddings: Mapped[list[ChunkEmbedding]] = relationship(
        back_populates="chunk", cascade="all, delete-orphan",
    )
    fulltext_screening_evidence: Mapped[list[FullTextScreeningEvidence]] = relationship(
        back_populates="chunk", cascade="all, delete-orphan",
    )
