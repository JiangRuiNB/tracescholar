"""Versioned, retryable vector representation of a PDF Chunk."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import JSON, CheckConstraint, DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from tracescholar.database.base import Base

if TYPE_CHECKING:
    from tracescholar.models.chunk import Chunk


vector_type = JSON().with_variant(VECTOR(), "postgresql")


class ChunkEmbedding(Base):
    """One embedding attempt/result per chunk and immutable encoder identity."""

    __tablename__ = "chunk_embeddings"
    __table_args__ = (
        UniqueConstraint("chunk_id", "provider", "model_name", "model_revision",
                         "encoder_version", name="uq_chunk_embeddings_identity"),
        Index("ix_chunk_embeddings_model_status", "provider", "model_name",
              "model_revision", "encoder_version", "status"),
        CheckConstraint("status IN ('success', 'failed')", name="status_valid"),
        CheckConstraint("dimensions BETWEEN 1 AND 2000", name="dimensions_valid"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False,
    )
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_revision: Mapped[str] = mapped_column(String(64), nullable=False)
    source_revision: Mapped[str] = mapped_column(String(255), nullable=False)
    endpoint_url: Mapped[str | None] = mapped_column(Text)
    encoder_version: Mapped[str] = mapped_column(String(32), nullable=False)
    dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    vector: Mapped[list[float] | None] = mapped_column(vector_type, nullable=True)
    failure_code: Mapped[str | None] = mapped_column(String(64))
    failure_detail: Mapped[str | None] = mapped_column(Text)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    attempted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )

    chunk: Mapped[Chunk] = relationship(back_populates="embeddings")
