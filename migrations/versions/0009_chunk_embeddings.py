"""Versioned pgvector embeddings for parsed chunks.

Revision ID: 0009_chunk_embeddings
Revises: 0008_pdf_parsing
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import VECTOR


revision: str = "0009_chunk_embeddings"
down_revision: str | None = "0008_pdf_parsing"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "chunk_embeddings",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("chunk_id", sa.Uuid(), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("model_revision", sa.String(length=64), nullable=False),
        sa.Column("source_revision", sa.String(length=255), nullable=False),
        sa.Column("endpoint_url", sa.Text(), nullable=True),
        sa.Column("encoder_version", sa.String(length=32), nullable=False),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column("input_hash", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("vector", VECTOR(), nullable=True),
        sa.Column("failure_code", sa.String(length=64), nullable=True),
        sa.Column("failure_detail", sa.Text(), nullable=True),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("attempted_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("status IN ('success', 'failed')", name="ck_chunk_embeddings_status_valid"),
        sa.CheckConstraint("dimensions BETWEEN 1 AND 2000", name="ck_chunk_embeddings_dimensions_valid"),
        sa.ForeignKeyConstraint(["chunk_id"], ["chunks.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("chunk_id", "provider", "model_name", "model_revision",
                            "encoder_version", name="uq_chunk_embeddings_identity"),
    )
    op.create_index("ix_chunk_embeddings_model_status", "chunk_embeddings",
                    ["provider", "model_name", "model_revision", "encoder_version", "status"])


def downgrade() -> None:
    op.drop_index("ix_chunk_embeddings_model_status", table_name="chunk_embeddings")
    op.drop_table("chunk_embeddings")
