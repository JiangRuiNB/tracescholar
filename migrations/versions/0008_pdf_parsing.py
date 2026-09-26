"""Store page-aware PDF parsing outcomes and exact-locator chunks.

Revision ID: 0008_pdf_parsing
Revises: 0007_fulltext_acquisition
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


revision: str = "0008_pdf_parsing"
down_revision: str | None = "0007_fulltext_acquisition"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "parsed_pages",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("paper_version_id", sa.Uuid(), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("char_count", sa.Integer(), nullable=False),
        sa.Column("width", sa.Float(), nullable=False),
        sa.Column("height", sa.Float(), nullable=False),
        sa.Column("column_count", sa.Integer(), nullable=False),
        sa.Column("quality_flags", JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("page_number >= 1", name="ck_parsed_pages_page_number_positive"),
        sa.ForeignKeyConstraint(["paper_version_id"], ["paper_versions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("paper_version_id", "page_number", name="uq_parsed_pages_version_page"),
    )
    op.create_table(
        "chunks",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("paper_version_id", sa.Uuid(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("page_start", sa.Integer(), nullable=False),
        sa.Column("page_end", sa.Integer(), nullable=False),
        sa.Column("section", sa.Text(), nullable=False),
        sa.Column("document_char_start", sa.Integer(), nullable=False),
        sa.Column("document_char_end", sa.Integer(), nullable=False),
        sa.Column("locator", JSONB(), nullable=False),
        sa.Column("char_count", sa.Integer(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("parser_version", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("page_start >= 1 AND page_end >= page_start", name="ck_chunks_page_range_valid"),
        sa.CheckConstraint("char_count > 0", name="ck_chunks_char_count_positive"),
        sa.CheckConstraint("document_char_end > document_char_start", name="ck_chunks_document_range_valid"),
        sa.ForeignKeyConstraint(["paper_version_id"], ["paper_versions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("paper_version_id", "ordinal", name="uq_chunks_version_ordinal"),
    )
    op.create_index("ix_chunks_version_page", "chunks", ["paper_version_id", "page_start"])
    op.create_table(
        "pdf_parse_records",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("paper_version_id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("parser_version", sa.String(length=64), nullable=False),
        sa.Column("input_hash", sa.String(length=64), nullable=False),
        sa.Column("page_count", sa.Integer(), nullable=False),
        sa.Column("text_page_count", sa.Integer(), nullable=False),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column("total_char_count", sa.Integer(), nullable=False),
        sa.Column("quality_flags", JSONB(), nullable=False),
        sa.Column("failure_code", sa.String(length=64), nullable=True),
        sa.Column("failure_detail", sa.Text(), nullable=True),
        sa.Column("attempted_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("status IN ('success', 'failed')", name="ck_pdf_parse_records_status_valid"),
        sa.ForeignKeyConstraint(["paper_version_id"], ["paper_versions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("paper_version_id"),
    )


def downgrade() -> None:
    op.drop_table("pdf_parse_records")
    op.drop_index("ix_chunks_version_page", table_name="chunks")
    op.drop_table("chunks")
    op.drop_table("parsed_pages")
