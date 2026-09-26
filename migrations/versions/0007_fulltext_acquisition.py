"""Track open-access acquisition and content-addressed paper versions.

Revision ID: 0007_fulltext_acquisition
Revises: 0006_screening_results
Create Date: 2026-09-23
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0007_fulltext_acquisition"
down_revision: str | None = "0006_screening_results"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "paper_versions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("paper_id", sa.Uuid(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("storage_path", sa.Text(), nullable=False),
        sa.Column("content_bytes", sa.Integer(), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("source_name", sa.String(length=128), nullable=False),
        sa.Column("license", sa.String(length=255), nullable=True),
        sa.Column("version_label", sa.String(length=64), nullable=True),
        sa.Column("retrieved_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.id"], name="fk_paper_versions_paper_id_papers", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_paper_versions"),
        sa.UniqueConstraint("paper_id", "content_hash", name="uq_paper_versions_paper_hash"),
    )
    op.create_index("ix_paper_versions_content_hash", "paper_versions", ["content_hash"])
    op.create_index("ix_paper_versions_source_url", "paper_versions", ["source_url"])
    op.create_table(
        "fulltext_acquisitions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("paper_id", sa.Uuid(), nullable=False),
        sa.Column("paper_version_id", sa.Uuid(), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column("source_name", sa.String(length=128), nullable=True),
        sa.Column("license", sa.String(length=255), nullable=True),
        sa.Column("failure_code", sa.String(length=64), nullable=True),
        sa.Column("failure_detail", sa.Text(), nullable=True),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("attempted_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("retry_after", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('located', 'downloaded', 'cached', 'unavailable', 'failed')",
            name="ck_fulltext_acquisitions_status_valid",
        ),
        sa.ForeignKeyConstraint(["run_id"], ["research_runs.id"], name="fk_fulltext_acquisitions_run_id_research_runs", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.id"], name="fk_fulltext_acquisitions_paper_id_papers", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["paper_version_id"], ["paper_versions.id"], name="fk_fulltext_acquisitions_paper_version_id_paper_versions", ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id", name="pk_fulltext_acquisitions"),
        sa.UniqueConstraint("run_id", "paper_id", name="uq_fulltext_acquisitions_run_paper"),
    )


def downgrade() -> None:
    op.drop_table("fulltext_acquisitions")
    op.drop_index("ix_paper_versions_source_url", table_name="paper_versions")
    op.drop_index("ix_paper_versions_content_hash", table_name="paper_versions")
    op.drop_table("paper_versions")
