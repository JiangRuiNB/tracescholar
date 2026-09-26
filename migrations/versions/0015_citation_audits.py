"""Persist deterministic synthesis citation-chain audits.

Revision ID: 0015_citation_audits
Revises: 0014_synthesis_drafts
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0015_citation_audits"
down_revision: Union[str, Sequence[str], None] = "0014_synthesis_drafts"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    json_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
    op.create_table(
        "citation_audits",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("draft_id", sa.Uuid(), nullable=False),
        sa.Column("auditor_version", sa.String(length=64), nullable=False),
        sa.Column("input_hash", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("sentence_count", sa.Integer(), nullable=False),
        sa.Column("citation_count", sa.Integer(), nullable=False),
        sa.Column("unique_evidence_count", sa.Integer(), nullable=False),
        sa.Column("issues_json", json_type, nullable=False),
        sa.Column("checked_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("status IN ('passed', 'failed')", name=op.f("ck_citation_audits_status_valid")),
        sa.ForeignKeyConstraint(["run_id"], ["research_runs.id"],
                                name=op.f("fk_citation_audits_run_id_research_runs"), ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["draft_id"], ["synthesis_drafts.id"],
                                name=op.f("fk_citation_audits_draft_id_synthesis_drafts"), ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_citation_audits")),
        sa.UniqueConstraint("draft_id", "auditor_version", "input_hash",
                            name="uq_citation_audits_draft_version_input"),
    )
    op.create_index("ix_citation_audits_run_checked", "citation_audits",
                    ["run_id", "checked_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_citation_audits_run_checked", table_name="citation_audits")
    op.drop_table("citation_audits")
