"""Persist deterministic citation audits per synthesis sentence.

Revision ID: 0019_citation_sentence_audits
Revises: 0018_omission_impact_types
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0019_citation_sentence_audits"
down_revision: Union[str, Sequence[str], None] = "0018_omission_impact_types"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    json_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
    op.create_table(
        "citation_sentence_audits",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("draft_id", sa.Uuid(), nullable=False),
        sa.Column("auditor_version", sa.String(length=64), nullable=False),
        sa.Column("section_index", sa.Integer(), nullable=False),
        sa.Column("paragraph_index", sa.Integer(), nullable=False),
        sa.Column("sentence_index", sa.Integer(), nullable=False),
        sa.Column("input_hash", sa.String(length=64), nullable=False),
        sa.Column("input_snapshot", json_type, nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("issues_json", json_type, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("status IN ('passed', 'failed')",
                           name=op.f("ck_citation_sentence_audits_status_valid")),
        sa.CheckConstraint(
            "section_index >= 0 AND paragraph_index >= 0 AND sentence_index >= 0",
            name=op.f("ck_citation_sentence_audits_indices_nonnegative")),
        sa.ForeignKeyConstraint(["run_id"], ["research_runs.id"],
                                name=op.f("fk_citation_sentence_audits_run_id_research_runs"),
                                ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["draft_id"], ["synthesis_drafts.id"],
                                name=op.f("fk_citation_sentence_audits_draft_id_synthesis_drafts"),
                                ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_citation_sentence_audits")),
        sa.UniqueConstraint(
            "draft_id", "auditor_version", "section_index", "paragraph_index",
            "sentence_index", "input_hash", name="uq_citation_sentence_audit_input"),
    )
    op.create_index("ix_citation_sentence_audits_run_draft",
                    "citation_sentence_audits", ["run_id", "draft_id"], unique=False)
    op.create_table(
        "citation_audit_sentence_links",
        sa.Column("citation_audit_id", sa.Uuid(), nullable=False),
        sa.Column("sentence_audit_id", sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(["citation_audit_id"], ["citation_audits.id"],
                                name=op.f("fk_citation_audit_sentence_links_citation_audit_id_citation_audits"),
                                ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["sentence_audit_id"], ["citation_sentence_audits.id"],
                                name=op.f("fk_citation_audit_sentence_links_sentence_audit_id_citation_sentence_audits"),
                                ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("citation_audit_id", "sentence_audit_id",
                                name=op.f("pk_citation_audit_sentence_links")),
    )
    op.create_index("ix_citation_audit_sentence_links_sentence",
                    "citation_audit_sentence_links", ["sentence_audit_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_citation_audit_sentence_links_sentence",
                  table_name="citation_audit_sentence_links")
    op.drop_table("citation_audit_sentence_links")
    op.drop_index("ix_citation_sentence_audits_run_draft",
                  table_name="citation_sentence_audits")
    op.drop_table("citation_sentence_audits")
