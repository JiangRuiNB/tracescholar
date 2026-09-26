"""Persist sentence-level omitted-counterevidence decisions.

Revision ID: 0017_omission_audits
Revises: 0016_semantic_citation_audits
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0017_omission_audits"
down_revision: Union[str, Sequence[str], None] = "0016_semantic_citation_audits"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    json_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
    op.create_table(
        "omission_audits",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("draft_id", sa.Uuid(), nullable=False),
        sa.Column("citation_audit_id", sa.Uuid(), nullable=False),
        sa.Column("section_index", sa.Integer(), nullable=False),
        sa.Column("paragraph_index", sa.Integer(), nullable=False),
        sa.Column("sentence_index", sa.Integer(), nullable=False),
        sa.Column("input_hash", sa.String(length=64), nullable=False),
        sa.Column("input_snapshot", json_type, nullable=False),
        sa.Column("prompt_version", sa.String(length=64), nullable=False),
        sa.Column("llm_model", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("verdict", sa.String(length=16), nullable=True),
        sa.Column("omitted_evidence_ids", json_type, nullable=False),
        sa.Column("rationale", sa.Text(), nullable=True),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("failure_code", sa.String(length=64), nullable=True),
        sa.Column("failure_detail", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("status IN ('success', 'failed')", name=op.f("ck_omission_audits_status_valid")),
        sa.CheckConstraint("verdict IS NULL OR verdict IN ('pass', 'revise', 'flag')",
                           name=op.f("ck_omission_audits_verdict_valid")),
        sa.CheckConstraint("section_index >= 0 AND paragraph_index >= 0 AND sentence_index >= 0",
                           name=op.f("ck_omission_audits_indices_nonnegative")),
        sa.ForeignKeyConstraint(["run_id"], ["research_runs.id"],
                                name=op.f("fk_omission_audits_run_id_research_runs"), ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["draft_id"], ["synthesis_drafts.id"],
                                name=op.f("fk_omission_audits_draft_id_synthesis_drafts"), ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["citation_audit_id"], ["citation_audits.id"],
                                name=op.f("fk_omission_audits_citation_audit_id_citation_audits"), ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_omission_audits")),
        sa.UniqueConstraint("draft_id", "section_index", "paragraph_index", "sentence_index",
                            "input_hash", name="uq_omission_audits_sentence_input"),
    )
    op.create_index("ix_omission_audits_run_draft", "omission_audits",
                    ["run_id", "draft_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_omission_audits_run_draft", table_name="omission_audits")
    op.drop_table("omission_audits")
