"""Auditable second-stage screening with chunk-level evidence.

Revision ID: 0010_fulltext_screening
Revises: 0009_chunk_embeddings
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0010_fulltext_screening"
down_revision: str | None = "0009_chunk_embeddings"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "fulltext_screening_results",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("run_id", sa.Uuid(), sa.ForeignKey("research_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("paper_id", sa.Uuid(), sa.ForeignKey("papers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("paper_version_id", sa.Uuid(), sa.ForeignKey("paper_versions.id", ondelete="CASCADE"), nullable=True),
        sa.Column("plan_id", sa.Uuid(), sa.ForeignKey("research_plans.id", ondelete="CASCADE"), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("label", sa.String(16)),
        sa.Column("rationale", sa.Text()),
        sa.Column("matched_inclusion_criteria", postgresql.JSONB(), nullable=False),
        sa.Column("matched_exclusion_criteria", postgresql.JSONB(), nullable=False),
        sa.Column("supported_sub_question_indices", postgresql.JSONB(), nullable=False),
        sa.Column("evidence_role", sa.String(40)),
        sa.Column("quality_warnings", postgresql.JSONB(), nullable=False),
        sa.Column("input_hash", sa.String(64), nullable=False),
        sa.Column("input_snapshot", postgresql.JSONB(), nullable=False),
        sa.Column("prompt_version", sa.String(64), nullable=False),
        sa.Column("llm_model", sa.String(128), nullable=False),
        sa.Column("retrieval_model_revision", sa.String(64), nullable=False),
        sa.Column("failure_code", sa.String(64)),
        sa.Column("failure_detail", sa.Text()),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("attempted_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("status IN ('success', 'failed')", name="ck_fulltext_screening_results_status_valid"),
        sa.CheckConstraint("label IS NULL OR label IN ('include', 'exclude', 'uncertain')",
                           name="ck_fulltext_screening_results_label_valid"),
        sa.UniqueConstraint("run_id", "paper_id", name="uq_fulltext_screening_run_paper"),
    )
    op.create_index("ix_fulltext_screening_run_status", "fulltext_screening_results", ["run_id", "status"])
    op.create_table(
        "fulltext_screening_evidence",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("screening_result_id", sa.Uuid(), sa.ForeignKey("fulltext_screening_results.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_id", sa.Uuid(), sa.ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("similarity", sa.Float(), nullable=False),
        sa.Column("retrieval_sub_question_indices", postgresql.JSONB(), nullable=False),
        sa.CheckConstraint("rank >= 1", name="ck_fulltext_screening_evidence_rank_positive"),
        sa.CheckConstraint("similarity >= -1 AND similarity <= 1", name="ck_fulltext_screening_evidence_similarity_range"),
        sa.UniqueConstraint("screening_result_id", "chunk_id", name="uq_fulltext_screening_evidence_chunk"),
    )


def downgrade() -> None:
    op.drop_table("fulltext_screening_evidence")
    op.drop_index("ix_fulltext_screening_run_status", table_name="fulltext_screening_results")
    op.drop_table("fulltext_screening_results")
