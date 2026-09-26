"""Persist auditable title/abstract screening decisions.

Revision ID: 0006_screening_results
Revises: 0005_planned_queries_scope
Create Date: 2026-09-23
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0006_screening_results"
down_revision: str | None = "0005_planned_queries_scope"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "screening_results",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("paper_id", sa.Uuid(), nullable=False),
        sa.Column("plan_id", sa.Uuid(), nullable=False),
        sa.Column("label", sa.String(length=16), nullable=False),
        sa.Column("relevance_score", sa.Float(), nullable=False),
        sa.Column("rationale", sa.Text(), nullable=False),
        sa.Column("matched_inclusion_criteria", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("matched_exclusion_criteria", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("needs_full_text", sa.Boolean(), nullable=False),
        sa.Column("sub_question_index", sa.Integer(), nullable=True),
        sa.Column("evidence_role", sa.String(length=32), nullable=False),
        sa.Column("input_hash", sa.String(length=64), nullable=False),
        sa.Column("input_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("prompt_version", sa.String(length=64), nullable=False),
        sa.Column("llm_model", sa.String(length=128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("label IN ('include', 'maybe', 'exclude')", name="ck_screening_results_label_valid"),
        sa.CheckConstraint("relevance_score >= 0 AND relevance_score <= 1", name="ck_screening_results_score_range"),
        sa.ForeignKeyConstraint(["run_id"], ["research_runs.id"], name="fk_screening_results_run_id_research_runs", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.id"], name="fk_screening_results_paper_id_papers", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["plan_id"], ["research_plans.id"], name="fk_screening_results_plan_id_research_plans", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_screening_results"),
        sa.UniqueConstraint("run_id", "paper_id", name="uq_screening_results_run_paper"),
    )


def downgrade() -> None:
    op.drop_table("screening_results")
