"""Add frozen, versioned research plans.

Revision ID: 0004_research_plans
Revises: 0003_search_result_hit_ids
Create Date: 2026-09-23
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0004_research_plans"
down_revision: str | None = "0003_search_result_hit_ids"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "research_plans",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("plan_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("input_question", sa.Text(), nullable=False),
        sa.Column("input_scope", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("prompt_version", sa.String(length=64), nullable=False),
        sa.Column("llm_model", sa.String(length=128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["research_runs.id"], name="fk_research_plans_run_id_research_runs", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_research_plans"),
        sa.UniqueConstraint("run_id", name="uq_research_plans_run_id"),
    )


def downgrade() -> None:
    op.drop_table("research_plans")
