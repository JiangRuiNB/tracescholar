"""Persist versioned structured synthesis drafts.

Revision ID: 0014_synthesis_drafts
Revises: 0013_claim_scopes
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0014_synthesis_drafts"
down_revision: Union[str, Sequence[str], None] = "0013_claim_scopes"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    json_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
    op.create_table(
        "synthesis_drafts",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("claim_generation_id", sa.Uuid(), nullable=False),
        sa.Column("input_hash", sa.String(length=64), nullable=False),
        sa.Column("input_snapshot", json_type, nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("prompt_version", sa.String(length=64), nullable=False),
        sa.Column("llm_model", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("document_json", json_type, nullable=True),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("failure_code", sa.String(length=64), nullable=True),
        sa.Column("failure_detail", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("status IN ('success', 'failed')", name=op.f("ck_synthesis_drafts_status_valid")),
        sa.ForeignKeyConstraint(["run_id"], ["research_runs.id"],
                                name=op.f("fk_synthesis_drafts_run_id_research_runs"), ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["claim_generation_id"], ["claim_generations.id"],
                                name=op.f("fk_synthesis_drafts_claim_generation_id_claim_generations"),
                                ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_synthesis_drafts")),
        sa.UniqueConstraint("run_id", "input_hash", name="uq_synthesis_drafts_run_input"),
    )
    op.create_index("ix_synthesis_drafts_run_created", "synthesis_drafts",
                    ["run_id", "created_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_synthesis_drafts_run_created", table_name="synthesis_drafts")
    op.drop_table("synthesis_drafts")
