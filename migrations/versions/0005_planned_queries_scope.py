"""Add planned query provenance and paper language for scoped discovery.

Revision ID: 0005_planned_queries_scope
Revises: 0004_research_plans
Create Date: 2026-09-23
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0005_planned_queries_scope"
down_revision: str | None = "0004_research_plans"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("papers", sa.Column("language", sa.String(length=32), nullable=True))
    op.create_table(
        "planned_queries",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("plan_id", sa.Uuid(), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("query_key", sa.Text(), nullable=False),
        sa.Column("purpose", sa.String(length=32), nullable=False),
        sa.Column("variant", sa.String(length=32), nullable=False),
        sa.Column("origins", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("generation_version", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["research_runs.id"], name="fk_planned_queries_run_id_research_runs", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["plan_id"], ["research_plans.id"], name="fk_planned_queries_plan_id_research_plans", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_planned_queries"),
        sa.UniqueConstraint("run_id", "query_key", name="uq_planned_queries_run_query_key"),
    )
    op.add_column("search_queries", sa.Column("planned_query_id", sa.Uuid(), nullable=True))
    op.add_column("search_queries", sa.Column("returned_count", sa.Integer(), server_default="0", nullable=False))
    op.add_column("search_queries", sa.Column("skipped_count", sa.Integer(), server_default="0", nullable=False))
    op.add_column("search_queries", sa.Column("scope_filtered_count", sa.Integer(), server_default="0", nullable=False))
    op.create_foreign_key(
        "fk_search_queries_planned_query_id_planned_queries",
        "search_queries", "planned_queries", ["planned_query_id"], ["id"], ondelete="CASCADE"
    )
    op.create_unique_constraint(
        "uq_search_queries_planned_source", "search_queries", ["planned_query_id", "source"]
    )


def downgrade() -> None:
    op.drop_constraint("uq_search_queries_planned_source", "search_queries", type_="unique")
    op.drop_constraint(
        "fk_search_queries_planned_query_id_planned_queries", "search_queries", type_="foreignkey"
    )
    op.drop_column("search_queries", "planned_query_id")
    op.drop_column("search_queries", "scope_filtered_count")
    op.drop_column("search_queries", "skipped_count")
    op.drop_column("search_queries", "returned_count")
    op.drop_table("planned_queries")
    op.drop_column("papers", "language")
