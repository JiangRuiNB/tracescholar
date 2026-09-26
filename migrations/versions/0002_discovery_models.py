"""Add search queries, canonical papers, and search result provenance.

Revision ID: 0002_discovery_models
Revises: 0001_research_runs
Create Date: 2026-09-23
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0002_discovery_models"
down_revision: str | None = "0001_research_runs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "papers",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("normalized_title", sa.Text(), nullable=False),
        sa.Column("doi", sa.Text(), nullable=True),
        sa.Column("arxiv_id", sa.String(length=64), nullable=True),
        sa.Column("year", sa.Integer(), nullable=True),
        sa.Column("venue", sa.Text(), nullable=True),
        sa.Column("authors", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("abstract", sa.Text(), nullable=True),
        sa.CheckConstraint("length(trim(title)) > 0", name="ck_papers_title_not_blank"),
        sa.CheckConstraint(
            "year IS NULL OR year BETWEEN 1000 AND 9999",
            name="ck_papers_year_range",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_papers"),
        sa.UniqueConstraint("doi", name="uq_papers_doi"),
        sa.UniqueConstraint("arxiv_id", name="uq_papers_arxiv_id"),
    )
    op.create_index(
        "ix_papers_normalized_title_year",
        "papers",
        ["normalized_title", "year"],
        unique=False,
    )

    op.create_table(
        "search_queries",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("source", sa.String(length=64), nullable=False),
        sa.Column("filters", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("executed_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("length(trim(query)) > 0", name="ck_search_queries_query_not_blank"),
        sa.CheckConstraint("length(trim(source)) > 0", name="ck_search_queries_source_not_blank"),
        sa.ForeignKeyConstraint(
            ["run_id"], ["research_runs.id"], name="fk_search_queries_run_id_research_runs", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_search_queries"),
    )
    op.create_index(
        "ix_search_queries_run_executed_at",
        "search_queries",
        ["run_id", "executed_at"],
        unique=False,
    )

    op.create_table(
        "search_results",
        sa.Column("search_query_id", sa.Uuid(), nullable=False),
        sa.Column("paper_id", sa.Uuid(), nullable=False),
        sa.Column("source_record_id", sa.String(length=255), nullable=True),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column("discovered_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(
            ["search_query_id"],
            ["search_queries.id"],
            name="fk_search_results_search_query_id_search_queries",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["paper_id"], ["papers.id"], name="fk_search_results_paper_id_papers", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("search_query_id", "paper_id", name="pk_search_results"),
    )
    op.create_index("ix_search_results_paper_id", "search_results", ["paper_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_search_results_paper_id", table_name="search_results")
    op.drop_table("search_results")
    op.drop_index("ix_search_queries_run_executed_at", table_name="search_queries")
    op.drop_table("search_queries")
    op.drop_index("ix_papers_normalized_title_year", table_name="papers")
    op.drop_table("papers")
