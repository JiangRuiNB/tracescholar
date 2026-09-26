"""Preserve every source record even when papers deduplicate.

Revision ID: 0003_search_result_hit_ids
Revises: 0002_discovery_models
Create Date: 2026-09-23
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0003_search_result_hit_ids"
down_revision: str | None = "0002_discovery_models"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("search_results", sa.Column("id", sa.Uuid(), nullable=True))
    op.execute("UPDATE search_results SET id = gen_random_uuid()")
    op.alter_column("search_results", "id", nullable=False)
    op.drop_constraint("pk_search_results", "search_results", type_="primary")
    op.create_primary_key("pk_search_results", "search_results", ["id"])
    op.create_index(
        "uq_search_results_query_source_record",
        "search_results",
        ["search_query_id", "source_record_id"],
        unique=True,
    )


def downgrade() -> None:
    # The old composite key cannot represent multiple source records for one
    # paper. Refuse to discard provenance during a downgrade.
    op.execute(
        "DO $$ BEGIN "
        "IF EXISTS (SELECT 1 FROM search_results "
        "GROUP BY search_query_id, paper_id HAVING count(*) > 1) THEN "
        "RAISE EXCEPTION 'Cannot downgrade: duplicate source hits would be lost'; "
        "END IF; END $$;"
    )
    op.drop_index("uq_search_results_query_source_record", table_name="search_results")
    op.drop_constraint("pk_search_results", "search_results", type_="primary")
    op.create_primary_key(
        "pk_search_results", "search_results", ["search_query_id", "paper_id"]
    )
    op.drop_column("search_results", "id")
