"""Persist immutable, content-addressed RunManifest JSON snapshots.

Revision ID: 0020_run_manifests
Revises: 0019_citation_sentence_audits
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0020_run_manifests"
down_revision: Union[str, Sequence[str], None] = "0019_citation_sentence_audits"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    json_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
    op.create_table(
        "run_manifests",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("manifest_version", sa.Integer(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("manifest_json", json_type, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["research_runs.id"],
                                name=op.f("fk_run_manifests_run_id_research_runs"),
                                ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_run_manifests")),
        sa.UniqueConstraint("run_id", "content_hash",
                            name="uq_run_manifests_run_content_hash"),
    )
    op.create_index("ix_run_manifests_run_created", "run_manifests",
                    ["run_id", "created_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_run_manifests_run_created", table_name="run_manifests")
    op.drop_table("run_manifests")
