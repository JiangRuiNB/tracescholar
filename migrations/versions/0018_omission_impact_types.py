"""Persist the sentence-level effect of each omitted EvidenceSpan.

Revision ID: 0018_omission_impact_types
Revises: 0017_omission_audits
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0018_omission_impact_types"
down_revision: Union[str, Sequence[str], None] = "0017_omission_audits"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    json_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
    op.add_column(
        "omission_audits",
        sa.Column("impact_types_json", json_type, nullable=False,
                  server_default=sa.text("'[]'")),
    )


def downgrade() -> None:
    op.drop_column("omission_audits", "impact_types_json")
