"""Persist one-stage-at-a-time workflow attempts.

Revision ID: 0021_workflow_stage_executions
Revises: 0020_run_manifests
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0021_workflow_stage_executions"
down_revision: Union[str, Sequence[str], None] = "0020_run_manifests"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    json_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
    op.create_table(
        "workflow_stage_executions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("stage", sa.String(length=32), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column("result_json", json_type, nullable=True),
        sa.Column("failure_type", sa.String(length=128), nullable=True),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.CheckConstraint("attempt >= 1", name="ck_workflow_stage_executions_attempt_positive"),
        sa.CheckConstraint(
            "status IN ('running', 'completed', 'failed', 'blocked')",
            name="ck_workflow_stage_executions_status_valid",
        ),
        sa.ForeignKeyConstraint(["run_id"], ["research_runs.id"],
                                ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "stage", "attempt",
                            name="uq_workflow_stage_attempt"),
    )
    op.create_index(
        "ix_workflow_stage_run_stage_started", "workflow_stage_executions",
        ["run_id", "stage", "started_at"], unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_workflow_stage_run_stage_started", table_name="workflow_stage_executions")
    op.drop_table("workflow_stage_executions")
