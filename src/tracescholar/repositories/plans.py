"""Persistence operations for immutable research plans."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy.orm import Session

from tracescholar.models import ResearchPlanRecord, ResearchRun

if TYPE_CHECKING:
    from tracescholar.planning.schemas import ResearchPlan


class PlanFrozenError(ValueError):
    """The run input changed after its plan was frozen."""


def load_research_plan(session: Session, run_id: uuid.UUID) -> ResearchPlan | None:
    """Reload a stored plan through the same fixed schema used by callers."""
    from tracescholar.planning.schemas import ResearchPlan

    run = session.get(ResearchRun, run_id)
    if run is None:
        raise LookupError(f"ResearchRun {run_id} does not exist.")
    record = run.research_plan
    if record is None:
        return None
    if record.schema_version != 1:
        raise PlanFrozenError(f"Unsupported ResearchPlan schema version {record.schema_version}.")
    if record.input_question != run.question or record.input_scope != run.scope:
        raise PlanFrozenError("ResearchRun question or scope changed after planning.")
    return ResearchPlan.model_validate(record.plan_json)


def save_research_plan(
    session: Session,
    *,
    run: ResearchRun,
    plan: ResearchPlan,
    model_name: str,
    prompt_version: str,
) -> ResearchPlanRecord:
    """Store one plan and the exact inputs that produced it."""
    if run.research_plan is not None:
        raise PlanFrozenError("ResearchRun already has a frozen ResearchPlan.")
    if plan.scope_snapshot != run.scope:
        raise ValueError("ResearchPlan scope_snapshot must match ResearchRun scope.")
    record = ResearchPlanRecord(
        run_id=run.id,
        plan_json=plan.model_dump(mode="json"),
        input_question=run.question,
        input_scope=run.scope,
        schema_version=1,
        prompt_version=prompt_version,
        llm_model=model_name,
    )
    session.add(record)
    session.flush()
    return record
