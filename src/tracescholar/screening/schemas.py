"""Strict, provider-independent decision contract for title/abstract screening."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tracescholar.planning.schemas import ResearchPlan


class ScreeningValidationError(ValueError):
    """A model decision does not refer to the frozen plan consistently."""


class ScreeningDecision(BaseModel):
    """One paper's first-stage, high-recall decision."""

    model_config = ConfigDict(extra="forbid", strict=True)

    label: Literal["include", "maybe", "exclude"]
    relevance_score: float = Field(ge=0, le=1)
    rationale: str
    matched_inclusion_indices: list[int]
    matched_exclusion_indices: list[int]
    needs_full_text: bool
    sub_question_index: int | None
    evidence_role: Literal[
        "method", "outcome", "comparison", "limitation", "counter_evidence",
        "background", "other",
    ]

    @model_validator(mode="after")
    def check_local_consistency(self) -> Self:
        if not self.rationale.strip():
            raise ValueError("Screening rationale must not be blank.")
        for field_name in ("matched_inclusion_indices", "matched_exclusion_indices"):
            indices = getattr(self, field_name)
            if any(index < 0 for index in indices) or len(indices) != len(set(indices)):
                raise ValueError(f"{field_name} must contain unique non-negative indices.")
        if self.sub_question_index is not None and self.sub_question_index < 0:
            raise ValueError("sub_question_index must be non-negative or null.")
        return self

    def validate_against_plan(self, plan: ResearchPlan) -> None:
        if any(index >= len(plan.inclusion_criteria) for index in self.matched_inclusion_indices):
            raise ScreeningValidationError("Decision cites an unknown inclusion criterion.")
        if any(index >= len(plan.exclusion_criteria) for index in self.matched_exclusion_indices):
            raise ScreeningValidationError("Decision cites an unknown exclusion criterion.")
        if self.sub_question_index is not None and self.sub_question_index >= len(plan.sub_questions):
            raise ScreeningValidationError("Decision cites an unknown sub-question.")
