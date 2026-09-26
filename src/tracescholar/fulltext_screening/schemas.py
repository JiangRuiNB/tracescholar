"""Strict full-text decision contract; chunk citations are checked against retrieval."""

from __future__ import annotations

import uuid
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from tracescholar.planning.schemas import ResearchPlan


class FullTextScreeningValidationError(ValueError):
    """The model cited nonexistent evidence or contradicted the frozen plan."""


class FullTextDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    label: Literal["include", "exclude", "uncertain"]
    rationale: str
    matched_inclusion_indices: list[int]
    matched_exclusion_indices: list[int]
    supported_sub_question_indices: list[int]
    evidence_role: Literal[
        "primary_empirical_evidence", "review_background", "methods",
        "benchmark_dataset", "other",
    ]
    evidence_chunk_ids: list[str]

    @model_validator(mode="after")
    def check_local_consistency(self) -> Self:
        if not self.rationale.strip():
            raise ValueError("Full-text screening rationale must not be blank.")
        for name in (
            "matched_inclusion_indices", "matched_exclusion_indices",
            "supported_sub_question_indices",
        ):
            values = getattr(self, name)
            if any(value < 0 for value in values) or len(values) != len(set(values)):
                raise ValueError(f"{name} must contain unique non-negative indices.")
        if len(self.evidence_chunk_ids) != len(set(self.evidence_chunk_ids)):
            raise ValueError("Evidence chunk IDs must not repeat.")
        if self.label in {"include", "exclude"} and not self.evidence_chunk_ids:
            raise ValueError("A definitive decision must cite at least one retrieved Chunk.")
        if self.label == "include" and not self.supported_sub_question_indices:
            raise ValueError("An included paper must support at least one sub-question.")
        if self.label == "exclude" and not self.matched_exclusion_indices:
            raise ValueError("Exclusion requires an explicit plan exclusion criterion.")
        return self

    def validate_against(
        self, plan: ResearchPlan, candidate_chunk_ids: set[uuid.UUID],
    ) -> list[uuid.UUID]:
        if any(index >= len(plan.inclusion_criteria) for index in self.matched_inclusion_indices):
            raise FullTextScreeningValidationError("Unknown inclusion criterion index.")
        if any(index >= len(plan.exclusion_criteria) for index in self.matched_exclusion_indices):
            raise FullTextScreeningValidationError("Unknown exclusion criterion index.")
        if any(index >= len(plan.sub_questions) for index in self.supported_sub_question_indices):
            raise FullTextScreeningValidationError("Unknown sub-question index.")
        try:
            cited = [uuid.UUID(value) for value in self.evidence_chunk_ids]
        except (TypeError, ValueError) as error:
            raise FullTextScreeningValidationError("Invalid Chunk ID in decision.") from error
        if len(cited) != len(set(cited)) or any(chunk_id not in candidate_chunk_ids for chunk_id in cited):
            raise FullTextScreeningValidationError("Decision cited a Chunk outside retrieved evidence.")
        return cited
