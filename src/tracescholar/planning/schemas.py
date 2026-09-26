"""Stable, validated scope-planning output consumed by later stages."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class PlanBase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


def _clean_text(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("Planning text fields must not be blank.")
    return cleaned


def _clean_list(values: list[str]) -> list[str]:
    cleaned = [_clean_text(value) for value in values]
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("Planning lists must not contain duplicates.")
    return cleaned


class Concept(PlanBase):
    term: str
    synonyms: list[str]
    abbreviations: list[str]

    _term = field_validator("term")(_clean_text)
    _synonyms = field_validator("synonyms")(_clean_list)
    _abbreviations = field_validator("abbreviations")(_clean_list)


class AmbiguityItem(PlanBase):
    item: str
    impact_on_scope: str
    conservative_assumption: str
    requires_clarification: bool

    _item = field_validator("item")(_clean_text)
    _impact = field_validator("impact_on_scope")(_clean_text)
    _assumption = field_validator("conservative_assumption")(_clean_text)


class SearchTrack(PlanBase):
    label: str
    intent: Literal["core", "method", "evaluation", "counter_evidence"]
    query: str
    rationale: str

    _label = field_validator("label")(_clean_text)
    _query = field_validator("query")(_clean_text)
    _rationale = field_validator("rationale")(_clean_text)


class ResearchPlanDraft(PlanBase):
    """Only the fields the LLM must generate under Structured Outputs."""

    normalized_question: str
    sub_questions: list[str]
    concepts: list[Concept]
    exclusion_terms: list[str]
    inclusion_criteria: list[str]
    exclusion_criteria: list[str]
    constraints: list[str]
    ambiguity_items: list[AmbiguityItem]
    search_tracks: list[SearchTrack]
    stop_conditions: list[str]

    _question = field_validator("normalized_question")(_clean_text)
    _sub_questions = field_validator("sub_questions")(_clean_list)
    _exclusion_terms = field_validator("exclusion_terms")(_clean_list)
    _inclusion = field_validator("inclusion_criteria")(_clean_list)
    _exclusion = field_validator("exclusion_criteria")(_clean_list)
    _constraints = field_validator("constraints")(_clean_list)
    _stop = field_validator("stop_conditions")(_clean_list)

    @model_validator(mode="after")
    def ensure_usable_plan(self) -> Self:
        if not self.sub_questions:
            raise ValueError("A research plan requires at least one sub-question.")
        if not self.concepts:
            raise ValueError("A research plan requires at least one concept.")
        if not self.inclusion_criteria or not self.exclusion_criteria:
            raise ValueError("A research plan requires inclusion and exclusion criteria.")
        if not self.search_tracks:
            raise ValueError("A research plan requires at least one search track.")
        if not any(track.intent == "counter_evidence" for track in self.search_tracks):
            raise ValueError("A research plan requires a counter-evidence search track.")
        if not self.stop_conditions:
            raise ValueError("A research plan requires stop conditions.")
        return self


class ResearchPlan(ResearchPlanDraft):
    """Frozen plan with the user's exact scope copied from ResearchRun."""

    scope_snapshot: dict[str, Any]
