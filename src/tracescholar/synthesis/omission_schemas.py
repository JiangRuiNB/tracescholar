"""Fixed decision contract for one sentence's omitted-counterevidence check."""

from __future__ import annotations

import uuid
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EvidenceImpact(BaseModel):
    """One uncited EvidenceSpan that materially affects this sentence."""

    model_config = ConfigDict(extra="forbid", strict=True)

    evidence_id: uuid.UUID
    impact_type: Literal["contradicts", "limits", "weakens", "uncertain"]


class OmissionJudgment(BaseModel):
    """Identify uncited evidence whose content materially affects the sentence."""

    model_config = ConfigDict(extra="forbid", strict=True)

    verdict: Literal["pass", "revise", "flag"]
    impacts: list[EvidenceImpact] = Field(default_factory=list)
    rationale: str = Field(min_length=1, max_length=1000)

    @field_validator("impacts")
    @classmethod
    def unique_evidence(cls, value: list[EvidenceImpact]) -> list[EvidenceImpact]:
        ids = [item.evidence_id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("Each Evidence ID may have only one impact type")
        return value

    @field_validator("rationale")
    @classmethod
    def strip_rationale(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Rationale must not be blank")
        return cleaned

    @model_validator(mode="after")
    def consistent_verdict(self) -> Self:
        if self.verdict == "pass" and self.impacts:
            raise ValueError("pass must not report material omitted evidence")
        if self.verdict in {"revise", "flag"} and not self.impacts:
            raise ValueError("revise/flag require one or more evidence impacts")
        if self.verdict == "revise" and any(
                item.impact_type == "uncertain" for item in self.impacts):
            raise ValueError("uncertain impact requires flag")
        return self
