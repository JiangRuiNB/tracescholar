"""Strict output for one sentence's semantic citation assessment."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SemanticCitationJudgment(BaseModel):
    """A narrowly scoped judgment; a revision is only a saved suggestion."""

    model_config = ConfigDict(extra="forbid", strict=True)

    verdict: Literal["pass", "revise", "reject"]
    entailment: Literal["entailed", "partial", "unsupported"]
    scope: Literal["aligned", "too_broad", "mismatched"]
    strength: Literal["calibrated", "overstated", "understated"]
    rationale: str = Field(min_length=1, max_length=1000)
    minimal_revision: str | None = Field(default=None, max_length=1200)

    @field_validator("rationale", "minimal_revision")
    @classmethod
    def strip_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Audit text must not be blank")
        return cleaned

    @model_validator(mode="after")
    def consistent_decision(self) -> Self:
        if self.verdict == "pass":
            if (self.entailment, self.scope, self.strength) != (
                "entailed", "aligned", "calibrated") or self.minimal_revision is not None:
                raise ValueError("pass requires all checks to pass and no revision")
        elif self.verdict == "revise":
            if self.entailment == "unsupported" or self.scope == "mismatched":
                raise ValueError("unsupported or mismatched evidence requires reject")
            if self.minimal_revision is None or "\n" in self.minimal_revision:
                raise ValueError("revise requires one nonblank replacement sentence")
        elif self.minimal_revision is not None:
            raise ValueError("reject must not invent replacement text")
        return self
