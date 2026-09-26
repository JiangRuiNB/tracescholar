"""Strict LLM contracts; grounding is independently verified in service code."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvidenceValidationError(ValueError):
    """The proposed claim or quote cannot be validated against stored PDF text."""


class StrictOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ClaimDraft(StrictOutput):
    sub_question_index: int = Field(ge=0)
    statement: str
    scope_kind: Literal["study_specific", "cross_study"]
    basis_chunk_id: str
    basis_quote: str

    @model_validator(mode="after")
    def check_text(self) -> Self:
        if len(self.statement.strip()) < 12 or len(self.statement) > 500:
            raise ValueError("Claim must be an atomic 12–500 character assertion")
        if len(self.basis_quote.strip()) < 12 or len(self.basis_quote) > 600:
            raise ValueError("Claim basis quote must contain 12–600 characters")
        return self


class ClaimDraftBatch(StrictOutput):
    claims: list[ClaimDraft] = Field(max_length=2)
    no_claim_reason: str


class EvidenceDraftSpan(StrictOutput):
    chunk_id: str
    quote: str
    stance: Literal["supports", "contradicts", "qualifies", "unrelated"]
    confidence: float = Field(ge=0, le=1)
    rationale: str
    study_context: str
    limitations: str

    @model_validator(mode="after")
    def check_span(self) -> Self:
        if len(self.quote.strip()) < 12 or len(self.quote) > 700:
            raise ValueError("Evidence quote must contain 12–700 exact characters")
        if not self.rationale.strip():
            raise ValueError("Evidence rationale must not be blank")
        return self


class EvidenceDecision(StrictOutput):
    no_evidence: bool
    no_evidence_reason: str
    spans: list[EvidenceDraftSpan] = Field(max_length=2)

    @model_validator(mode="after")
    def check_decision(self) -> Self:
        if self.no_evidence:
            if self.spans or not self.no_evidence_reason.strip():
                raise ValueError("No-evidence decisions need a reason and zero spans")
        elif not self.spans:
            raise ValueError("An evidence decision needs one or two spans")
        elif all(span.stance == "unrelated" for span in self.spans):
            raise ValueError("Unrelated-only passages must be reported as no_evidence")
        return self
