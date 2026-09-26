"""Fixed, provider-independent schema for a future research synthesis."""

from __future__ import annotations

import uuid
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SYNTHESIS_SCHEMA_VERSION = 2


class SynthesisOutput(BaseModel):
    """Base contract: strict values and no unrecognized output fields."""

    model_config = ConfigDict(extra="forbid", strict=True)


def _nonblank(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("Synthesis text fields must not be blank.")
    return cleaned


def _unique_ids(values: list[uuid.UUID]) -> list[uuid.UUID]:
    if len(values) != len(set(values)):
        raise ValueError("Citation ID lists must not contain duplicates.")
    return values


class SynthesisSentence(SynthesisOutput):
    """One sentence with explicit Claim and EvidenceSpan provenance links."""

    text: str = Field(min_length=1)
    claim_ids: list[uuid.UUID] = Field(min_length=1)
    evidence_ids: list[uuid.UUID] = Field(min_length=1)

    _text = field_validator("text")(_nonblank)
    _claims_unique = field_validator("claim_ids")(_unique_ids)
    _evidence_unique = field_validator("evidence_ids")(_unique_ids)


class SynthesisParagraph(SynthesisOutput):
    """A paragraph is represented as ordered sentences, not a Markdown blob."""

    sentences: list[SynthesisSentence] = Field(min_length=1)


class SynthesisSection(SynthesisOutput):
    """A named section containing ordered paragraphs."""

    heading: str = Field(min_length=1)
    paragraphs: list[SynthesisParagraph] = Field(min_length=1)

    _heading = field_validator("heading")(_nonblank)


class SynthesisDocument(SynthesisOutput):
    """Structured synthesis content, before any presentation or rendering."""

    title: str = Field(min_length=1)
    research_question: str = Field(min_length=1)
    sections: list[SynthesisSection] = Field(min_length=1)

    _title = field_validator("title")(_nonblank)
    _research_question = field_validator("research_question")(_nonblank)

    @model_validator(mode="after")
    def check_section_headings(self) -> Self:
        headings = [section.heading.casefold() for section in self.sections]
        if len(headings) != len(set(headings)):
            raise ValueError("Synthesis section headings must be unique.")
        return self
