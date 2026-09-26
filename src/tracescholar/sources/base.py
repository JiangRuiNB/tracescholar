"""Provider-neutral search results consumed by the discovery workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol


class SourceError(RuntimeError):
    """A paper source could not provide a usable search response."""


class UnsupportedScopeError(ValueError):
    """A frozen scope cannot be enforced by the current discovery layer."""


@dataclass(frozen=True, slots=True)
class SearchScope:
    """Validated hard constraints shared by every paper source."""

    year_from: int | None = None
    year_to: int | None = None
    languages: tuple[str, ...] = ()

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, Any]) -> SearchScope:
        allowed = {"year_from", "year_to", "language", "languages"}
        unsupported = set(snapshot) - allowed
        if unsupported:
            raise UnsupportedScopeError(
                f"Unsupported frozen scope keys: {', '.join(sorted(unsupported))}."
            )
        for name in ("year_from", "year_to"):
            year = snapshot.get(name)
            if year is not None and (type(year) is not int or not 1000 <= year <= 9999):
                raise UnsupportedScopeError(f"{name} must be a four-digit year.")
        year_from = snapshot.get("year_from")
        year_to = snapshot.get("year_to")
        if year_from is not None and year_to is not None and year_from > year_to:
            raise UnsupportedScopeError("year_from must not exceed year_to.")

        raw_languages = snapshot.get("languages")
        single_language = snapshot.get("language")
        if raw_languages is not None and single_language is not None:
            raise UnsupportedScopeError("Use either language or languages, not both.")
        if single_language is not None:
            raw_languages = [single_language]
        if raw_languages is None:
            languages: tuple[str, ...] = ()
        elif (
            not isinstance(raw_languages, list)
            or not raw_languages
            or any(not isinstance(item, str) or not item.strip() for item in raw_languages)
        ):
            raise UnsupportedScopeError("languages must be a non-empty list of language codes.")
        else:
            languages = tuple(dict.fromkeys(item.strip().casefold() for item in raw_languages))
            if any(not 2 <= len(language) <= 8 or not language.replace("-", "").isalpha()
                   for language in languages):
                raise UnsupportedScopeError("languages must contain valid language codes.")
        return cls(year_from=year_from, year_to=year_to, languages=languages)

    def to_dict(self) -> dict[str, Any]:
        return {
            "year_from": self.year_from,
            "year_to": self.year_to,
            "languages": list(self.languages),
        }

    def matches(self, paper: PaperMetadata) -> bool:
        if self.year_from is not None and (paper.year is None or paper.year < self.year_from):
            return False
        if self.year_to is not None and (paper.year is None or paper.year > self.year_to):
            return False
        if self.languages:
            if paper.language is None:
                return False
            normalized = paper.language.casefold().split("-", 1)[0]
            if normalized not in {language.split("-", 1)[0] for language in self.languages}:
                return False
        return True


@dataclass(frozen=True, slots=True)
class PaperMetadata:
    """Paper fields shared by every academic metadata source."""

    title: str
    doi: str | None = None
    arxiv_id: str | None = None
    year: int | None = None
    venue: str | None = None
    authors: tuple[str, ...] = ()
    abstract: str | None = None
    language: str | None = None


@dataclass(frozen=True, slots=True)
class PaperHit:
    """One provider result with source provenance and normalized metadata."""

    paper: PaperMetadata
    source_record_id: str
    source_url: str | None = None


@dataclass(frozen=True, slots=True)
class SearchBatch:
    """One completed query against one source."""

    query: str
    source: str
    results: tuple[PaperHit, ...]
    returned_count: int
    skipped_count: int
    executed_at: datetime
    filters: dict[str, Any] = field(default_factory=dict)
    total_count: int | None = None
    scope_filtered_count: int = 0


class PaperSource(Protocol):
    """Contract shared by OpenAlex, Crossref, and future adapters."""

    name: str

    def search(
        self, query: str, *, limit: int = 20, scope: SearchScope | None = None
    ) -> SearchBatch:
        """Execute a keyword search and return provider-neutral paper hits."""
