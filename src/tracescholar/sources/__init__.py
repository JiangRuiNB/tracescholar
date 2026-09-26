"""Provider-neutral contracts and academic paper source adapters."""

from tracescholar.sources.base import (
    PaperHit, PaperMetadata, PaperSource, SearchBatch, SearchScope, SourceError,
    UnsupportedScopeError,
)
from tracescholar.sources.crossref import CrossrefSource
from tracescholar.sources.openalex import OpenAlexSource

__all__ = [
    "CrossrefSource",
    "OpenAlexSource",
    "PaperHit",
    "PaperMetadata",
    "PaperSource",
    "SearchBatch",
    "SearchScope",
    "SourceError",
    "UnsupportedScopeError",
]
