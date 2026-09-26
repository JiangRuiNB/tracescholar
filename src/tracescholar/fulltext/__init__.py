"""Legal open-access PDF location, fetching, and version persistence."""

from tracescholar.fulltext.fetcher import DocumentFetcher, FetchError, FetchedPDF
from tracescholar.fulltext.locator import (
    FullTextLocation, FullTextPaper, LocateError, OpenAlexOALocator,
)
from tracescholar.fulltext.service import AcquisitionSummary, acquire_fulltext

__all__ = [
    "AcquisitionSummary", "DocumentFetcher", "FetchError", "FetchedPDF",
    "FullTextLocation", "FullTextPaper", "LocateError", "OpenAlexOALocator",
    "acquire_fulltext",
]
