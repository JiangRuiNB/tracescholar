"""Source-independent discovery workflows."""

from tracescholar.discovery.planned import (
    QueryCandidate,
    QueryGenerationError,
    PlannedDiscoverySummary,
    generate_search_queries,
    run_planned_discovery,
)
from tracescholar.discovery.service import (
    DiscoverySummary,
    SearchSummary,
    SourceFailure,
    discover_papers,
    search_and_persist,
)

__all__ = [
    "QueryCandidate",
    "QueryGenerationError",
    "PlannedDiscoverySummary",
    "generate_search_queries",
    "run_planned_discovery",
    "DiscoverySummary",
    "SearchSummary",
    "SourceFailure",
    "discover_papers",
    "search_and_persist",
]
